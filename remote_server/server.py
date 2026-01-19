#!/usr/bin/env python3
"""
gRPC Video Streaming Server for Gaze Processing
Receives video frames, processes with MediaPipe FaceMesh, returns landmarks
"""

import grpc
from concurrent import futures
import time
import cv2
import numpy as np
import mediapipe as mp
import threading
import json
import queue
from collections import defaultdict
from flask import Flask, jsonify
import logging

# Import generated protobuf code (will be generated)
import gaze_stream_pb2
import gaze_stream_pb2_grpc
from grpc_health.v1 import health
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc



VERSION = "0.2.52"

class StreamManager:
    """Manages Pub/Sub for gaze streams"""
    def __init__(self):
        # Map stream_id -> list of Queue
        self._subscribers = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, stream_id):
        """Register a new subscriber and return their queue"""
        q = queue.Queue(maxsize=30)  # Buffer ~1s of frames
        with self._lock:
            self._subscribers[stream_id].append(q)
        print(f"[MANAGER] New subscriber to stream '{stream_id}'")
        return q

    def unsubscribe(self, stream_id, q):
        """Remove a subscriber"""
        with self._lock:
            if q in self._subscribers[stream_id]:
                self._subscribers[stream_id].remove(q)
        print(f"[MANAGER] Subscriber removed from stream '{stream_id}'")

    def publish(self, stream_id, result):
        """Fan-out result to all subscribers"""
        with self._lock:
            queues = self._subscribers.get(stream_id, []).copy()
        
        for q in queues:
            try:
                q.put_nowait(result)
            except queue.Full:
                # Drop frame for slow subscriber to avoid blocking publisher
                pass

# Global manager instance
stream_manager = StreamManager()

class GazeStreamService(gaze_stream_pb2_grpc.GazeStreamServiceServicer):
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("[SERVER] MediaPipe FaceMesh initialized")
        self.health_status = health_pb2.HealthCheckResponse.SERVING
        self.total_frames_processed = 0
        self.active_clients = 0
    
    def get_metadata(self):
        return {
            "service": "GazeStreamService",
            "status": "SERVING" if self.health_status == health_pb2.HealthCheckResponse.SERVING else "NOT_SERVING",
            "backend": "MediaPipe FaceMesh",
            "stats": {
                "total_frames_processed": self.total_frames_processed,
                "active_clients": self.active_clients
            },
            "version": VERSION
        }
    
    def StreamGaze(self, request_iterator, context):
        """
        Bidirectional streaming RPC - DEBUG MODE
        """
        print(f"[SERVER] New client connected: {context.peer()}")
        # context.send_initial_metadata(())
        
        frame_count = 0
        try:
            for i, video_frame in enumerate(request_iterator):
                frame_count += 1
                if i % 30 == 0:
                   print(f"[SERVER] Frame {i} received from {video_frame.stream_id}")
                
                # Yield dummy result
                res = gaze_stream_pb2.FaceMeshResult()
                res.timestamp_us = video_frame.timestamp_us
                res.stream_id = video_frame.stream_id
                
                yield res
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[SERVER] Stream Error: {e}")
            
        print(f"[SERVER] Client disconnected. Processed {frame_count} frames total")
        self.active_clients -= 1

    def SubscribeGaze(self, request, context):
        """
        Server Streaming RPC
        Allows a client to observe results from a specific video stream without sending video.
        """
        stream_id = request.stream_id
        print(f"[SERVER] Subscriber connected to: {stream_id}")
        
        q = stream_manager.subscribe(stream_id)
        
        try:
            while context.is_active():
                # Get result from queue (blocking with timeout to check connectivity)
                try:
                    result = q.get(timeout=1.0)
                    yield result
                except queue.Empty:
                    # Keep loop alive to check context.is_active()
                    continue
        except Exception as e:
             print(f"[SERVER] Subscriber error: {e}")
        finally:
            stream_manager.unsubscribe(stream_id, q)


# REST API
app = Flask(__name__)
# Global reference to service for stats
grpc_service_ref = None

@app.route('/health', methods=['GET'])
def health_check():
    if grpc_service_ref:
        return jsonify(grpc_service_ref.get_metadata()), 200
    return jsonify({"status": "STARTING"}), 503

def run_rest_server(port=8080):
    print(f"[SERVER] REST API starting on port {port}")
    # Disable flask banner
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host='0.0.0.0', port=port)

def serve(port=50051, rest_port=8080):
    """Start the gRPC server"""
    global grpc_service_ref
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    # Keep reference for REST API
    service_instance = GazeStreamService()
    grpc_service_ref = service_instance
    
    gaze_stream_pb2_grpc.add_GazeStreamServiceServicer_to_server(
        service_instance, server
    )
    
    # Create Health Service
    health_servicer = health.HealthServicer(
        experimental_non_blocking=True,
        experimental_thread_pool=futures.ThreadPoolExecutor(max_workers=1)
    )
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    
    # Mark all services as SERVING
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    health_servicer.set("gazestream.GazeStreamService", health_pb2.HealthCheckResponse.SERVING)
    
    # Bind to all interfaces (IPv4)
    bound_port = server.add_insecure_port(f'0.0.0.0:{port}')
    if bound_port == 0:
        raise RuntimeError(f"Failed to bind to port {port}. Is it already in use?")
    
    server.start()
    
    print(f"[SERVER] gRPC Gaze Streaming Server v{VERSION} started on bound port {bound_port}")
    
    # Start REST API in background thread
    rest_thread = threading.Thread(target=run_rest_server, args=(rest_port,), daemon=True)
    rest_thread.start()
    
    print(f"[SERVER] Waiting for clients...")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n[SERVER] Shutting down...")
        server.stop(0)


if __name__ == '__main__':
    serve()
