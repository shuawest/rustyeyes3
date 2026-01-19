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



VERSION = "0.2.34"

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
        Bidirectional streaming RPC
        Receives video frames, processes them, yields face mesh results
        """
        print(f"[SERVER] New client connected: {context.peer()}")
        
        # FIX: Send headers immediately so client stops waiting and can start sending frames
        context.send_initial_metadata(())
        
        self.active_clients += 1
        
        frame_count = 0
        
        try:
            for video_frame in request_iterator:
                # Trace: Server Receive
                ts_recv = int(time.time() * 1_000_000)
                
                frame_count += 1
                self.total_frames_processed += 1
                
                # Decode H.264 frame (for MVP, assume JPEG/Raw for simplicity or use opencv to decode buffer if possible)
                # Note: H.264 decoding manually from bytes in python opencv is tricky without a container or using ffmpeg bindings.
                # For this MVP, let's assume the client sends encoded JPEG or Raw bytes that cv2.imdecode can handle.
                # If sending H.264 packets (NAL units), we would need a proper decoder (av/pyav).
                # To keep MVP simple: assuming frame_data is an image buffer (JPEG/PNG) or raw.
                
                nparr = np.frombuffer(video_frame.frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    # If it's not a standard image format, it might be raw H264 stream chunk.
                    # Handling raw H264 stream in simple python is hard.
                    # Suggest MVP client sends MJPEG for now? Or let's just log and skip.
                    # bandwidth might be high for MJPEG.
                    # print(f"[SERVER] Failed to decode frame {frame_count}")
                    continue
                
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = self.face_mesh.process(rgb_frame)
                
                result = gaze_stream_pb2.FaceMeshResult()
                result.timestamp_us = video_frame.timestamp_us
                result.stream_id = video_frame.stream_id
                
                # Copy trace timestamps from request and add server ones
                if video_frame.trace_timestamps:
                    for k, v in video_frame.trace_timestamps.items():
                        result.trace_timestamps[k] = v
                        
                # Add Server Timestamps
                now_us = int(time.time() * 1_000_000)
                # We can't easily get 'recv' time precisely here as iterator yields, but 'now' is close to start of processing
                result.trace_timestamps["server_recv"] = now_us 
                # Actually, inference just finished (lines above), so 'server_inference_end' is now.
                # 'server_recv' was before cv2.imdecode.
                # Let's adjust slightly:
                # We can't go back in time, but... 
                
                # Let's just mark inference_end
                result.trace_timestamps["server_proc_end"] = now_us
                
                if results.multi_face_landmarks:
                    # Extract first face
                    face_landmarks = results.multi_face_landmarks[0]
                    
                    # Convert landmarks to proto format
                    for landmark in face_landmarks.landmark:
                        lm = result.landmarks.add()
                        lm.x = landmark.x
                        lm.y = landmark.y
                        lm.z = landmark.z
                    
                    # Calculate bounding box
                    x_coords = [lm.x for lm in result.landmarks]
                    y_coords = [lm.y for lm in result.landmarks]
                    if x_coords and y_coords:
                        result.face_box.x = min(x_coords)
                        result.face_box.y = min(y_coords)
                        result.face_box.width = max(x_coords) - min(x_coords)
                        result.face_box.height = max(y_coords) - min(y_coords)
                    
                    # Add recv timestamp we captured earlier
                    result.trace_timestamps["server_recv"] = ts_recv
                    
                    # Estimate gaze using Iris Tracking (Geometric)
                    # 468: Left Iris Center, 473: Right Iris Center
                    # Left Eye: 33 (Inner), 133 (Outer)
                    # Right Eye: 362 (Inner), 263 (Outer)
                    
                    try:
                        # Extract key points
                        lm_pts = face_landmarks.landmark
                        
                        # Helper to get numpy point
                        get_pt = lambda idx: np.array([lm_pts[idx].x, lm_pts[idx].y])
                        
                        # Left Eye
                        p_left_iris = get_pt(468)
                        p_left_inner = get_pt(133) # Inner corner (closer to nose in image? 133 is outer, 33 is inner for left eye usually? let's stick to X ranges)
                        p_left_outer = get_pt(33) 
                        
                        # Right Eye
                        p_right_iris = get_pt(473)
                        p_right_inner = get_pt(362)
                        p_right_outer = get_pt(263)
                        
                        # Calculate horizontal ratio (0.0 = looking left, 1.0 = looking right)
                        # We use simple linear interpolation of Iris X between Inner/Outer X
                        
                        def get_ratio(val, min_v, max_v):
                            return (val - min_v) / (max_v - min_v + 1e-6)
                            
                        # Left Eye Ratio (Outer is left in image X? No, Inner 133 is left? MediaPipe X increases left-to-right)
                        # Eye 1 (Left, user's right): 33 (inner/right side), 133 (outer/left side)
                        # Wait, X increases Left->Right.
                        # Left Eye (User's Left) is on Right side of image (if mirrored).
                        # Let's assume non-mirrored raw frame.
                        # Left Eye (User's Left) is generally x > 0.5.
                        # Right Eye (User's Right) is generally x < 0.5.
                        
                        # Just use distance logic to be safe against mirroring assumptions
                        dl_width = np.linalg.norm(p_left_outer - p_left_inner)
                        dl_iris = np.linalg.norm(p_left_iris - p_left_outer) # Distance from outer
                        ratio_l = dl_iris / dl_width
                        
                        dr_width = np.linalg.norm(p_right_outer - p_right_inner)
                        dr_iris = np.linalg.norm(p_right_iris - p_right_outer) # Distance from outer
                        ratio_r = dr_iris / dr_width
                        
                        avg_ratio = (ratio_l + ratio_r) / 2.0
                        
                        # Map ratio to Yaw (0.5 = center/0 deg)
                        # If ratio > 0.5, looking right (positive yaw)
                        # If ratio < 0.5, looking left (negative yaw)
                        # Scale: +/- 30 degrees seems reasonable for screen bounds
                        result.gaze.yaw = (avg_ratio - 0.5) * 60.0 # +/- 30 deg
                        
                        # For Pitch (Vertical)
                        # Left Eye Top/Bottom: 159 / 145
                        # Right Eye Top/Bottom: 386 / 374
                        p_left_top = get_pt(159)
                        p_left_bot = get_pt(145)
                        h_left = np.linalg.norm(p_left_top - p_left_bot)
                        h_iris_l = np.linalg.norm(p_left_iris - p_left_top)
                        ratio_y_l = h_iris_l / h_left
                        
                        p_right_top = get_pt(386)
                        p_right_bot = get_pt(374)
                        h_right = np.linalg.norm(p_right_top - p_right_bot)
                        h_iris_r = np.linalg.norm(p_right_iris - p_right_top)
                        ratio_y_r = h_iris_r / h_right
                        
                        avg_ratio_y = (ratio_y_l + ratio_y_r) / 2.0
                        
                        # Map to Pitch
                        # 0.5 = center
                        # < 0.5 = looking up (iris closer to top)
                        # > 0.5 = looking down
                        # Invert because usually Pitch Up is positive or negative depending on convention.
                        # Let's assume Pitch Up (looking top) is positive.
                        # Low ratio = High Pitch.
                        result.gaze.pitch = -(avg_ratio_y - 0.5) * 40.0 # +/- 20 deg
                        result.gaze.roll = 0.0
                        
                    except Exception as e:
                        # Fallback if indices fail
                        print(f"[SERVER] Gaze calc failed: {e}")
                        result.gaze.yaw = 0.0
                        result.gaze.pitch = 0.0
                        result.gaze.roll = 0.0
                    
                
                if frame_count % 30 == 0:
                    print(f"[SERVER] Processed {frame_count} frames, detected face with {len(result.landmarks)} landmarks")
            
                # 1. Publish to subscribers (Broadcast)
                stream_manager.publish(video_frame.stream_id, result)

                # 2. Yield result back to the source client (Unicast)
                yield result
        
        except Exception as e:
            print(f"[SERVER] Error processing stream: {e}")
            # Don't crash the server loop, just end this stream
            # context.set_code(grpc.StatusCode.INTERNAL)
            # context.set_details(str(e))
        
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
    
    # Bind to all interfaces
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    
    print(f"[SERVER] gRPC Gaze Streaming Server v{VERSION} started on port {port}")
    
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
