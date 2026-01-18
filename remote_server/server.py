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

# Import generated protobuf code (will be generated)
import gaze_stream_pb2
import gaze_stream_pb2_grpc


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
    
    def StreamGaze(self, request_iterator, context):
        """
        Bidirectional streaming RPC
        Receives video frames, processes them, yields face mesh results
        """
        print(f"[SERVER] New client connected: {context.peer()}")
        
        frame_count = 0
        
        try:
            for video_frame in request_iterator:
                frame_count += 1
                
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
                
                # Create result message
                result = gaze_stream_pb2.FaceMeshResult()
                result.timestamp_us = video_frame.timestamp_us
                result.stream_id = video_frame.stream_id
                
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
                    
                    # Estimate gaze (simplified placeholder)
                    result.gaze.yaw = 0.0
                    result.gaze.pitch = 0.0
                    result.gaze.roll = 0.0
                    
                    if frame_count % 30 == 0:
                        print(f"[SERVER] Processed {frame_count} frames, detected face with {len(result.landmarks)} landmarks")
                
                # Yield result back to client
                yield result
        
        except Exception as e:
            print(f"[SERVER] Error processing stream: {e}")
            # Don't crash the server loop, just end this stream
            # context.set_code(grpc.StatusCode.INTERNAL)
            # context.set_details(str(e))
        
        print(f"[SERVER] Client disconnected. Processed {frame_count} frames total")


def serve(port=50051):
    """Start the gRPC server"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    gaze_stream_pb2_grpc.add_GazeStreamServiceServicer_to_server(
        GazeStreamService(), server
    )
    
    # Bind to all interfaces
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    
    print(f"[SERVER] gRPC Gaze Streaming Server started on port {port}")
    print(f"[SERVER] Waiting for clients...")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n[SERVER] Shutting down...")
        server.stop(0)


if __name__ == '__main__':
    serve()
