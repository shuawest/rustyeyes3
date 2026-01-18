#!/usr/bin/env python3
import grpc
import gaze_stream_pb2
import gaze_stream_pb2_grpc
import sys

def run():
    target = 'localhost:50051'
    stream_id = 'camera1'
    if len(sys.argv) > 1:
        target = sys.argv[1]
    
    print(f"Connecting to {target} subscribing into channel '{stream_id}'...")
    
    channel = grpc.insecure_channel(target)
    stub = gaze_stream_pb2_grpc.GazeStreamServiceStub(channel)
    
    request = gaze_stream_pb2.SubscriptionRequest(stream_id=stream_id)
    
    try:
        for result in stub.SubscribeGaze(request):
            print(f"Received Result! Timestamp: {result.timestamp_us}")
            print(f"  Gaze: Yaw={result.gaze.yaw:.2f}, Pitch={result.gaze.pitch:.2f}")
            print(f"  Landmarks: {len(result.landmarks)}")
    except grpc.RpcError as e:
        print(f"Stream ended or error: {e}")
    except KeyboardInterrupt:
        print("Stopping...")

if __name__ == '__main__':
    run()
