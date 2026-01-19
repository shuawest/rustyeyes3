import grpc
import time
import gaze_stream_pb2
import gaze_stream_pb2_grpc

def generate_frames():
    for i in range(10):
        print(f"Sending frame {i}")
        frame = gaze_stream_pb2.VideoFrame()
        frame.timestamp_us = int(time.time() * 1000000)
        frame.stream_id = "test_cam"
        frame.width = 640
        frame.height = 480
        yield frame
        time.sleep(0.1)

def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = gaze_stream_pb2_grpc.GazeStreamServiceStub(channel)
    
    print("Connecting to StreamGaze...")
    try:
        responses = stub.StreamGaze(generate_frames())
        for response in responses:
            print(f"Received result: {response.timestamp_us}")
    except grpc.RpcError as e:
        print(f"RPC Error: {e}")

if __name__ == '__main__':
    run()
