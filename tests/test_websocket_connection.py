import asyncio
import pytest
import base64
import json

import websockets


@pytest.mark.asyncio
@pytest.mark.skip(reason="WebSocket server not available - see issue #402")
async def test_websocket():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        print(f"Connected to {uri}")

        # Send request
        request = {
            "text": "Hello! The WebSocket server is working correctly.",
            "voice": "javert",
        }
        await websocket.send(json.dumps(request))
        print(f"Sent: {request['text']}")

        chunks_received = 0
        while True:
            response = await websocket.recv()
            data = json.loads(response)

            if data["type"] == "audio":
                chunks_received += 1
                audio_len = len(base64.b64decode(data["data"]))
                print(f"Received chunk {data['chunk']} ({audio_len} bytes)")
            elif data["type"] == "done":
                print(f"Generation complete. Total chunks: {data['total_chunks']}")
                break
            elif data["type"] == "error":
                print(f"Error: {data['message']}")
                break


if __name__ == "__main__":
    try:
        asyncio.run(test_websocket())
    except Exception as e:
        print(f"Test failed: {e}")
