import asyncio
import websockets
import json
import logging
import base64

# CONFIGURATION
LAPTOP_WS_URL = "ws://10.2.4.46:8090"  # Your Laptop IP
UNMUTE_WS_URL = "ws://127.0.0.1:80/api/v1/realtime"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UnmuteBridge")

async def run_bridge():
    while True:
        try:
            logger.info(f"Connecting to Laptop at {LAPTOP_WS_URL}...")
            async with websockets.connect(LAPTOP_WS_URL) as laptop_ws:
                logger.info("Connected to Laptop!")
                
                logger.info(f"Connecting to Unmute at {UNMUTE_WS_URL}...")
                async with websockets.connect(UNMUTE_WS_URL, subprotocols=["realtime"]) as unmute_ws:
                    logger.info("Connected to Unmute! Bridge is active.")
                    
                    # Task to forward audio: Laptop -> Unmute
                    async def forward_audio_to_unmute():
                        async for message in laptop_ws:
                            try:
                                data = json.loads(message)
                                if data.get("type") == "audio":
                                    # Forward as Raw PCM to Unmute
                                    # We use the custom event we created!
                                    unmute_msg = {
                                        "type": "unmute.input_audio_buffer.append_pcm",
                                        "audio": data["data"],
                                        "format": "int16" # Assuming audio_publisher.py sends int16
                                    }
                                    await unmute_ws.send(json.dumps(unmute_msg))
                            except Exception as e:
                                logger.error(f"Error forwarding audio: {e}")

                    # Task to receive responses: Unmute -> Laptop
                    async def forward_response_to_laptop():
                        async for message in unmute_ws:
                            try:
                                data = json.loads(message)
                                msg_type = data.get("type")
                                
                                # We only care about audio delta (voice) or text (subtitles/commands)
                                if msg_type == "response.audio.delta":
                                    # Forward audio back to laptop
                                    # We wrap it in a custom message for our laptop server
                                    payload = {
                                        "type": "robot.voice_audio",
                                        "audio": data["delta"] # This is Base64 Opus from Unmute
                                    }
                                    await laptop_ws.send(json.dumps(payload))
                                    
                                elif msg_type == "response.text.delta":
                                    # Forward text
                                    payload = {
                                        "type": "robot.text",
                                        "text": data["delta"]
                                    }
                                    await laptop_ws.send(json.dumps(payload))
                                    
                                # TODO: Handle robot commands here later if we implement that event
                                
                            except Exception as e:
                                logger.error(f"Error forwarding response: {e}")

                    # Run both tasks concurrently
                    await asyncio.gather(
                        forward_audio_to_unmute(),
                        forward_response_to_laptop()
                    )
                            
        except Exception as e:
            logger.error(f"Bridge connection error: {e}")
            logger.info("Retrying in 3 seconds...")
            await asyncio.sleep(3)

if __name__ == "__main__":
    asyncio.run(run_bridge())