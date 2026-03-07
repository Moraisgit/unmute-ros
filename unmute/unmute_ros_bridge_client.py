import asyncio
import base64
import json
import logging
import os

import numpy as np
import websockets

# Defaults target the "ROS on laptop + remote Unmute over SSH tunnel" setup.
# Example tunnel:
#   ssh -N -L 3333:localhost:80 <remote-host>
LAPTOP_WS_URL = os.environ.get("LAPTOP_WS_URL", "ws://127.0.0.1:8090")
UNMUTE_WS_URL = os.environ.get(
    "UNMUTE_WS_URL", "ws://127.0.0.1:3333/api/v1/realtime"
)
PCM_FORMAT = os.environ.get("PCM_FORMAT", "int16")
INPUT_SAMPLE_RATE = int(os.environ.get("INPUT_SAMPLE_RATE", "16000"))
UNMUTE_SAMPLE_RATE = int(os.environ.get("UNMUTE_SAMPLE_RATE", "24000"))
RESAMPLE_AUDIO = os.environ.get("RESAMPLE_AUDIO", "true").lower() == "true"
UNMUTE_VOICE = os.environ.get("UNMUTE_VOICE", None)
ALLOW_RECORDING = os.environ.get("ALLOW_RECORDING", "false").lower() == "true"
RECONNECT_DELAY_SEC = float(os.environ.get("RECONNECT_DELAY_SEC", "3.0"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("UnmuteBridge")


def _to_float32_pcm(raw_audio_b64: str, pcm_format: str) -> np.ndarray:
    decoded = base64.b64decode(raw_audio_b64)
    if pcm_format == "float32":
        return np.frombuffer(decoded, dtype=np.float32)
    if pcm_format == "int16":
        pcm_int16 = np.frombuffer(decoded, dtype=np.int16)
        return (pcm_int16.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
    raise ValueError(f"Unsupported PCM_FORMAT={pcm_format!r}")


def _resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if audio.size == 0 or src_rate == dst_rate:
        return audio

    # Linear interpolation is fast, robust, and adequate for voice transport.
    src_len = audio.shape[0]
    dst_len = max(1, int(round(src_len * dst_rate / src_rate)))
    src_x = np.linspace(0.0, 1.0, num=src_len, endpoint=False)
    dst_x = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
    return np.interp(dst_x, src_x, audio).astype(np.float32)


def _encode_float32_b64(audio: np.ndarray) -> str:
    return base64.b64encode(audio.astype(np.float32).tobytes()).decode("utf-8")


async def _send_initial_session_update(unmute_ws: websockets.ClientConnection) -> None:
    """Initialize Unmute session so generation can start when audio arrives."""
    session = {
        "allow_recording": ALLOW_RECORDING,
    }
    if UNMUTE_VOICE:
        session["voice"] = UNMUTE_VOICE

    payload = {
        "type": "session.update",
        "session": session,
    }
    await unmute_ws.send(json.dumps(payload))


async def run_bridge() -> None:
    if PCM_FORMAT not in {"int16", "float32"}:
        raise ValueError("PCM_FORMAT must be one of: int16, float32")

    while True:
        try:
            logger.info("Connecting to laptop audio websocket: %s", LAPTOP_WS_URL)
            async with websockets.connect(LAPTOP_WS_URL) as laptop_ws:
                logger.info("Laptop socket connected")

                logger.info("Connecting to Unmute websocket: %s", UNMUTE_WS_URL)
                async with websockets.connect(
                    UNMUTE_WS_URL,
                    subprotocols=["realtime"],
                ) as unmute_ws:
                    logger.info("Unmute socket connected")
                    await _send_initial_session_update(unmute_ws)
                    logger.info(
                        (
                            "Bridge active (pcm_format=%s, allow_recording=%s, "
                            "resample=%s, input_sr=%s, unmute_sr=%s)"
                        ),
                        PCM_FORMAT,
                        ALLOW_RECORDING,
                        RESAMPLE_AUDIO,
                        INPUT_SAMPLE_RATE,
                        UNMUTE_SAMPLE_RATE,
                    )

                    async def forward_audio_to_unmute() -> None:
                        async for message in laptop_ws:
                            try:
                                data = json.loads(message)
                                if data.get("type") != "audio":
                                    continue

                                outgoing_audio_b64 = data["data"]
                                outgoing_format = PCM_FORMAT

                                if RESAMPLE_AUDIO and INPUT_SAMPLE_RATE != UNMUTE_SAMPLE_RATE:
                                    audio_f32 = _to_float32_pcm(data["data"], PCM_FORMAT)
                                    resampled = _resample_linear(
                                        audio_f32,
                                        src_rate=INPUT_SAMPLE_RATE,
                                        dst_rate=UNMUTE_SAMPLE_RATE,
                                    )
                                    outgoing_audio_b64 = _encode_float32_b64(resampled)
                                    outgoing_format = "float32"

                                unmute_msg = {
                                    "type": "unmute.input_audio_buffer.append_pcm",
                                    "audio": outgoing_audio_b64,
                                    "format": outgoing_format,
                                }
                                await unmute_ws.send(json.dumps(unmute_msg))
                            except Exception as exc:
                                logger.error("Error forwarding audio to Unmute: %s", exc)

                    async def forward_response_to_laptop() -> None:
                        async for message in unmute_ws:
                            try:
                                data = json.loads(message)
                                msg_type = data.get("type")

                                if msg_type == "response.audio.delta":
                                    payload = {
                                        "type": "robot.voice_audio",
                                        "audio": data["delta"],
                                    }
                                    await laptop_ws.send(json.dumps(payload))
                                elif msg_type == "response.text.delta":
                                    payload = {
                                        "type": "robot.text",
                                        "text": data["delta"],
                                    }
                                    await laptop_ws.send(json.dumps(payload))
                            except Exception as exc:
                                logger.error(
                                    "Error forwarding Unmute response: %s", exc
                                )

                    await asyncio.gather(
                        forward_audio_to_unmute(),
                        forward_response_to_laptop(),
                    )

        except Exception as exc:
            logger.error("Bridge connection error: %s", exc)
            logger.info("Retrying in %.1f seconds...", RECONNECT_DELAY_SEC)
            await asyncio.sleep(RECONNECT_DELAY_SEC)


if __name__ == "__main__":
    asyncio.run(run_bridge())