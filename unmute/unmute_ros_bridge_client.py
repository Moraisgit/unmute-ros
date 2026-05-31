import asyncio
import base64
import json
import logging
import os
import sys
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np
import websockets

from unmute.llm.unmute_tag_parser import LLMTagPrinter

# Defaults target the "ROS on laptop + remote Unmute over SSH tunnel" setup.
# Example tunnel:
#   ssh -N -L 3333:localhost:80 <remote-host>
LAPTOP_WS_URL = os.environ.get("LAPTOP_WS_URL", "ws://127.0.0.1:8090")
UNMUTE_WS_URL = os.environ.get(
    "UNMUTE_WS_URL", "ws://127.0.0.1:3333/api/v1/realtime"
)
PCM_FORMAT = os.environ.get("PCM_FORMAT", "int16")
INPUT_SAMPLE_RATE = int(os.environ.get("INPUT_SAMPLE_RATE", "24000"))
UNMUTE_SAMPLE_RATE = int(os.environ.get("UNMUTE_SAMPLE_RATE", "24000"))
RESAMPLE_AUDIO = os.environ.get("RESAMPLE_AUDIO", "false").lower() == "true"
UNMUTE_VOICE = os.environ.get("UNMUTE_VOICE", None)
ALLOW_RECORDING = os.environ.get("ALLOW_RECORDING", "false").lower() == "true"
RECONNECT_DELAY_SEC = float(os.environ.get("RECONNECT_DELAY_SEC", "3.0"))
PRINT_TEXT_DELTAS = os.environ.get("PRINT_TEXT_DELTAS", "false").lower() == "true"
DEBUG_MIC_INPUT = os.environ.get("DEBUG_MIC_INPUT", "false").lower() == "true"
DEBUG_MIC_EVERY_N_PACKETS = int(os.environ.get("DEBUG_MIC_EVERY_N_PACKETS", "25"))
DEBUG_STT_EVENTS = os.environ.get("DEBUG_STT_EVENTS", "false").lower() == "true"
PRINT_USER_TRANSCRIPT_DELTAS = (
    os.environ.get("PRINT_USER_TRANSCRIPT_DELTAS", "true").lower() == "true"
)
DEBUG_LLM_OUTPUT_TAGS = (
    os.environ.get("DEBUG_LLM_OUTPUT_TAGS", "false").lower() == "true"
)
DEBUG_LLM_RAW_OUTPUT = (
    os.environ.get("DEBUG_LLM_RAW_OUTPUT", "false").lower() == "true"
)
ACTION_RESULT_QUEUE_MAXSIZE = int(
    os.environ.get("ACTION_RESULT_QUEUE_MAXSIZE", "64")
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("UnmuteBridge")


class SessionResetRequested(Exception):
    def __init__(self, source: str, reason: str) -> None:
        super().__init__(f"Session reset requested by {source}: {reason}")
        self.source = source
        self.reason = reason


def _supports_color_output() -> bool:
    if os.environ.get("FORCE_COLOR") is not None:
        return True
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _make_label(tag: str, color_code: str) -> str:
    if not _supports_color_output():
        return f"[{tag}]"
    reset = "\033[0m"
    return f"{color_code}[{tag}]{reset}"


USER_LABEL = _make_label("User", "\033[92m")
UNMUTE_LABEL = _make_label("Unmute", "\033[96m")
THINK_LABEL = _make_label("Unmute - Think", "\033[95m")
PLAN_LABEL = _make_label("Unmute - Plan", "\033[93m")
SPEECH_TAG_LABEL = _make_label("Unmute - Speech", "\033[96m")
EXEC_LABEL = _make_label("Unmute - Exec", "\033[94m")
ACTION_RESULT_LABEL = _make_label("Action Feedback", "\033[92m")
RAW_LLM_LABEL = _make_label("Unmute - Raw LLM", "\033[36m")
TAG_LABELS: dict[str, str] = {
    "think": THINK_LABEL,
    "plan": PLAN_LABEL,
    "speech": SPEECH_TAG_LABEL,
    "exec": EXEC_LABEL,
    "action_result": ACTION_RESULT_LABEL,
}


def _format_action_result_for_print(content: str) -> str:
    trimmed = content.strip()
    if trimmed.startswith("<action_result>") and trimmed.endswith("</action_result>"):
        trimmed = trimmed[len("<action_result>") : -len("</action_result>")].strip()
    try:
        payload = json.loads(trimmed)
    except json.JSONDecodeError:
        return trimmed
    return json.dumps(payload, indent=2, ensure_ascii=True)


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


def _audio_level_stats(audio: np.ndarray) -> tuple[float, float]:
    if audio.size == 0:
        return 0.0, 0.0
    rms = float(np.sqrt(np.mean(np.square(audio))))
    peak = float(np.max(np.abs(audio)))
    return rms, peak


def _needs_boundary_space(last_char: str | None, new_text: str) -> bool:
    if not last_char or not new_text:
        return False

    first_char = new_text[0]
    if last_char.isspace() or first_char.isspace():
        return False

    # Keep punctuation/contractions tight.
    if first_char in ",.!?;:)]}\"'":
        return False
    if last_char in "([{\"":
        return False
    if last_char in "'-/":
        return False

    # Word-like boundary without whitespace: add one.
    if last_char.isalnum() and first_char.isalnum():
        return True

    # Fallback: separate most non-space boundaries for readability.
    return True


async def _send_initial_session_update(unmute_ws: websockets.ClientConnection) -> None:
    """Initialize Unmute session so generation can start when audio arrives."""

    def _voices_url_from_ws_url(ws_url: str) -> str | None:
        parsed = urlparse(ws_url)
        if parsed.scheme not in {"ws", "wss"}:
            return None

        scheme = "https" if parsed.scheme == "wss" else "http"

        # Support both /v1/realtime and /api/v1/realtime style paths.
        path = parsed.path.rstrip("/")
        if path.endswith("/v1/realtime"):
            prefix = path[: -len("/v1/realtime")]
            voices_path = f"{prefix}/v1/voices"
        else:
            voices_path = "/v1/voices"

        return f"{scheme}://{parsed.netloc}{voices_path}"

    def _resolve_voice_and_instructions(
        requested_voice: str | None,
    ) -> tuple[str | None, dict | None]:
        if not requested_voice:
            return None, None

        voices_url = _voices_url_from_ws_url(UNMUTE_WS_URL)
        if not voices_url:
            logger.warning(
                "Couldn't infer voices URL from UNMUTE_WS_URL=%s; using UNMUTE_VOICE as-is",
                UNMUTE_WS_URL,
            )
            return requested_voice, None

        try:
            with urlopen(voices_url, timeout=5.0) as response:
                voices = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            logger.warning(
                "Failed to fetch voices from %s (%s); using UNMUTE_VOICE as-is",
                voices_url,
                exc,
            )
            return requested_voice, None

        for voice in voices:
            source = voice.get("source") or {}
            path_on_server = source.get("path_on_server")
            if requested_voice in {voice.get("name"), path_on_server}:
                return path_on_server or requested_voice, voice.get("instructions")

        logger.warning(
            "UNMUTE_VOICE=%s not found in /v1/voices; using value as raw voice id",
            requested_voice,
        )
        return requested_voice, None

    resolved_voice, resolved_instructions = await asyncio.to_thread(
        _resolve_voice_and_instructions,
        UNMUTE_VOICE,
    )

    session = {
        "allow_recording": ALLOW_RECORDING,
    }
    if resolved_voice:
        session["voice"] = resolved_voice
    if resolved_instructions:
        session["instructions"] = resolved_instructions

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
                while True:
                    paused_session = False
                    logger.info("Connecting to Unmute websocket: %s", UNMUTE_WS_URL)
                    async with websockets.connect(
                        UNMUTE_WS_URL,
                        subprotocols=[websockets.Subprotocol("realtime")],
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

                        assistant_speaking = False
                        assistant_audio_seen = False
                        user_speaking = False
                        action_result_queue: asyncio.Queue[str] = asyncio.Queue(
                            maxsize=ACTION_RESULT_QUEUE_MAXSIZE
                        )
                        send_lock = asyncio.Lock()
                        laptop_send_lock = asyncio.Lock()

                        async def _send_to_unmute(payload: dict) -> None:
                            async with send_lock:
                                await unmute_ws.send(json.dumps(payload))

                        async def _send_to_laptop(payload: dict) -> None:
                            async with laptop_send_lock:
                                await laptop_ws.send(json.dumps(payload))

                        def _can_inject_action_result() -> bool:
                            return (
                                not paused_session
                                and not assistant_speaking
                                and not user_speaking
                            )

                        async def _send_action_result(content: str) -> None:
                            formatted = _format_action_result_for_print(content)
                            print(f"{ACTION_RESULT_LABEL} {formatted}", flush=True)
                            await _send_to_unmute(
                                {
                                    "type": "unmute.user_message",
                                    "content": content,
                                }
                            )
                            await _send_to_laptop(
                                {
                                    "type": "robot.llm_tag_block",
                                    "tag_name": "action_result",
                                    "content": content,
                                }
                            )

                        async def _flush_action_results() -> None:
                            if not _can_inject_action_result():
                                return
                            try:
                                content = action_result_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                return
                            await _send_action_result(content)

                        def _queue_action_result(content: str) -> None:
                            try:
                                action_result_queue.put_nowait(content)
                            except asyncio.QueueFull:
                                try:
                                    _ = action_result_queue.get_nowait()
                                except asyncio.QueueEmpty:
                                    pass
                                try:
                                    action_result_queue.put_nowait(content)
                                except asyncio.QueueFull:
                                    logger.warning(
                                        "Dropping action_result because queue is full."
                                    )

                        async def forward_audio_to_unmute() -> None:
                            nonlocal paused_session
                            nonlocal assistant_speaking
                            nonlocal assistant_audio_seen
                            nonlocal user_speaking
                            packet_count = 0
                            async for message in laptop_ws:
                                try:
                                    data = json.loads(message)
                                except json.JSONDecodeError:
                                    continue
                                msg_type = data.get("type")

                                if msg_type == "bridge.reset_session":
                                    source = data.get("source", "unknown")
                                    reason = data.get("reason", "unspecified")
                                    raise SessionResetRequested(source=source, reason=reason)

                                if msg_type == "bridge.pause_session":
                                    source = data.get("source", "unknown")
                                    reason = data.get("reason", "unspecified")
                                    paused_session = True
                                    assistant_speaking = False
                                    assistant_audio_seen = False
                                    user_speaking = False
                                    logger.info(
                                        "Session paused by %s (%s). Suppressing bridge output until resume.",
                                        source,
                                        reason,
                                    )
                                    continue

                                if msg_type == "bridge.resume_session":
                                    source = data.get("source", "unknown")
                                    reason = data.get("reason", "unspecified")
                                    paused_session = False
                                    logger.info(
                                        "Session resumed by %s (%s). Restoring bridge output.",
                                        source,
                                        reason,
                                    )
                                    await _flush_action_results()
                                    continue

                                if msg_type == "bridge.action_result":
                                    content = data.get("content", "")
                                    if not content:
                                        continue
                                    if _can_inject_action_result():
                                        await _send_action_result(content)
                                    else:
                                        _queue_action_result(content)
                                    continue

                                if msg_type == "browser.audio_opus":
                                    # Browser mic: already Opus-encoded with AEC
                                    # applied by the browser. Forward as-is to the
                                    # remote Unmute backend via the standard
                                    # input_audio_buffer.append event.
                                    if paused_session:
                                        continue
                                    audio_b64 = data.get("audio", "")
                                    if not audio_b64:
                                        continue
                                    try:
                                        await _send_to_unmute(
                                            {
                                                "type": "input_audio_buffer.append",
                                                "audio": audio_b64,
                                            }
                                        )
                                    except websockets.exceptions.ConnectionClosed as exc:
                                        logger.info(
                                            "Unmute websocket closed while forwarding browser audio; reconnecting: %s",
                                            exc,
                                        )
                                        raise
                                    continue

                                if msg_type != "audio":
                                    continue

                                if paused_session:
                                    continue

                                packet_count += 1

                                try:
                                    outgoing_audio_b64 = data["data"]
                                    outgoing_format = PCM_FORMAT

                                    mic_audio_f32 = _to_float32_pcm(data["data"], PCM_FORMAT)

                                    if (
                                        DEBUG_MIC_INPUT
                                        and packet_count % max(1, DEBUG_MIC_EVERY_N_PACKETS) == 0
                                    ):
                                        rms, peak = _audio_level_stats(mic_audio_f32)
                                        logger.info(
                                            (
                                                "Mic input packet=%s samples=%s rms=%.4f peak=%.4f "
                                                "in_format=%s"
                                            ),
                                            packet_count,
                                            mic_audio_f32.size,
                                            rms,
                                            peak,
                                            PCM_FORMAT,
                                        )

                                    if RESAMPLE_AUDIO and INPUT_SAMPLE_RATE != UNMUTE_SAMPLE_RATE:
                                        resampled = _resample_linear(
                                            mic_audio_f32,
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
                                    await _send_to_unmute(unmute_msg)
                                except websockets.exceptions.ConnectionClosed as exc:
                                    logger.info(
                                        "Unmute websocket closed while forwarding audio; reconnecting: %s",
                                        exc,
                                    )
                                    raise
                                except Exception as exc:
                                    logger.error("Error forwarding audio to Unmute: %s", exc)

                        async def forward_response_to_laptop() -> None:
                            nonlocal paused_session
                            nonlocal assistant_speaking
                            nonlocal assistant_audio_seen
                            nonlocal user_speaking
                            text_deltas: list[str] = []
                            active_stream_speaker: str | None = None
                            last_char_by_speaker: dict[str, str | None] = {
                                "user": None,
                                "unmute": None,
                                "llm_tag_think": None,
                                "llm_tag_plan": None,
                                "llm_tag_speech": None,
                                "llm_tag_exec": None,
                                "llm_tag_action_result": None,
                                "llm_raw_output": None,
                            }
                            tag_printer: LLMTagPrinter | None = (
                                LLMTagPrinter() if DEBUG_LLM_OUTPUT_TAGS else None
                            )
                            # Accumulate parsed tag blocks during streaming so they
                            # can be printed all at once after the raw stream, rather
                            # than interleaved with it.
                            pending_tag_blocks: list[tuple[str, str]] = []

                            def _print_stream_chunk(speaker: str, label: str, text: str) -> None:
                                nonlocal active_stream_speaker
                                if not text:
                                    return
                                if active_stream_speaker != speaker:
                                    if active_stream_speaker is not None:
                                        print("", flush=True)
                                    print(f"{label} ", end="", flush=True)
                                    active_stream_speaker = speaker
                                if _needs_boundary_space(last_char_by_speaker[speaker], text):
                                    print(" ", end="", flush=True)
                                print(text, end="", flush=True)
                                last_char_by_speaker[speaker] = text[-1]

                            def _reset_tag_state(reason: str) -> None:
                                nonlocal active_stream_speaker
                                if reason and DEBUG_STT_EVENTS:
                                    logger.debug("Resetting tag parser (%s)", reason)
                                active_stream_speaker = None
                                pending_tag_blocks.clear()
                                text_deltas.clear()
                                if tag_printer is not None:
                                    tag_printer.flush()
                                for key in list(last_char_by_speaker):
                                    if key.startswith("llm_tag_") or key == "llm_raw_output":
                                        last_char_by_speaker[key] = None

                            async for message in unmute_ws:
                                try:
                                    data = json.loads(message)
                                    msg_type = data.get("type")

                                    if paused_session:
                                        if msg_type == "response.text.done":
                                            active_stream_speaker = None
                                            last_char_by_speaker["unmute"] = None
                                            text_deltas.clear()
                                            pending_tag_blocks.clear()
                                            if tag_printer is not None:
                                                tag_printer.flush()
                                        elif msg_type == "conversation.item.input_audio_transcription.completed":
                                            active_stream_speaker = None
                                            last_char_by_speaker["user"] = None
                                        elif msg_type == "unmute.response.text.delta.ready":
                                            pending_tag_blocks.clear()
                                            if tag_printer is not None:
                                                tag_printer.flush()
                                        continue

                                    if msg_type == "response.audio.delta":
                                        assistant_speaking = True
                                        assistant_audio_seen = True
                                        payload = {
                                            "type": "robot.voice_audio",
                                            "audio": data["delta"],
                                        }
                                        await laptop_ws.send(json.dumps(payload))
                                    elif msg_type == "input_audio_buffer.speech_started":
                                        user_speaking = True
                                        _reset_tag_state("speech_started")
                                        if DEBUG_STT_EVENTS:
                                            logger.debug("STT/VAD: speech_started")
                                        await laptop_ws.send(
                                            json.dumps({"type": "robot.speech_started"})
                                        )
                                    elif msg_type == "input_audio_buffer.speech_stopped":
                                        user_speaking = False
                                        if DEBUG_STT_EVENTS:
                                            logger.debug("STT/VAD: speech_stopped")
                                        await laptop_ws.send(
                                            json.dumps({"type": "robot.speech_stopped"})
                                        )
                                        await _flush_action_results()
                                    elif msg_type == "conversation.item.input_audio_transcription.delta":
                                        delta = data.get("delta", "")
                                        if PRINT_USER_TRANSCRIPT_DELTAS and delta:
                                            _print_stream_chunk("user", USER_LABEL, delta)
                                        if delta:
                                            _reset_tag_state("user_transcript_delta")
                                            await laptop_ws.send(
                                                json.dumps(
                                                    {
                                                        "type": "robot.user_text_delta",
                                                        "delta": delta,
                                                    }
                                                )
                                            )
                                    elif msg_type == "unmute.interrupted_by_vad":
                                        _reset_tag_state("interrupted_by_vad")
                                    elif msg_type == "response.text.delta":
                                        assistant_speaking = True
                                        text_delta = data.get("delta", "")
                                        if text_delta:
                                            text_deltas.append(text_delta)
                                            if PRINT_TEXT_DELTAS and not DEBUG_LLM_OUTPUT_TAGS:
                                                _print_stream_chunk(
                                                    "unmute", UNMUTE_LABEL, text_delta
                                                )
                                        payload = {
                                            "type": "robot.text",
                                            "text": text_delta,
                                        }
                                        await laptop_ws.send(json.dumps(payload))
                                    elif msg_type == "response.audio.done":
                                        assistant_speaking = False
                                        assistant_audio_seen = False
                                        await _flush_action_results()
                                    elif msg_type == "unmute.response.text.delta.ready":
                                        raw_delta = data.get("delta", "")
                                        if raw_delta:
                                            if DEBUG_LLM_RAW_OUTPUT:
                                                _print_stream_chunk(
                                                    "llm_raw_output", RAW_LLM_LABEL, raw_delta
                                                )
                                                await laptop_ws.send(
                                                    json.dumps(
                                                        {
                                                            "type": "robot.llm_raw_delta",
                                                            "delta": raw_delta,
                                                        }
                                                    )
                                                )
                                            if tag_printer is not None:
                                                pending_tag_blocks.extend(
                                                    tag_printer.feed(raw_delta)
                                                )
                                    elif msg_type == "response.text.done":
                                        if not assistant_audio_seen:
                                            assistant_speaking = False
                                        assistant_audio_seen = False
                                        # Streaming-only mode: no final full-sentence print.
                                        if active_stream_speaker == "unmute" or (
                                            active_stream_speaker is not None
                                            and (
                                                active_stream_speaker.startswith("llm_tag_")
                                                or active_stream_speaker == "llm_raw_output"
                                            )
                                        ):
                                            print("", flush=True)
                                            active_stream_speaker = None
                                        last_char_by_speaker["unmute"] = None
                                        last_char_by_speaker["llm_raw_output"] = None
                                        for key in list(last_char_by_speaker):
                                            if key.startswith("llm_tag_"):
                                                last_char_by_speaker[key] = None
                                        # Now that the raw stream is done, flush the
                                        # accumulated parsed tag blocks in arrival order.
                                        for tag_name, content in pending_tag_blocks:
                                            label = TAG_LABELS.get(tag_name, UNMUTE_LABEL)
                                            _print_stream_chunk(
                                                f"llm_tag_{tag_name}", label, content
                                            )
                                            if DEBUG_LLM_OUTPUT_TAGS:
                                                await laptop_ws.send(
                                                    json.dumps(
                                                        {
                                                            "type": "robot.llm_tag_block",
                                                            "tag_name": tag_name,
                                                            "content": content,
                                                        }
                                                    )
                                                )
                                        if pending_tag_blocks and active_stream_speaker is not None:
                                            print("", flush=True)
                                            active_stream_speaker = None
                                        pending_tag_blocks.clear()
                                        text_deltas.clear()
                                        if tag_printer is not None:
                                            tag_printer.flush()
                                            tag_printer = LLMTagPrinter()
                                        await laptop_ws.send(
                                            json.dumps({"type": "robot.response_complete"})
                                        )
                                        await _flush_action_results()
                                    elif msg_type == "conversation.item.input_audio_transcription.completed":
                                        if PRINT_USER_TRANSCRIPT_DELTAS and active_stream_speaker == "user":
                                            print("", flush=True)
                                            active_stream_speaker = None
                                        last_char_by_speaker["user"] = None
                                except Exception as exc:
                                    logger.error(
                                        "Error forwarding Unmute response: %s", exc
                                    )

                        bridge_tasks = (
                            asyncio.create_task(forward_audio_to_unmute()),
                            asyncio.create_task(forward_response_to_laptop()),
                        )

                        done, pending = await asyncio.wait(
                            bridge_tasks,
                            return_when=asyncio.FIRST_EXCEPTION,
                        )

                        for task in pending:
                            task.cancel()

                        results = await asyncio.gather(*bridge_tasks, return_exceptions=True)

                        reset_request: SessionResetRequested | None = None
                        unexpected_error: Exception | None = None
                        for task, result in zip(bridge_tasks, results):
                            if isinstance(result, SessionResetRequested):
                                reset_request = result
                                continue
                            if isinstance(result, asyncio.CancelledError):
                                continue
                            if isinstance(result, Exception):
                                if task in done and unexpected_error is None:
                                    unexpected_error = result

                        if reset_request is not None:
                            exc = reset_request
                            logger.info(
                                "Restart requested from %s (%s). Reconnecting Unmute websocket to reset session context.",
                                exc.source,
                                exc.reason,
                            )
                            continue

                        if unexpected_error is not None:
                            raise unexpected_error

                        logger.info("Unmute stream task completed; reconnecting websocket...")
                        continue
                        
                    logger.info("Unmute socket disconnected; reconnecting...")

        except Exception as exc:
            logger.error("Bridge connection error: %s", exc)
            logger.info("Retrying in %.1f seconds...", RECONNECT_DELAY_SEC)
            await asyncio.sleep(RECONNECT_DELAY_SEC)


if __name__ == "__main__":
    try:
        asyncio.run(run_bridge())
    except KeyboardInterrupt:
        logger.info("Bridge stopped by user")
