from __future__ import annotations

from dataclasses import dataclass
from queue import Queue
import re
import time
from typing import Generator, Iterator, Optional

import sounddevice as sd
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech


def validate_audio_device(device: Optional[int | str], sample_rate: int = 16000) -> tuple[bool, Optional[str]]:
    """
    Validate that an audio device can be opened.
    Returns (is_valid, error_message).
    
    Uses callback-based (non-blocking) mode to support devices that don't
    support blocking API (common with USB audio devices on Windows).
    """
    if device is None:
        return True, None
    
    try:
        # Try to query device info first
        devices = sd.query_devices()
        if isinstance(device, int):
            if device < 0 or device >= len(devices):
                return False, f"Device index {device} is out of range (0-{len(devices)-1})"
            dev_info = devices[device]
            if dev_info.get("max_input_channels", 0) <= 0:
                return False, f"Device {device} ({dev_info.get('name', 'unknown')}) has no input channels"
        
        # Use callback-based stream (non-blocking) to support devices that
        # don't support blocking API (e.g., some USB audio devices on Windows)
        test_callback_called = False
        
        def test_callback(indata, frames, time, status):
            nonlocal test_callback_called
            test_callback_called = True
        
        test_stream = sd.RawInputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=int(sample_rate * 0.05),  # 50ms
            device=device,
            callback=test_callback,
        )
        test_stream.start()
        # Give it a moment to potentially call the callback
        time.sleep(0.1)
        test_stream.stop()
        test_stream.close()
        return True, None
    except sd.PortAudioError as e:
        device_name = "unknown"
        if isinstance(device, int):
            try:
                devices = sd.query_devices()
                if 0 <= device < len(devices):
                    device_name = devices[device].get("name", "unknown")
            except Exception:
                pass
        error_msg = str(e)
        if "Blocking API not supported" in error_msg or "-9999" in error_msg:
            # This device doesn't support blocking mode, but callback mode should work
            # Skip validation and let the actual stream handle it
            return True, None
        return False, f"Cannot open device {device} ({device_name}): {e}"
    except Exception as e:
        return False, f"Error validating device {device}: {e}"


@dataclass
class STTResult:
    """A single transcript event emitted from streaming recognition."""

    text: str
    is_final: bool
    stability: float
    language_code: str


class _MicrophoneStream:
    """Capture mono PCM16 audio from default microphone and yield raw bytes."""

    def __init__(self, sample_rate: int, chunk_ms: int, device: Optional[int | str] = None) -> None:
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.block_size = int(sample_rate * chunk_ms / 1000)
        self.device = device
        self._queue: Queue[Optional[bytes]] = Queue()
        self._stream: Optional[sd.RawInputStream] = None
        self._closed = True

    def __enter__(self) -> _MicrophoneStream:
        self._closed = False
        # Validate device before attempting to open
        is_valid, error_msg = validate_audio_device(self.device, self.sample_rate)
        if not is_valid:
            raise ValueError(f"Invalid audio device: {error_msg}")
        
        try:
            self._stream = sd.RawInputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self.block_size,
                device=self.device,
                callback=self._callback,
            )
            self._stream.start()
        except sd.PortAudioError as e:
            device_info = ""
            if isinstance(self.device, int):
                try:
                    devices = sd.query_devices()
                    if 0 <= self.device < len(devices):
                        dev = devices[self.device]
                        device_info = f" ({dev.get('name', 'unknown')})"
                except Exception:
                    pass
            raise RuntimeError(
                f"Failed to open audio device {self.device}{device_info}: {e}. "
                f"Make sure the device is not in use by another application and has proper permissions."
            ) from e
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._closed = True
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        self._queue.put(None)

    def _callback(self, indata, frames, time, status) -> None:
        if status:
            # Keep streaming even on transient PortAudio status flags.
            pass
        self._queue.put(bytes(indata))

    def audio_chunks(self) -> Iterator[bytes]:
        """Yield audio chunks from the queue. Blocks until chunk is available."""
        chunk_count = 0
        while not self._closed:
            try:
                chunk = self._queue.get()
                if chunk is None:
                    return
                chunk_count += 1
                if chunk_count <= 5:
                    print(f"[DEBUG] Audio chunk {chunk_count} received, size={len(chunk)} bytes")
                yield chunk
            except Exception as e:
                if self._closed:
                    return
                print(f"[WARN] Error getting audio chunk: {e}")
                continue


def _build_streaming_config(
    language_code: str,
    interim_results: bool,
    model: Optional[str],
    sample_rate: int,
    phrase_hints: Optional[list[str]] = None,
) -> cloud_speech.StreamingRecognitionConfig:
    """
    Build streaming recognition config.
    
    Note: If model is None, the recognizer's default model will be used.
    This is recommended when using a recognizer that already has a model configured.
    
    Args:
        phrase_hints: List of phrases to boost recognition accuracy.
                     These are words or phrases that are likely to appear in the audio.
    """
    recognition_config_dict = {
        "explicit_decoding_config": cloud_speech.ExplicitDecodingConfig(
            encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            audio_channel_count=1,
        ),
        "language_codes": [language_code],
    }
    # Only set model if explicitly provided
    # Recognizer may already have a model configured, which takes precedence
    if model:
        recognition_config_dict["model"] = model
    
    # Note: Speech-to-Text v2 uses PhraseSet resources for phrase hints, not SpeechContext
    # Phrase hints are configured at the Recognizer level, not in RecognitionConfig
    # If phrase_hints are provided, we log a warning but don't add them to the config
    # To use phrase hints in v2, create a PhraseSet resource and reference it in the Recognizer
    # (Message is logged once in main.py, not here to avoid repetition)
    
    recognition_config = cloud_speech.RecognitionConfig(**recognition_config_dict)
    return cloud_speech.StreamingRecognitionConfig(
        config=recognition_config,
        streaming_features=cloud_speech.StreamingRecognitionFeatures(
            interim_results=interim_results
        ),
    )


def stream_microphone_transcripts(
    recognizer: str,
    language_code: str = "ko-KR",
    sample_rate: int = 16000,
    chunk_ms: int = 50,
    interim_results: bool = True,
    model: Optional[str] = None,
    location: str = "global",
    audio_device: Optional[int | str] = None,
    phrase_hints: Optional[list[str]] = None,
) -> Generator[STTResult, None, None]:
    """
    Stream mic audio to Speech-to-Text v2 via gRPC StreamingRecognize.

    Notes:
    - Speech-to-Text v2 StreamingRecognize is gRPC-only.
    - `recognizer` must be full resource name:
      projects/{PROJECT_ID}/locations/{LOCATION}/recognizers/{RECOGNIZER_ID}
    """
    # Keep endpoint location aligned with recognizer location to avoid 400 mismatch.
    location_from_recognizer = _extract_location_from_recognizer(recognizer)
    effective_location = location_from_recognizer or location or "global"
    if effective_location == "global":
        client = SpeechClient(transport="grpc")
    else:
        client = SpeechClient(
            transport="grpc",
            client_options={"api_endpoint": f"{effective_location}-speech.googleapis.com"},
        )
    streaming_config = _build_streaming_config(
        language_code=language_code,
        interim_results=interim_results,
        model=model,
        sample_rate=sample_rate,
        phrase_hints=phrase_hints,
    )

    with _MicrophoneStream(sample_rate=sample_rate, chunk_ms=chunk_ms, device=audio_device) as mic:
        audio_iter = mic.audio_chunks()
        
        # Give the audio stream a moment to start producing data
        # This helps ensure audio is flowing before we start the API stream
        time.sleep(0.1)

        def request_iter() -> Iterator[cloud_speech.StreamingRecognizeRequest]:
            # Send the initial config request
            yield cloud_speech.StreamingRecognizeRequest(
                recognizer=recognizer,
                streaming_config=streaming_config,
            )
            # Then stream audio chunks
            chunk_count = 0
            for chunk in audio_iter:
                chunk_count += 1
                yield cloud_speech.StreamingRecognizeRequest(audio=chunk)
                # Log first few chunks to verify audio is flowing
                if chunk_count <= 3:
                    print(f"[DEBUG] Sent audio chunk {chunk_count}, size={len(chunk)} bytes")

        try:
            responses = client.streaming_recognize(requests=request_iter())
        except Exception as e:
            error_str = str(e)
            if "403" in error_str or "permission" in error_str.lower() or "IAM_PERMISSION_DENIED" in error_str:
                print(f"[ERROR] Permission denied: {e}")
                print("[ERROR] Your service account is missing required permissions.")
                print("[ERROR] Required permission: 'speech.recognizers.recognize'")
                print("[ERROR]")
                
                # Extract project IDs for comparison
                import re
                recognizer_project_match = re.search(r"projects/([^/]+)/", recognizer)
                recognizer_project = recognizer_project_match.group(1) if recognizer_project_match else "unknown"
                
                print(f"[ERROR] IMPORTANT: Project ID mismatch detected!")
                print(f"[ERROR]   - Recognizer is in project: {recognizer_project}")
                print("[ERROR]   - Service account may be in a different project")
                print("[ERROR]")
                print("[ERROR] To fix this:")
                print(f"[ERROR]   1. Go to GCP Console -> IAM & Admin -> IAM for project '{recognizer_project}'")
                print("[ERROR]   2. Find your service account (from GOOGLE_APPLICATION_CREDENTIALS)")
                print("[ERROR]   3. Add one of these roles:")
                print("[ERROR]      - 'Speech-to-Text API User' (recommended)")
                print("[ERROR]      - 'Speech-to-Text Admin' (if you need to create recognizers too)")
                print("[ERROR]   4. Or grant custom role with 'speech.recognizers.recognize' permission")
                print("[ERROR]")
                print("[ERROR] NOTE: The service account must have permissions in the SAME project")
                print(f"[ERROR]       where the recognizer exists ({recognizer_project}), not just")
                print("[ERROR]       the project where the service account was created.")
                print("[ERROR]")
                print("[ERROR] After adding permissions, wait a few minutes for them to propagate.")
            elif "500" in error_str or "Internal error" in error_str:
                print(f"[ERROR] Google Cloud Speech-to-Text API error: {e}")
                print("[ERROR] This is a server-side error. Possible causes:")
                print("[ERROR]   1. Recognizer resource may not exist or be misconfigured")
                print(f"[ERROR]   2. Check if recognizer '{recognizer}' exists in GCP")
                print("[ERROR]   3. Verify your GCP project has Speech-to-Text API enabled")
                print("[ERROR]   4. Check billing and quota status in GCP Console")
                print("[ERROR]   5. Verify network connectivity to Google Cloud")
            elif "499" in error_str or "cancelled" in error_str.lower():
                print(f"[ERROR] gRPC connection cancelled: {e}")
                print("[ERROR] Possible causes:")
                print(f"[ERROR]   1. Recognizer '{recognizer}' may not exist or be inaccessible")
                print("[ERROR]   2. Check if the recognizer name has the correct project ID")
                print("[ERROR]   3. Verify the recognizer exists: python scripts/check_recognizer.py")
                print("[ERROR]   4. Check service account permissions")
                print("[ERROR]   5. Network/firewall issues blocking gRPC connection")
            raise

        response_count = 0
        for response in responses:
            response_count += 1
            if response_count <= 3:
                print(f"[DEBUG] Received response {response_count}, results count: {len(response.results)}")
            
            for result in response.results:
                if not result.alternatives:
                    if response_count <= 3:
                        print(f"[DEBUG] Result has no alternatives")
                    continue
                top = result.alternatives[0]
                text = top.transcript.strip()
                if not text:
                    if response_count <= 3:
                        print(f"[DEBUG] Result transcript is empty")
                    continue
                
                if response_count <= 3:
                    print(f"[DEBUG] Yielding STT result: is_final={result.is_final}, text='{text[:50]}...'")
                
                yield STTResult(
                    text=text,
                    is_final=result.is_final,
                    stability=float(result.stability),
                    language_code=result.language_code,
                )


def _extract_location_from_recognizer(recognizer: str) -> str:
    match = re.search(r"/locations/([^/]+)/", recognizer)
    if not match:
        return ""
    return match.group(1).strip()


if __name__ == "__main__":
    # Quick manual smoke run:
    # set GOOGLE_APPLICATION_CREDENTIALS
    # set GCP_RECOGNIZER=projects/.../locations/.../recognizers/...
    import os

    recognizer_name = os.getenv("GCP_RECOGNIZER", "").strip()
    if not recognizer_name:
        raise SystemExit("Missing GCP_RECOGNIZER")

    for item in stream_microphone_transcripts(recognizer=recognizer_name):
        tag = "FINAL" if item.is_final else "INTERIM"
        print(f"[{tag}] {item.text}")
