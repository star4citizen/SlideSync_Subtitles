from __future__ import annotations

import argparse
import os
import queue
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication
import sounddevice as sd

try:
    from .matcher import MatcherConfig, ScriptMatcher, SubtitleResolver
    from .overlay_qt import OverlaySettings, SubtitleOverlay
    from .stt_google_v2 import STTResult, stream_microphone_transcripts, validate_audio_device
    from .translate_google_v3 import TranslateConfig, GoogleTranslateV3
except ImportError:
    from matcher import MatcherConfig, ScriptMatcher, SubtitleResolver
    from overlay_qt import OverlaySettings, SubtitleOverlay
    from stt_google_v2 import STTResult, stream_microphone_transcripts, validate_audio_device
    from translate_google_v3 import TranslateConfig, GoogleTranslateV3


@dataclass
class PipelineConfig:
    recognizer: str
    script_path: str
    project_id: str
    stt_location: str = "global"
    translate_location: str = "global"
    glossary_id: str = ""
    glossary_location: str = "global"
    language_code: str = "ko-KR"
    model: str = "long"
    chunk_ms: int = 50
    font_size: int = 36
    background_opacity: float = 0.45
    match_threshold: float = 78.0
    show_interim: bool = False
    disable_translate_fallback: bool = False
    audio_device: str = ""
    list_audio_devices: bool = False
    phrase_hints: Optional[list[str]] = None


def _ms(start: float, end: float) -> float:
    return (end - start) * 1000.0


def _build_pipeline_config() -> PipelineConfig:
    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(dotenv_path=repo_root / ".env")

    parser = argparse.ArgumentParser(description="Live subtitle pipeline (STT -> match/translate -> overlay)")
    parser.add_argument("--recognizer", default=os.getenv("GCP_RECOGNIZER", ""))
    parser.add_argument("--script-path", default=os.getenv("SCRIPT_PATH", "data/script.csv"))
    parser.add_argument("--project-id", default=os.getenv("GCP_PROJECT_ID", ""))
    parser.add_argument("--stt-location", default=os.getenv("GCP_LOCATION", "global"))
    parser.add_argument(
        "--translate-location",
        default=os.getenv("GCP_TRANSLATE_LOCATION", os.getenv("GCP_LOCATION", "global")),
    )
    parser.add_argument("--glossary-id", default=os.getenv("GCP_GLOSSARY_ID", ""))
    parser.add_argument("--glossary-location", default=os.getenv("GCP_TRANSLATE_LOCATION", "global"))
    parser.add_argument("--language-code", default="ko-KR")
    parser.add_argument("--model", default="long")
    parser.add_argument("--chunk-ms", type=int, default=50)
    parser.add_argument("--font-size", type=int, default=36)
    parser.add_argument("--bg-opacity", type=float, default=0.45)
    parser.add_argument("--match-threshold", type=float, default=78.0)
    parser.add_argument("--show-interim", action="store_true")
    parser.add_argument("--disable-translate-fallback", action="store_true")
    parser.add_argument("--audio-device", default=os.getenv("AUDIO_DEVICE", ""))
    parser.add_argument("--list-audio-devices", action="store_true")
    parser.add_argument(
        "--phrase-hints",
        default=os.getenv("PHRASE_HINTS", ""),
        help="Comma-separated list of phrases to boost recognition (e.g., '용어1,용어2,용어3')",
    )
    args = parser.parse_args()

    if not args.recognizer:
        raise SystemExit("Missing recognizer. Set GCP_RECOGNIZER or pass --recognizer")
    script_path = Path(args.script_path)
    if not script_path.is_absolute():
        # Resolve relative paths from repository root, not current working directory.
        script_path = repo_root / script_path
    if not script_path.exists():
        raise SystemExit(f"Script CSV not found: {script_path}")

    project_id = args.project_id.strip()
    if not project_id:
        project_id = _extract_project_id_from_recognizer(args.recognizer)
    audio_device = args.audio_device.strip()
    if audio_device.isdigit():
        audio_device = str(int(audio_device))
    
    # Parse phrase hints from comma-separated string
    phrase_hints: list[str] = []
    if args.phrase_hints:
        phrase_hints = [h.strip() for h in args.phrase_hints.split(",") if h.strip()]

    return PipelineConfig(
        recognizer=args.recognizer,
        script_path=str(script_path),
        project_id=project_id,
        stt_location=args.stt_location,
        translate_location=args.translate_location,
        glossary_id=args.glossary_id,
        glossary_location=args.glossary_location,
        language_code=args.language_code,
        model=args.model,
        chunk_ms=args.chunk_ms,
        font_size=args.font_size,
        background_opacity=max(0.0, min(1.0, args.bg_opacity)),
        match_threshold=args.match_threshold,
        show_interim=args.show_interim,
        disable_translate_fallback=args.disable_translate_fallback,
        audio_device=audio_device,
        list_audio_devices=args.list_audio_devices,
        phrase_hints=phrase_hints,
    )


def _extract_project_id_from_recognizer(recognizer: str) -> str:
    match = re.search(r"projects/([^/]+)/locations/", recognizer or "")
    if not match:
        return ""
    return match.group(1).strip()


def _parse_audio_device(value: str) -> Optional[int | str]:
    v = (value or "").strip()
    if not v:
        return None
    if v.isdigit():
        return int(v)
    return v


def _default_input_device_index() -> Optional[int]:
    try:
        dev = sd.default.device
        if isinstance(dev, (tuple, list)) and len(dev) >= 1:
            idx = int(dev[0])
            if idx >= 0:
                return idx
    except Exception:
        return None
    return None


def _first_input_device_index() -> Optional[int]:
    try:
        devices = sd.query_devices()
        scored: list[tuple[int, int]] = []
        for i, d in enumerate(devices):
            if int(d.get("max_input_channels", 0)) <= 0:
                continue
            name = str(d.get("name", "")).lower()
            score = 0
            if "microphone" in name or "mic" in name:
                score += 5
            if "array" in name:
                score += 2
            if "sound mapper" in name:
                score -= 6
            if "stereo mix" in name:
                score -= 6
            if "pc speaker" in name:
                score -= 6
            if name.strip() in {"input ()", "input"}:
                score -= 4
            scored.append((score, i))
        if not scored:
            return None
        scored.sort(reverse=True)
        return scored[0][1]
    except Exception:
        return None
    return None


def _print_audio_devices() -> None:
    print("[AUDIO] available input devices:")
    try:
        devices = sd.query_devices()
        default_idx = _default_input_device_index()
        for i, d in enumerate(devices):
            in_ch = int(d.get("max_input_channels", 0))
            if in_ch <= 0:
                continue
            marker = "*" if default_idx is not None and i == default_idx else " "
            print(f"[AUDIO] {marker} idx={i} in_ch={in_ch} name={d.get('name')}")
    except Exception as exc:
        print(f"[WARN] failed to query audio devices: {exc}")


def _extract_phrase_hints_from_script(script_path: str) -> list[str]:
    """Extract all keywords from script.csv to use as phrase hints."""
    import csv
    from pathlib import Path
    
    path = Path(script_path)
    if not path.exists():
        return []
    
    phrase_hints: set[str] = set()
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_keywords = (row.get("keywords") or "").strip()
                if raw_keywords:
                    keywords = [k.strip() for k in raw_keywords.split(";") if k.strip()]
                    phrase_hints.update(keywords)
    except Exception as exc:
        print(f"[WARN] Failed to extract phrase hints from script: {exc}")
        return []
    
    return sorted(list(phrase_hints))


def _build_resolver(cfg: PipelineConfig) -> tuple[SubtitleResolver, list[str]]:
    matcher = ScriptMatcher.from_csv(
        cfg.script_path, config=MatcherConfig(threshold=cfg.match_threshold)
    )

    translator: Optional[GoogleTranslateV3] = None
    if not cfg.disable_translate_fallback:
        if not cfg.project_id:
            print("[WARN] GCP_PROJECT_ID missing. Translation fallback disabled.")
        else:
            try:
                translator = GoogleTranslateV3(
                    TranslateConfig(
                        project_id=cfg.project_id,
                        location=cfg.translate_location,
                        source_language="ko",
                        target_language="en",
                        glossary_id=cfg.glossary_id,
                        glossary_location=cfg.glossary_location,
                    )
                )
            except Exception as exc:
                print(f"[WARN] Translation fallback init failed, disabling fallback: {exc}")

    resolver = SubtitleResolver(
        matcher=matcher,
        translator=translator,
        use_glossary_on_fallback=bool(cfg.glossary_id),
    )
    
    # Extract phrase hints from script.csv keywords
    script_phrase_hints = _extract_phrase_hints_from_script(cfg.script_path)
    
    # Merge with user-provided phrase hints (user hints take precedence)
    all_phrase_hints = list(script_phrase_hints)
    if cfg.phrase_hints:
        # Add user hints, avoiding duplicates
        for hint in cfg.phrase_hints:
            if hint not in all_phrase_hints:
                all_phrase_hints.append(hint)
    
    return resolver, all_phrase_hints


def run() -> int:
    cfg = _build_pipeline_config()
    if cfg.list_audio_devices:
        _print_audio_devices()
        return 0
    resolver, phrase_hints = _build_resolver(cfg)
    if phrase_hints:
        print(f"[INFO] Using {len(phrase_hints)} phrase hints from script.csv keywords")
        if len(phrase_hints) <= 10:
            print(f"[INFO] Phrase hints: {', '.join(phrase_hints)}")
        else:
            print(f"[INFO] Phrase hints (first 10): {', '.join(phrase_hints[:10])}...")
    
    device_for_log = _parse_audio_device(cfg.audio_device)
    if device_for_log is None:
        device_for_log = _default_input_device_index()
    if device_for_log is None:
        device_for_log = _first_input_device_index()
    if device_for_log is None:
        device_for_log = "default"
    print(
        "[BOOT] "
        f"recognizer={cfg.recognizer} "
        f"language={cfg.language_code} "
        f"chunk_ms={cfg.chunk_ms} "
        f"script_path={cfg.script_path} "
        f"audio_device={device_for_log}"
    )
    _print_audio_devices()
    
    # Validate audio device before starting
    device_to_validate = device_for_log if device_for_log != "default" else None
    is_valid, error_msg = validate_audio_device(device_to_validate)
    if not is_valid:
        print(f"[ERROR] Audio device validation failed: {error_msg}")
        print("[ERROR] Please check:")
        print("[ERROR]   1. The device index is correct (use --list-audio-devices to see available devices)")
        print("[ERROR]   2. The device is not in use by another application")
        print("[ERROR]   3. The device has proper permissions in Windows settings")
        print("[ERROR]   4. Try using a different device index or omit --audio-device to use default")
        return 1
    
    # Warn about recognizer validation
    print(f"[INFO] Using recognizer: {cfg.recognizer}")
    print("[INFO] If you encounter 499/500 errors, verify the recognizer exists:")
    print("[INFO]   python scripts/check_recognizer.py")
    print("[INFO] Make sure the project ID in the recognizer name matches your actual GCP project.")

    app = QApplication.instance() or QApplication([])
    overlay = SubtitleOverlay(
        OverlaySettings(
            font_size=cfg.font_size,
            background_opacity=cfg.background_opacity,
        )
    )
    overlay.show()

    ui_queue: queue.Queue[str] = queue.Queue(maxsize=200)
    stop_event = threading.Event()
    last_stt_event_t = [time.perf_counter()]

    def worker() -> None:
        retry_count = 0
        while not stop_event.is_set():
            try:
                print(
                    "[STT] connecting "
                    f"retry={retry_count} "
                    f"location={cfg.stt_location} "
                    f"audio_device={device_for_log}"
                )
                # Use None for model to let recognizer use its configured model
                # This avoids conflicts when recognizer has a different model (e.g., chirp_3)
                model_param = None if cfg.model == "long" else cfg.model
                for stt in stream_microphone_transcripts(
                    recognizer=cfg.recognizer,
                    language_code=cfg.language_code,
                    sample_rate=16000,
                    chunk_ms=cfg.chunk_ms,
                    interim_results=True,
                    model=model_param,
                    location=cfg.stt_location,
                    audio_device=device_for_log if device_for_log != "default" else None,
                    phrase_hints=phrase_hints if phrase_hints else None,
                ):
                    if stop_event.is_set():
                        break
                    stt_t = time.perf_counter()
                    last_stt_event_t[0] = stt_t
                    _handle_stt_result(stt, resolver, ui_queue, cfg, stt_t)
                if stop_event.is_set():
                    break
                print("[WARN] STT stream ended. Reconnecting in 1s.")
            except ValueError as exc:
                # Device validation errors - don't retry, exit with clear message
                if "Invalid audio device" in str(exc) or "Failed to open audio device" in str(exc):
                    print(f"[ERROR] Audio device error: {exc}")
                    print("[ERROR] Please check:")
                    print("[ERROR]   1. The device index is correct (use --list-audio-devices to see available devices)")
                    print("[ERROR]   2. The device is not in use by another application")
                    print("[ERROR]   3. The device has proper permissions in Windows settings")
                    print("[ERROR] Exiting. Fix the device issue and try again.")
                    stop_event.set()
                    break
                raise
            except Exception as exc:  # pragma: no cover
                print(f"[ERROR] STT worker failed: {exc}")
                if "499" in str(exc):
                    print("[WARN] gRPC 499 cancelled. Will reconnect.")
                elif "500" in str(exc) or "Internal error" in str(exc):
                    print("[WARN] Google Cloud API 500 error. This may be temporary. Will retry.")
                    print("[WARN] If this persists, check:")
                    print("[WARN]   1. Your GCP project quota and billing status")
                    print("[WARN]   2. The recognizer resource is properly configured")
                    print("[WARN]   3. Network connectivity to Google Cloud")
                elif "Failed to open audio device" in str(exc) or "Invalid device" in str(exc):
                    print("[ERROR] Audio device error detected. Will retry, but device may need to be fixed.")
            retry_count += 1
            time.sleep(1.0)

    def poll_ui_queue() -> None:
        # Keep latest subtitle if queue builds up.
        try:
            latest = None
            while True:
                try:
                    latest = ui_queue.get_nowait()
                except queue.Empty:
                    break
            if latest is not None:
                ui_t0 = time.perf_counter()
                overlay.set_subtitle(latest)
                ui_t1 = time.perf_counter()
                print(f"[LATENCY] ui_update_ms={_ms(ui_t0, ui_t1):.1f}")
            idle_ms = _ms(last_stt_event_t[0], time.perf_counter())
            if idle_ms > 5000:
                print("[WARN] no STT results for 5s. Check mic input device/volume/permissions.")
                last_stt_event_t[0] = time.perf_counter()
        except Exception as exc:
            # Don't let UI polling errors crash the app
            print(f"[ERROR] UI polling error: {exc}")

    timer = QTimer()
    timer.timeout.connect(poll_ui_queue)
    timer.start(30)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    try:
        return app.exec()
    finally:
        stop_event.set()
        timer.stop()


def _handle_stt_result(
    stt: STTResult,
    resolver: SubtitleResolver,
    ui_queue: queue.Queue[str],
    cfg: PipelineConfig,
    stt_t: float,
) -> None:
    tag = "FINAL" if stt.is_final else "INTERIM"
    text = stt.text.strip()
    if not text:
        return

    resolve_t0 = time.perf_counter()
    
    # Try matching even for interim results to get faster response
    # This allows showing matched subtitles immediately without waiting for final result
    resolved = resolver.resolve(text)
    resolve_t1 = time.perf_counter()
    total_t1 = resolve_t1

    if not stt.is_final:
        print(f"[STT] type={tag} text={text}")
        # For interim results, only show if matched (to avoid flickering)
        # or if show_interim is enabled
        if resolved.mode == "matched" or cfg.show_interim:
            out = resolved.output_text.strip() if resolved.output_text.strip() else text
            _queue_latest(ui_queue, out)
        return

    out = resolved.output_text.strip()
    if not out:
        # Keep subtitles visible even if translation fails at runtime.
        out = text
    _queue_latest(ui_queue, out)

    print(
        "[PIPELINE] "
        f"mode={resolved.mode} "
        f"score={resolved.score:.1f} "
        f"matched_id={resolved.matched_id or '-'} "
        f"method={resolved.method or '-'} "
        f"stt_text={text} "
        f"subtitle={out or '(empty)'}"
    )
    print(
        "[LATENCY] "
        f"resolve_ms={_ms(resolve_t0, resolve_t1):.1f} "
        f"from_stt_to_resolve_ms={_ms(stt_t, resolve_t1):.1f} "
        f"pipeline_total_ms={_ms(stt_t, total_t1):.1f}"
    )


def _queue_latest(ui_queue: queue.Queue[str], text: str) -> None:
    try:
        ui_queue.put_nowait(text)
        return
    except queue.Full:
        pass
    try:
        _ = ui_queue.get_nowait()
    except queue.Empty:
        pass
    try:
        ui_queue.put_nowait(text)
    except queue.Full:
        pass


if __name__ == "__main__":
    raise SystemExit(run())
