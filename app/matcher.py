from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from rapidfuzz import fuzz

try:
    from .translate_google_v3 import GoogleTranslateV3
except ImportError:
    from translate_google_v3 import GoogleTranslateV3


_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w가-힣]+")


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text)
    return text


@dataclass
class ScriptEntry:
    line_id: str
    ko: str
    en: str
    keywords: list[str] = field(default_factory=list)

    @property
    def ko_norm(self) -> str:
        return _normalize(self.ko)


@dataclass
class MatchResult:
    entry: ScriptEntry
    score: float
    method: str


@dataclass
class ResolveResult:
    mode: str  # matched | translated
    output_text: str
    score: float = 0.0
    matched_id: str = ""
    method: str = ""


@dataclass
class MatcherConfig:
    window_back: int = 3
    window_forward: int = 20
    threshold: float = 78.0
    global_threshold_delta: float = 3.0
    wratio_weight: float = 0.65
    token_weight: float = 0.25
    keyword_weight: float = 0.10


class ScriptMatcher:
    def __init__(self, entries: list[ScriptEntry], config: MatcherConfig | None = None) -> None:
        self.entries = entries
        self.config = config or MatcherConfig()
        self.current_index = 0

    @classmethod
    def from_csv(cls, csv_path: str | Path, config: MatcherConfig | None = None) -> ScriptMatcher:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Script CSV not found: {path}")

        entries: list[ScriptEntry] = []
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ko = (row.get("ko") or "").strip()
                en = (row.get("en") or "").strip()
                if not ko or not en:
                    continue
                raw_keywords = (row.get("keywords") or "").strip()
                keywords = [k.strip() for k in raw_keywords.split(";") if k.strip()]
                entries.append(
                    ScriptEntry(
                        line_id=(row.get("id") or "").strip(),
                        ko=ko,
                        en=en,
                        keywords=keywords,
                    )
                )
        return cls(entries=entries, config=config)

    def _window_indices(self) -> range:
        if not self.entries:
            return range(0, 0)
        start = max(0, self.current_index - self.config.window_back)
        end = min(len(self.entries), self.current_index + self.config.window_forward + 1)
        return range(start, end)

    def _keyword_score(self, transcript_norm: str, keywords: Iterable[str]) -> float:
        keys = [_normalize(k) for k in keywords if k.strip()]
        if not keys:
            return 0.0
        hits = sum(1 for k in keys if k and k in transcript_norm)
        return 100.0 * hits / max(1, len(keys))

    def _score(self, transcript_norm: str, entry: ScriptEntry) -> tuple[float, str]:
        wratio = float(fuzz.WRatio(transcript_norm, entry.ko_norm))
        token = float(fuzz.token_set_ratio(transcript_norm, entry.ko_norm))
        partial = float(fuzz.partial_ratio(transcript_norm, entry.ko_norm))
        keyword = self._keyword_score(transcript_norm, entry.keywords)
        cfg = self.config
        # Keep overall scale comparable to 0~100 while making partial utterances easier to match.
        wratio_w = max(0.0, cfg.wratio_weight - 0.10)
        token_w = max(0.0, cfg.token_weight - 0.05)
        keyword_w = cfg.keyword_weight
        partial_w = 1.0 - (wratio_w + token_w + keyword_w)
        total = (
            wratio * wratio_w
            + token * token_w
            + partial * max(0.0, partial_w)
            + keyword * keyword_w
        )
        method = "hybrid(wratio+token+partial+keyword)"
        return total, method

    def _best_in_indices(self, transcript_norm: str, indices: Iterable[int]) -> tuple[int, float, str]:
        best_idx = -1
        best_score = -1.0
        best_method = ""
        for idx in indices:
            score, method = self._score(transcript_norm, self.entries[idx])
            if score > best_score:
                best_score = score
                best_idx = idx
                best_method = method
        return best_idx, best_score, best_method

    def match(self, transcript_ko: str) -> Optional[MatchResult]:
        if not self.entries:
            return None
        transcript_norm = _normalize(transcript_ko)
        if not transcript_norm:
            return None

        best_idx = -1
        best_score = -1.0
        best_method = ""

        window_range = list(self._window_indices())
        
        # Fast path: Check for exact or near-exact matches first
        # This significantly speeds up matching when transcript matches script exactly
        for idx in window_range:
            entry = self.entries[idx]
            # Quick exact match check (case-insensitive, normalized)
            if transcript_norm == entry.ko_norm:
                self.current_index = min(idx + 1, len(self.entries) - 1)
                return MatchResult(
                    entry=entry,
                    score=100.0,
                    method="exact_match"
                )
            
            # Quick substring check for very high similarity
            if len(transcript_norm) > 10 and entry.ko_norm in transcript_norm:
                # If script text is contained in transcript, likely a match
                score, method = self._score(transcript_norm, entry)
                if score >= 95.0:  # Very high confidence
                    self.current_index = min(idx + 1, len(self.entries) - 1)
                    return MatchResult(entry=entry, score=score, method=method)

        # Standard matching in local window first.
        best_idx, best_score, best_method = self._best_in_indices(transcript_norm, window_range)

        if best_idx < 0 or best_score < self.config.threshold:
            # Recovery path when speaker jumps slides or index drifts.
            global_idx, global_score, global_method = self._best_in_indices(
                transcript_norm, range(len(self.entries))
            )
            global_threshold = self.config.threshold + self.config.global_threshold_delta
            if global_idx < 0 or global_score < global_threshold:
                return None
            best_idx = global_idx
            best_score = global_score
            best_method = f"{global_method}+global"

        self.current_index = min(best_idx + 1, len(self.entries) - 1)
        return MatchResult(entry=self.entries[best_idx], score=best_score, method=best_method)


class SubtitleResolver:
    """
    Resolve Korean transcript into English subtitle:
    1) Try script matching inside a moving window.
    2) If no confident match, fallback to Cloud Translation v3.
    """

    def __init__(
        self,
        matcher: ScriptMatcher,
        translator: Optional[GoogleTranslateV3] = None,
        use_glossary_on_fallback: bool = True,
    ) -> None:
        self.matcher = matcher
        self.translator = translator
        self.use_glossary_on_fallback = use_glossary_on_fallback
        self._last_matched_entry: Optional[ScriptEntry] = None
        self._sticky_partial_threshold = 92.0
        self._sticky_wratio_threshold = 88.0

    def resolve(self, transcript_ko: str) -> ResolveResult:
        match = self.matcher.match(transcript_ko)
        if match is not None:
            self._last_matched_entry = match.entry
            return ResolveResult(
                mode="matched",
                output_text=match.entry.en,
                score=match.score,
                matched_id=match.entry.line_id,
                method=match.method,
            )

        # Guard against subtitle flicker: when a follow-up final chunk from the same
        # spoken sentence arrives, keep the previous matched subtitle instead of
        # replacing it with fallback MT output.
        if self._last_matched_entry is not None:
            transcript_norm = _normalize(transcript_ko)
            if transcript_norm:
                prev_norm = self._last_matched_entry.ko_norm
                sticky_partial = float(fuzz.partial_ratio(transcript_norm, prev_norm))
                sticky_wratio = float(fuzz.WRatio(transcript_norm, prev_norm))
                if (
                    sticky_partial >= self._sticky_partial_threshold
                    or sticky_wratio >= self._sticky_wratio_threshold
                ):
                    return ResolveResult(
                        mode="matched_sticky",
                        output_text=self._last_matched_entry.en,
                        score=max(sticky_partial, sticky_wratio),
                        matched_id=self._last_matched_entry.line_id,
                        method="sticky_previous_match",
                    )

        if self.translator is None:
            return ResolveResult(mode="translated", output_text="")

        translated = self.translator.translate_text(
            transcript_ko, use_glossary=self.use_glossary_on_fallback
        )
        return ResolveResult(mode="translated", output_text=translated)
