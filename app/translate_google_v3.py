from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from google.cloud import translate_v3 as translate


@dataclass
class TranslateConfig:
    project_id: str
    location: str = "global"
    target_language: str = "en"
    source_language: str = "ko"
    glossary_id: str = ""
    glossary_location: str = "global"


class GoogleTranslateV3:
    def __init__(self, config: TranslateConfig) -> None:
        if not config.project_id:
            raise ValueError("TranslateConfig.project_id is required")
        self.config = config
        self.client = translate.TranslationServiceClient()
        self.parent = f"projects/{config.project_id}/locations/{config.location}"

    def _glossary_path(self) -> str:
        cfg = self.config
        return (
            f"projects/{cfg.project_id}/locations/{cfg.glossary_location}"
            f"/glossaries/{cfg.glossary_id}"
        )

    def translate_text(self, text: str, use_glossary: bool = True) -> str:
        text = text.strip()
        if not text:
            return ""

        request: dict = {
            "parent": self.parent,
            "contents": [text],
            "mime_type": "text/plain",
            "source_language_code": self.config.source_language,
            "target_language_code": self.config.target_language,
        }

        glossary_enabled = bool(self.config.glossary_id) and use_glossary
        if glossary_enabled:
            request["glossary_config"] = {
                "glossary": self._glossary_path(),
                "ignore_case": True,
            }

        response = self.client.translate_text(request=request)

        if glossary_enabled and response.glossary_translations:
            return response.glossary_translations[0].translated_text
        if response.translations:
            return response.translations[0].translated_text
        return ""


def build_translator_from_env(
    project_id: str,
    location: str = "global",
    source_language: str = "ko",
    target_language: str = "en",
    glossary_id: str = "",
    glossary_location: str = "global",
) -> GoogleTranslateV3:
    cfg = TranslateConfig(
        project_id=project_id,
        location=location,
        source_language=source_language,
        target_language=target_language,
        glossary_id=glossary_id,
        glossary_location=glossary_location,
    )
    return GoogleTranslateV3(cfg)
