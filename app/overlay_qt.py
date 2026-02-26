from __future__ import annotations

import sys
from dataclasses import dataclass

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget


@dataclass
class OverlaySettings:
    font_family: str = "Malgun Gothic"
    font_size: int = 36
    text_color: str = "#FFFFFF"
    background_opacity: float = 0.45  # 0.0 ~ 1.0
    padding_px: int = 20
    width_ratio: float = 0.85
    height_ratio: float = 0.12  # Reduced from 0.24 to half
    bottom_margin_px: int = 36


class SubtitleOverlay(QWidget):
    """
    Transparent always-on-top subtitle overlay.
    - Click-through is enabled so it does not steal input focus.
    - Font size / background opacity can be adjusted at runtime.
    """

    def __init__(self, settings: OverlaySettings | None = None) -> None:
        super().__init__()
        self.settings = settings or OverlaySettings()
        self._subtitle_text = ""

        self._configure_window()
        self._build_ui()
        self._apply_style()
        self._position_bottom_center()

    def _configure_window(self) -> None:
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def _build_ui(self) -> None:
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel("")
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)

        layout.addWidget(self.label)
        self.setLayout(layout)

    def _apply_style(self) -> None:
        s = self.settings
        alpha = int(max(0.0, min(1.0, s.background_opacity)) * 255)
        font = QFont(s.font_family, s.font_size)
        self.label.setFont(font)
        self.label.setStyleSheet(
            f"""
            QLabel {{
                color: {s.text_color};
                background-color: rgba(0, 0, 0, {alpha});
                border-radius: 14px;
                padding: {s.padding_px}px;
            }}
            """
        )

    def _position_bottom_center(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        width = int(geo.width() * max(0.2, min(1.0, self.settings.width_ratio)))
        height = int(geo.height() * max(0.1, min(0.9, self.settings.height_ratio)))
        x = geo.x() + (geo.width() - width) // 2
        y = geo.y() + geo.height() - height - max(0, self.settings.bottom_margin_px)
        self.setGeometry(x, y, width, height)

    def set_subtitle(self, text: str) -> None:
        self._subtitle_text = text.strip()
        self.label.setText(self._subtitle_text)
        self.show()

    def clear_subtitle(self) -> None:
        self._subtitle_text = ""
        self.label.clear()

    def set_font_size(self, font_size: int) -> None:
        self.settings.font_size = max(8, int(font_size))
        self._apply_style()

    def set_background_opacity(self, opacity: float) -> None:
        self.settings.background_opacity = max(0.0, min(1.0, float(opacity)))
        self._apply_style()

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        super().resizeEvent(event)
        self._apply_style()


def run_overlay(settings: OverlaySettings | None = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    overlay = SubtitleOverlay(settings=settings)
    overlay.show()
    return app.exec()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    overlay = SubtitleOverlay(
        OverlaySettings(
            font_size=34,
            background_opacity=0.5,
        )
    )
    overlay.set_subtitle("실시간 자막 오버레이 테스트입니다.")
    overlay.show()

    # Demo update
    QTimer.singleShot(2500, lambda: overlay.set_subtitle("폰트 크기와 배경 투명도를 조절할 수 있습니다."))
    QTimer.singleShot(5000, lambda: overlay.set_font_size(42))
    QTimer.singleShot(7000, lambda: overlay.set_background_opacity(0.25))

    raise SystemExit(app.exec())
