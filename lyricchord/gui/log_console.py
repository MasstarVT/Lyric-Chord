"""Read-only, colour-coded log console widget."""

from __future__ import annotations

import logging
import time

import customtkinter as ctk

LEVEL_COLORS = {
    "DEBUG": "#7f8c8d",
    "INFO": "#d0d7de",
    "WARNING": "#f5b041",
    "ERROR": "#ec7063",
    "CRITICAL": "#ec7063",
}
MAX_LINES = 2000


class LogConsole(ctk.CTkTextbox):
    def __init__(self, master, **kwargs):
        kwargs.setdefault("wrap", "word")
        kwargs.setdefault("font", ctk.CTkFont(family="Consolas", size=12))
        super().__init__(master, **kwargs)
        for level, color in LEVEL_COLORS.items():
            self.tag_config(level, foreground=color)
        self.configure(state="disabled")
        self._count = 0

    def append(self, record: logging.LogRecord) -> None:
        stamp = time.strftime("%H:%M:%S", time.localtime(record.created))
        text = f"{stamp}  {record.levelname[:4]:<4}  {record.getMessage()}\n"
        self.configure(state="normal")
        self.insert("end", text, record.levelname)
        self._count += 1
        if self._count > MAX_LINES:
            self.delete("1.0", "200.0")
            self._count -= 200
        self.see("end")
        self.configure(state="disabled")

    def clear(self) -> None:
        self.configure(state="normal")
        self.delete("1.0", "end")
        self.configure(state="disabled")
        self._count = 0
