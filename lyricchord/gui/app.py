"""
Main application window.

Threading model: all pipeline work runs in `BatchRunner` (a background thread).
It communicates back through two queues - one for log records and one for
progress events - which the window drains every 100 ms from the Tk main loop.
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Dict, List, Optional

import customtkinter as ctk

from ..config import Settings
from ..pipeline.processor import BatchRunner, find_audio_files
from ..utils.audio import AUDIO_EXTENSIONS
from ..utils.ffmpeg import FFmpegNotFound, ffmpeg_version
from ..utils.logging_utils import get_logger, setup_logging
from .log_console import LogConsole
from .settings_panel import SettingsPanel

try:  # drag & drop is optional; the Add buttons always work
    from tkinterdnd2 import DND_FILES, TkinterDnD
    _DND_IMPORT_OK = True
except Exception:  # pragma: no cover - missing native library
    DND_FILES, TkinterDnD = None, None
    _DND_IMPORT_OK = False

log = get_logger()

AUDIO_FILETYPES = [("Audio files", " ".join(f"*{ext}" for ext in sorted(AUDIO_EXTENSIONS))), ("All files", "*.*")]
STATUS_ICON = {"pending": "   ", "running": "▶ ", "done": "✓ ", "error": "✗ "}

_Base = (ctk.CTk, TkinterDnD.DnDWrapper) if _DND_IMPORT_OK else (ctk.CTk,)


class App(*_Base):  # type: ignore[misc]
    POLL_MS = 100

    def __init__(self) -> None:
        self.settings = Settings.load()
        ctk.set_appearance_mode(self.settings.appearance_mode)
        ctk.set_default_color_theme("blue")
        super().__init__()

        self.dnd_available = False
        if _DND_IMPORT_OK:
            try:
                self.TkdndVersion = TkinterDnD._require(self)
                self.dnd_available = True
            except Exception as exc:  # tkdnd Tcl package failed to load
                log.warning("Drag & drop unavailable: %s", exc)

        self.title("LyricChord - Play-Along Video Generator")
        self.geometry("1360x880")
        self.minsize(1100, 720)

        self.log_queue = setup_logging()
        self.events: "queue.Queue" = queue.Queue()
        self.cancel_event = threading.Event()
        self.runner: Optional[BatchRunner] = None
        self.files: List[Path] = []
        self.status: Dict[Path, str] = {}

        self._build()
        self._check_ffmpeg()
        self.after(self.POLL_MS, self._poll)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        log.info("Ready. Drop audio files or folders, adjust settings, then press Start batch.")

    # ------------------------------------------------------------------ UI construction
    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(1, weight=1)

        # Header --------------------------------------------------------
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=(10, 4))
        ctk.CTkLabel(header, text="LyricChord", font=ctk.CTkFont(size=22, weight="bold")).pack(side="left")
        ctk.CTkLabel(header, text="Play-along lyric + chord video generator",
                     text_color=("gray40", "gray60")).pack(side="left", padx=12)
        self.appearance_menu = ctk.CTkOptionMenu(header, values=["dark", "light", "system"], width=100,
                                                 command=self._set_appearance)
        self.appearance_menu.set(self.settings.appearance_mode)
        self.appearance_menu.pack(side="right")
        self.ffmpeg_label = ctk.CTkLabel(header, text="FFmpeg: checking...", text_color=("gray40", "gray60"))
        self.ffmpeg_label.pack(side="right", padx=12)

        # Left column: drop zone + file list -------------------------------
        left = ctk.CTkFrame(self)
        left.grid(row=1, column=0, sticky="nsew", padx=(12, 6), pady=4)
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        self.drop_zone = ctk.CTkFrame(left, height=90, corner_radius=12, border_width=2,
                                      border_color=("gray65", "gray35"))
        self.drop_zone.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 6))
        self.drop_zone.grid_propagate(False)
        self.drop_label = ctk.CTkLabel(self.drop_zone, text="Drop MP3 files or folders here",
                                       font=ctk.CTkFont(size=16))
        self.drop_label.place(relx=0.5, rely=0.5, anchor="center")

        self.listbox = tk.Listbox(left, selectmode="extended", activestyle="none", bd=0,
                                  highlightthickness=0, font=("Segoe UI", 11))
        self.listbox.grid(row=1, column=0, sticky="nsew", padx=(10, 0))
        scrollbar = ctk.CTkScrollbar(left, command=self.listbox.yview)
        scrollbar.grid(row=1, column=1, sticky="ns", padx=(0, 10))
        self.listbox.configure(yscrollcommand=scrollbar.set)
        self._style_listbox()

        buttons = ctk.CTkFrame(left, fg_color="transparent")
        buttons.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=8)
        ctk.CTkButton(buttons, text="Add files", width=100, command=self._add_files).pack(side="left", padx=(0, 6))
        ctk.CTkButton(buttons, text="Add folder", width=100, command=self._add_folder).pack(side="left", padx=6)
        ctk.CTkButton(buttons, text="Remove selected", width=130, fg_color="gray40", hover_color="gray30",
                      command=self._remove_selected).pack(side="left", padx=6)
        ctk.CTkButton(buttons, text="Clear", width=80, fg_color="gray40", hover_color="gray30",
                      command=self._clear).pack(side="left", padx=6)
        self.count_label = ctk.CTkLabel(buttons, text="0 files", text_color=("gray40", "gray60"))
        self.count_label.pack(side="right")

        # Right column: settings ------------------------------------------
        self.settings_panel = SettingsPanel(self, self.settings, width=470)
        self.settings_panel.grid(row=1, column=1, sticky="nsew", padx=(6, 12), pady=4)

        # Bottom: progress + actions --------------------------------------
        bottom = ctk.CTkFrame(self)
        bottom.grid(row=2, column=0, columnspan=2, sticky="ew", padx=12, pady=4)
        self.progress = ctk.CTkProgressBar(bottom)
        self.progress.set(0)
        self.progress.pack(fill="x", padx=10, pady=(10, 4))
        row = ctk.CTkFrame(bottom, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=(0, 10))
        self.status_label = ctk.CTkLabel(row, text="Ready", anchor="w")
        self.status_label.pack(side="left", fill="x", expand=True)
        self.start_btn = ctk.CTkButton(row, text="Start batch", width=140, command=self._start)
        self.start_btn.pack(side="right", padx=4)
        self.cancel_btn = ctk.CTkButton(row, text="Cancel", width=100, state="disabled", fg_color="gray40",
                                        hover_color="gray30", command=self._cancel)
        self.cancel_btn.pack(side="right", padx=4)
        ctk.CTkButton(row, text="Preview layout", width=130, fg_color="gray40", hover_color="gray30",
                      command=self._preview).pack(side="right", padx=4)
        ctk.CTkButton(row, text="Open output folder", width=150, fg_color="gray40", hover_color="gray30",
                      command=self._open_output).pack(side="right", padx=4)

        # Log console ----------------------------------------------------
        self.console = LogConsole(self, height=190)
        self.console.grid(row=3, column=0, columnspan=2, sticky="ew", padx=12, pady=(4, 12))

        # Drag & drop registration (whole window accepts drops) -----------
        if self.dnd_available:
            self.drop_target_register(DND_FILES)
            self.dnd_bind("<<Drop>>", self._on_drop)
            self.dnd_bind("<<DropEnter>>", lambda e: self.drop_zone.configure(border_color="#1f6aa5"))
            self.dnd_bind("<<DropLeave>>", lambda e: self.drop_zone.configure(border_color=("gray65", "gray35")))
        else:
            self.drop_label.configure(text="Drag & drop unavailable - use the Add buttons below")

    def _style_listbox(self) -> None:
        dark = ctk.get_appearance_mode() == "Dark"
        self.listbox.configure(bg="#2b2b2b" if dark else "#f5f5f7", fg="#dce4ee" if dark else "#1a1a1a",
                               selectbackground="#1f6aa5", selectforeground="#ffffff")

    # ------------------------------------------------------------------ file list
    def _add_paths(self, raw_paths) -> None:
        recursive = self.settings_panel.get_bool("scan_subfolders")
        found = find_audio_files([str(p) for p in raw_paths], recursive=recursive)
        existing = {str(f.resolve()).lower() for f in self.files}
        added = 0
        for f in found:
            key = str(f.resolve()).lower()
            if key not in existing:
                self.files.append(f)
                self.status[f] = "pending"
                existing.add(key)
                added += 1
        if added:
            log.info("Added %d file(s); %d in queue", added, len(self.files))
        elif raw_paths:
            log.warning("No new audio files found in the dropped items")
        self._refresh_list()

    def _refresh_list(self) -> None:
        self.listbox.delete(0, "end")
        for f in self.files:
            self.listbox.insert("end", STATUS_ICON.get(self.status.get(f, "pending"), "   ") + f.name)
        self.count_label.configure(text=f"{len(self.files)} file{'s' if len(self.files) != 1 else ''}")

    def _on_drop(self, event) -> None:
        try:
            items = self.tk.splitlist(event.data)
        except tk.TclError:
            items = [event.data]
        self.drop_zone.configure(border_color=("gray65", "gray35"))
        self._add_paths(items)

    def _add_files(self) -> None:
        files = filedialog.askopenfilenames(parent=self, title="Choose audio files", filetypes=AUDIO_FILETYPES)
        if files:
            self._add_paths(files)

    def _add_folder(self) -> None:
        d = filedialog.askdirectory(parent=self, title="Choose a folder of audio files")
        if d:
            self._add_paths([d])

    def _remove_selected(self) -> None:
        if self._busy():
            return
        for idx in sorted(self.listbox.curselection(), reverse=True):
            self.status.pop(self.files[idx], None)
            del self.files[idx]
        self._refresh_list()

    def _clear(self) -> None:
        if self._busy():
            return
        self.files.clear()
        self.status.clear()
        self._refresh_list()

    # ------------------------------------------------------------------ batch control
    def _busy(self) -> bool:
        return self.runner is not None and self.runner.is_alive()

    def _collect_settings(self) -> Settings:
        self.settings_panel.apply_to(self.settings)
        self.settings.appearance_mode = self.appearance_menu.get()
        return self.settings

    def _start(self) -> None:
        if self._busy():
            return
        if not self.files:
            messagebox.showinfo("Nothing to do", "Add some audio files first.", parent=self)
            return
        settings = self._collect_settings()
        settings.save()
        try:
            Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            messagebox.showerror("Output folder", f"Cannot create output folder:\n{exc}", parent=self)
            return
        if settings.background_style == "loop" and not Path(settings.loop_video_path).is_file():
            if not messagebox.askyesno("Loop video missing",
                                       "The loop background video was not found. Continue with a gradient instead?",
                                       parent=self):
                return

        self.cancel_event.clear()
        for f in self.files:
            self.status[f] = "pending"
        self._refresh_list()
        self.progress.set(0)
        self.start_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.status_label.configure(text=f"Starting batch of {len(self.files)} file(s)...")
        # The worker gets its own copy: the panel keeps mutating self.settings on the Tk thread.
        self.runner = BatchRunner(list(self.files), Settings.from_dict(settings.to_dict()),
                                  self.events, self.cancel_event)
        self.runner.start()

    def _cancel(self) -> None:
        if self._busy():
            self.cancel_event.set()
            self.cancel_btn.configure(state="disabled")
            self.status_label.configure(text="Cancelling after the current step...")

    def _handle_event(self, ev: tuple) -> None:
        kind = ev[0]
        if kind == "file_start":
            _, i, n, path = ev
            self.status[path] = "running"
            self._refresh_list()
            self.listbox.see(i)
            self.status_label.configure(text=f"Processing {i + 1} of {n}: {path.name}")
        elif kind == "progress":
            _, frac, msg = ev
            self.progress.set(max(0.0, min(1.0, frac)))
            self.status_label.configure(text=msg)
        elif kind == "file_done":
            self.status[ev[3]] = "done"
            self._refresh_list()
        elif kind == "file_error":
            self.status[ev[3]] = "error"
            self._refresh_list()
        elif kind == "batch_done":
            _, ok, failed, cancelled = ev
            self.start_btn.configure(state="normal")
            self.cancel_btn.configure(state="disabled")
            self.progress.set(1.0 if not cancelled else self.progress.get())
            summary = f"Finished: {ok} succeeded, {failed} failed" + (" (cancelled)" if cancelled else "")
            self.status_label.configure(text=summary)
            log.info(summary)
            if not cancelled:
                messagebox.showinfo("Batch complete", f"{summary}\n\nOutput folder:\n{self.settings.output_dir}",
                                    parent=self)

    def _poll(self) -> None:
        if self._ffmpeg_status is not None:
            text, color = self._ffmpeg_status
            self._ffmpeg_status = None
            self.ffmpeg_label.configure(text=text, text_color=color)
        try:
            while True:
                self.console.append(self.log_queue.get_nowait())
        except queue.Empty:
            pass
        try:
            while True:
                self._handle_event(self.events.get_nowait())
        except queue.Empty:
            pass
        self.after(self.POLL_MS, self._poll)

    # ------------------------------------------------------------------ misc actions
    def _preview(self) -> None:
        from ..render.frames import render_preview

        temp = Settings.from_dict(self.settings.to_dict())
        self.settings_panel.apply_to(temp)
        try:
            img = render_preview(temp)
        except Exception as exc:
            messagebox.showerror("Preview failed", str(exc), parent=self)
            return
        win = ctk.CTkToplevel(self)
        win.title("Layout preview")
        scale = min(1.0, 1100 / img.width, 800 / img.height)
        size = (int(img.width * scale), int(img.height * scale))
        photo = ctk.CTkImage(light_image=img, dark_image=img, size=size)
        label = ctk.CTkLabel(win, image=photo, text="")
        label.image = photo  # keep a reference
        label.pack(padx=10, pady=10)
        ctk.CTkLabel(win, text="Rendered with the current settings and demo lyrics/chords.",
                     text_color=("gray40", "gray60")).pack(pady=(0, 10))
        win.after(150, win.lift)

    def _open_output(self) -> None:
        path = Path(self.settings_panel.vars["output_dir"].get() or self.settings.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except OSError as exc:
            messagebox.showerror("Open folder", str(exc), parent=self)

    def _set_appearance(self, mode: str) -> None:
        ctk.set_appearance_mode(mode)
        self.settings.appearance_mode = mode
        self.after(50, self._style_listbox)

    def _check_ffmpeg(self) -> None:
        """Probe FFmpeg off the main thread; the result is applied from _poll()."""
        self._ffmpeg_status: Optional[tuple] = None

        def worker() -> None:
            try:
                version = ffmpeg_version() or "found"
                result = (f"FFmpeg: {version.replace('ffmpeg version ', '')[:28]}", ("gray40", "gray60"))
            except FFmpegNotFound as exc:
                result = ("FFmpeg: NOT FOUND", "#ec7063")
                log.error("%s", exc)
            self._ffmpeg_status = result  # picked up by _poll on the Tk thread

        threading.Thread(target=worker, daemon=True).start()

    def _on_close(self) -> None:
        if self._busy():
            if not messagebox.askyesno("Batch running", "A batch is still running. Cancel it and quit?", parent=self):
                return
            self.cancel_event.set()
        try:
            self._collect_settings().save()
        except Exception:  # pragma: no cover
            pass
        self.destroy()


def run() -> int:
    app = App()
    app.mainloop()
    return 0
