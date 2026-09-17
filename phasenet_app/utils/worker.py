"""QThread wrappers so network I/O and heavy math never block the GUI thread."""
from __future__ import annotations

import asyncio
import traceback
from typing import Any, Callable, Optional

from PyQt6.QtCore import QObject, QThread, pyqtSignal


class WorkerSignals(QObject):
    """Signals available for any background worker."""

    started = pyqtSignal()
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)


class FunctionWorker(QThread):
    """Runs a plain synchronous callable on a background thread.

    The callable may accept a ``progress_cb(percent, message)`` keyword
    argument if it declares one; PhaseNet-Sim's heavy layer functions do.
    """

    def __init__(self, func: Callable, *args, use_progress_cb: bool = False, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.use_progress_cb = use_progress_cb
        self.signals = WorkerSignals()

    def run(self) -> None:
        self.signals.started.emit()
        try:
            if self.use_progress_cb:
                self.kwargs["progress_cb"] = lambda pct, msg="": self.signals.progress.emit(int(pct), msg)
            result = self.func(*self.args, **self.kwargs)
            self.signals.finished.emit(result)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI, not swallowed
            self.signals.error.emit(f"{exc}\n{traceback.format_exc()}")


class AsyncWorker(QThread):
    """Runs an ``async def`` coroutine function on a private event loop.

    Used for aiohttp-based UniProt / STRING-DB fetching so the GUI event
    loop is never touched.
    """

    def __init__(self, coro_func: Callable[..., Any], *args, **kwargs):
        super().__init__()
        self.coro_func = coro_func
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def progress_cb(self, pct: int, msg: str = "") -> None:
        self.signals.progress.emit(int(pct), msg)

    def run(self) -> None:
        self.signals.started.emit()
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self.kwargs.setdefault("progress_cb", self.progress_cb)
            result = self._loop.run_until_complete(self.coro_func(*self.args, **self.kwargs))
            self.signals.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.signals.error.emit(f"{exc}\n{traceback.format_exc()}")
        finally:
            self._loop.close()
