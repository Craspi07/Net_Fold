"""ProcessPool / QThread hybrid wrappers.

QThread alone shares the interpreter's GIL with the GUI thread, so it is
fine for I/O (aiohttp fetches) but cannot actually parallelize CPU-bound
work such as the stochastic kinetic simulations and mutual-information
sweeps in Layers 3-5 - those need real OS processes. The pattern used
throughout this app is:

- A single persistent ``ProcessPoolExecutor`` (``get_executor``) does the
  actual CPU-bound work, off the GIL, in separate processes.
- Core-layer "orchestrator" functions (e.g. ``layer5_mutagenesis.deletion_scan``)
  call ``run_parallel`` to fan a batch of independent tasks out across that
  executor and gather results as they complete.
- Those orchestrator functions are themselves invoked from the GUI via
  ``FunctionWorker``, a plain QThread wrapper, so the blocking wait on the
  process-pool futures happens on a background thread and never touches
  the Qt event loop.

Async network I/O (UniProt / STRING-DB) uses ``AsyncWorker`` instead, which
runs an ``asyncio`` event loop on its own QThread.
"""
from __future__ import annotations

import asyncio
import logging
import traceback
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from PyQt6.QtCore import QObject, QThread, pyqtSignal

log = logging.getLogger("phasenet.parallel_worker")


class WorkerSignals(QObject):
    """Signals available for any background worker."""

    started = pyqtSignal()
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)


class FunctionWorker(QThread):
    """Runs a plain synchronous callable on a background thread.

    The callable may accept a ``progress_cb(percent, message)`` keyword
    argument if it declares one; PhaseNet-Sim's layer functions do. This
    is used both for lightweight work (Layer 1/2, file I/O) and for
    orchestrator functions that internally fan out to the process pool via
    ``run_parallel`` - in the latter case this QThread just supervises the
    blocking wait so the GUI thread stays responsive.
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


# --------------------------------------------------------------- ProcessPool
_executor: Optional[ProcessPoolExecutor] = None


def get_executor(max_workers: Optional[int] = None) -> ProcessPoolExecutor:
    """Lazily create the single persistent ProcessPoolExecutor for this app."""
    global _executor
    if _executor is None:
        _executor = ProcessPoolExecutor(max_workers=max_workers)
        log.info("Started ProcessPoolExecutor (max_workers=%s)", max_workers or "auto")
    return _executor


def shutdown_executor() -> None:
    """Cleanly tear down the process pool, e.g. on application quit."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)
        _executor = None
        log.info("ProcessPoolExecutor shut down.")


ProgressCB = Optional[Callable[[int, str], None]]


def run_parallel(func: Callable, tasks: Sequence[Tuple[tuple, dict]],
                  progress_cb: ProgressCB = None, max_workers: Optional[int] = None,
                  label: str = "task") -> List[Any]:
    """Fan ``tasks`` out across the persistent process pool and gather results.

    ``tasks`` is a sequence of ``(args, kwargs)`` pairs, each an independent
    call to the top-level, picklable ``func``. Results are returned in the
    same order as ``tasks`` regardless of completion order; ``progress_cb``
    is invoked once per completed task (real-time, but order-independent).
    """
    executor = get_executor(max_workers=max_workers)
    futures: Dict[Future, int] = {}
    for i, (args, kwargs) in enumerate(tasks):
        future = executor.submit(func, *args, **kwargs)
        futures[future] = i

    results: List[Any] = [None] * len(tasks)
    errors: List[str] = []
    done = 0
    total = max(1, len(tasks))
    for future in as_completed(futures):
        idx = futures[future]
        try:
            results[idx] = future.result()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{label} #{idx}: {exc}")
            results[idx] = None
        done += 1
        if progress_cb:
            progress_cb(int(100 * done / total), f"{label} {done}/{total}")

    if errors:
        log.warning("run_parallel: %d/%d %s(s) failed: %s", len(errors), total, label, "; ".join(errors[:5]))

    return results
