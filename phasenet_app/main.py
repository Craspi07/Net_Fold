"""PhaseNet-Sim entry point.

Launches the PyQt6 desktop application: a 5-layer computational pipeline
integrating graph network theory, biomolecular phase separation (LLPS)
prediction, spatial/kinetic modeling, and information-theoretic analysis.
"""
from __future__ import annotations

import logging
import multiprocessing
import sys

from PyQt6.QtWidgets import QApplication

from ui.main_window import DARK_STYLESHEET, MainWindow
from utils.parallel_worker import shutdown_executor


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    configure_logging()
    app = QApplication(sys.argv)
    app.setApplicationName("PhaseNet-Sim")
    app.setStyleSheet(DARK_STYLESHEET)
    app.aboutToQuit.connect(shutdown_executor)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    # Required for ProcessPoolExecutor portability under the "spawn" start
    # method (macOS/Windows default): child processes re-import this module,
    # so all pool submission must stay behind this guard.
    multiprocessing.freeze_support()
    sys.exit(main())
