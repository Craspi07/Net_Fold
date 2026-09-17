"""Matplotlib-backed plot widgets for the Kinetics/Information workbench
and the In-Silico Perturbation & Sensitivity tab.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QVBoxLayout, QWidget

DARK_BG = "#1e1e2e"
DARK_FG = "#cdd6f4"
GRID_COLOR = "#313244"
ACCENT = "#89b4fa"
ACCENT2 = "#f38ba8"
ACCENT3 = "#a6e3a1"


class _BaseCanvas(QWidget):
    def __init__(self, title: str, xlabel: str, ylabel: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 3.5), facecolor=DARK_BG)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        self._style_axes(title, xlabel, ylabel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self.canvas)

    def _style_axes(self, title: str, xlabel: str, ylabel: str) -> None:
        self.ax.set_facecolor(DARK_BG)
        self.ax.set_title(title, color=DARK_FG, fontsize=10)
        self.ax.set_xlabel(xlabel, color=DARK_FG, fontsize=9)
        self.ax.set_ylabel(ylabel, color=DARK_FG, fontsize=9)
        self.ax.tick_params(colors=DARK_FG, labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color(GRID_COLOR)
        self.ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.6)

    def clear(self, title: str, xlabel: str, ylabel: str) -> None:
        self.ax.clear()
        self._style_axes(title, xlabel, ylabel)

    def redraw(self) -> None:
        self.figure.tight_layout()
        self.canvas.draw_idle()


class KineticsPlotWidget(_BaseCanvas):
    """Panel A: [Protein]_t trajectories across soluble vs. condensate nodes."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("Kinetic Trajectories", "Time", "Concentration", parent)

    def update_trajectories(self, times: np.ndarray, trajectories: Dict[str, np.ndarray],
                             condensate_fraction: Dict[str, float], highlight: Optional[Sequence[str]] = None):
        self.clear("Kinetic Trajectories", "Time", "Concentration")
        highlight = set(highlight) if highlight else None
        colors = [ACCENT, ACCENT2, ACCENT3, "#fab387", "#cba6f7", "#94e2d5"]

        plotted = 0
        for i, (node, series) in enumerate(trajectories.items()):
            if highlight is not None and node not in highlight:
                continue
            frac = condensate_fraction.get(node, 0.0)
            style = "--" if frac > 0.5 else "-"
            self.ax.plot(times, series, style, color=colors[plotted % len(colors)],
                         label=f"{node} (cond={frac:.2f})", linewidth=1.6)
            plotted += 1
            if plotted >= 8:
                break
        if plotted:
            self.ax.legend(fontsize=7, facecolor=DARK_BG, labelcolor=DARK_FG, framealpha=0.6)
        self.redraw()


class MutualInfoPlotWidget(_BaseCanvas):
    """Panel B: Input vs. output mutual information over varying noise levels."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("Mutual Information vs. Noise", "Noise Amplitude", "I(X;Y) [bits]", parent)

    def update_curve(self, noise_levels: np.ndarray, mi_values: np.ndarray) -> None:
        self.clear("Mutual Information vs. Noise", "Noise Amplitude", "I(X;Y) [bits]")
        self.ax.plot(noise_levels, mi_values, "o-", color=ACCENT, linewidth=1.8, markersize=4)
        self.redraw()


class NoiseComparisonPlotWidget(_BaseCanvas):
    """Panel C: Noise-buffering comparison, with vs. without LLPS kinetics."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("Noise Buffering: LLPS vs. No-LLPS", "Metric", "Value", parent)

    def update_comparison(self, metrics_with: dict, metrics_without: dict) -> None:
        self.clear("Noise Buffering: LLPS vs. No-LLPS", "Metric", "Value")
        labels = ["Fano Factor", "SNR", "I(X;Y)"]
        with_vals = [metrics_with["fano_factor"], min(metrics_with["snr"], 50), metrics_with["mutual_information"]]
        without_vals = [metrics_without["fano_factor"], min(metrics_without["snr"], 50),
                         metrics_without["mutual_information"]]

        x = np.arange(len(labels))
        width = 0.35
        self.ax.bar(x - width / 2, with_vals, width, label="With LLPS", color=ACCENT3)
        self.ax.bar(x + width / 2, without_vals, width, label="Without LLPS", color=ACCENT2)
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(labels, color=DARK_FG, fontsize=8)
        self.ax.legend(fontsize=7, facecolor=DARK_BG, labelcolor=DARK_FG, framealpha=0.6)
        self.redraw()


class HeatmapWidget(_BaseCanvas):
    """Layer 5: information-loss heatmaps (knockout scan / sensitivity matrix)."""

    def __init__(self, title: str = "Sensitivity Heatmap", parent: Optional[QWidget] = None):
        super().__init__(title, "", "", parent)
        self._title = title
        self._colorbar = None

    def update_bar_ranking(self, labels: List[str], values: List[float], ylabel: str = "I(X;Y) drop") -> None:
        """Used for 1D knockout / valency-mutagenesis rankings."""
        self.clear(self._title, "Node", ylabel)
        order = np.argsort(values)[::-1]
        labels_sorted = [labels[i] for i in order][:25]
        values_sorted = [values[i] for i in order][:25]
        colors = ["#f38ba8" if v > 0 else "#94e2d5" for v in values_sorted]
        self.ax.bar(range(len(labels_sorted)), values_sorted, color=colors)
        self.ax.set_xticks(range(len(labels_sorted)))
        self.ax.set_xticklabels(labels_sorted, rotation=75, ha="right", fontsize=6, color=DARK_FG)
        self.redraw()

    def update_scatter_heat(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                              xlabel: str = "K_part threshold", ylabel: str = "gamma") -> None:
        """Used for the LHS sensitivity matrix over (K_part threshold, gamma)."""
        self.clear(self._title, xlabel, ylabel)
        if self._colorbar is not None:
            try:
                self._colorbar.remove()
            except (ValueError, AttributeError):
                pass
            self._colorbar = None
        sc = self.ax.scatter(x, y, c=z, cmap="magma", s=80, edgecolors="white", linewidths=0.3)
        self._colorbar = self.figure.colorbar(sc, ax=self.ax)
        self._colorbar.set_label("I(X;Y) [bits]", color=DARK_FG, fontsize=8)
        self._colorbar.ax.yaxis.set_tick_params(color=DARK_FG)
        for label in self._colorbar.ax.get_yticklabels():
            label.set_color(DARK_FG)
        self.redraw()
