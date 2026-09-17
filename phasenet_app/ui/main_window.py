"""PhaseNet-Sim main window: dark-themed multi-panel PyQt6 workspace tying
together the 5-layer computational pipeline via background QThread workers.
"""
from __future__ import annotations

import csv
import logging
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
from PyQt6.QtCore import QAbstractTableModel, QModelIndex, QObject, Qt, QSortFilterProxyModel, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QSpinBox, QSplitter, QTableView, QTabWidget, QTextEdit, QVBoxLayout,
    QWidget,
)

from core import layer1_topology, layer2_llps, layer4_infotheory, layer5_mutagenesis, pathways
from core.network_builder import NetworkBuilder
from ui.graph_view import GraphView
from ui.plot_widgets import HeatmapWidget, KineticsPlotWidget, MutualInfoPlotWidget, NoiseComparisonPlotWidget
from utils.parallel_worker import AsyncWorker, FunctionWorker

log = logging.getLogger("phasenet.main_window")

CUSTOM_PATHWAY_LABEL = "Custom (from input above)"

NODE_COLUMNS = [
    ("id", "ID"), ("label", "Label"), ("role", "Role"),
    ("degree_centrality", "Degree Cent."), ("betweenness_centrality", "Betweenness"),
    ("pagerank", "PageRank"), ("eigenvector_centrality", "Eigenvector"),
    ("vulnerability", "Vulnerability"), ("s_llps", "S_LLPS"), ("s_llps_base", "S_LLPS (base)"),
    ("disorder_score", "Disorder"), ("kappa", "Kappa"), ("valency", "Valency"),
    ("ps_critical_hub", "PS Critical Hub"),
]


class QtLogSignal(QObject):
    message = pyqtSignal(str, str)  # level, text


class QtLogHandler(logging.Handler):
    """Bridges Python logging records into the Qt log console thread-safely."""

    def __init__(self):
        super().__init__()
        self.emitter = QtLogSignal()
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self.emitter.message.emit(record.levelname, msg)


class NodeTableModel(QAbstractTableModel):
    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._rows: List[Dict[str, Any]] = []

    def set_rows(self, rows: List[Dict[str, Any]]) -> None:
        self.beginResetModel()
        self._rows = rows
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(NODE_COLUMNS)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role not in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            return None
        key, _ = NODE_COLUMNS[index.column()]
        value = self._rows[index.row()].get(key, "")
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return NODE_COLUMNS[section][1]
        return str(section + 1)


DARK_STYLESHEET = """
QWidget { background-color: #1e1e2e; color: #cdd6f4; font-size: 12px; }
QMainWindow { background-color: #1e1e2e; }
QGroupBox { border: 1px solid #313244; border-radius: 6px; margin-top: 10px; padding-top: 8px; font-weight: bold; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #89b4fa; }
QPushButton { background-color: #313244; border: 1px solid #45475a; border-radius: 4px; padding: 6px 10px; }
QPushButton:hover { background-color: #45475a; }
QPushButton:pressed { background-color: #585b70; }
QPushButton:disabled { color: #6c7086; }
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #181825; border: 1px solid #313244; border-radius: 4px; padding: 3px; color: #cdd6f4;
}
QTableView { background-color: #181825; alternate-background-color: #1e1e2e; gridline-color: #313244; }
QHeaderView::section { background-color: #313244; padding: 4px; border: none; }
QTabWidget::pane { border: 1px solid #313244; }
QTabBar::tab { background: #181825; padding: 8px 16px; border: 1px solid #313244; }
QTabBar::tab:selected { background: #313244; color: #89b4fa; }
QProgressBar { border: 1px solid #313244; border-radius: 4px; text-align: center; background-color: #181825; }
QProgressBar::chunk { background-color: #89b4fa; }
QScrollBar:vertical { background: #181825; width: 10px; }
QScrollBar::handle:vertical { background: #45475a; border-radius: 5px; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhaseNet-Sim — LLPS Network Simulation Suite")
        self.resize(1500, 950)

        self.graph: Optional[nx.Graph] = None
        self.last_sim_comparison: Optional[dict] = None
        self._fetch_worker: Optional[AsyncWorker] = None
        self._pipeline_worker: Optional[FunctionWorker] = None
        self._pending_pathway: Optional[pathways.PathwayTemplate] = None

        self._build_ui()
        self._wire_logging()
        log.info("PhaseNet-Sim initialized.")

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(main_splitter, stretch=1)

        main_splitter.addWidget(self._build_sidebar())

        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.addWidget(self._build_tabs())
        right_splitter.addWidget(self._build_log_panel())
        right_splitter.setStretchFactor(0, 4)
        right_splitter.setStretchFactor(1, 1)
        main_splitter.addWidget(right_splitter)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setFixedWidth(340)
        layout = QVBoxLayout(sidebar)

        input_group = QGroupBox("Protein Input")
        input_layout = QVBoxLayout(input_group)

        input_layout.addWidget(QLabel("Pre-curated pathway:"))
        self.pathway_combo = QComboBox()
        self.pathway_combo.addItem(CUSTOM_PATHWAY_LABEL)
        self.pathway_combo.addItems(pathways.get_pathway_names())
        self.pathway_combo.currentTextChanged.connect(self._on_pathway_selected)
        input_layout.addWidget(self.pathway_combo)

        self.id_input = QTextEdit()
        self.id_input.setPlaceholderText("UniProt IDs or gene symbols, one per line\n(e.g. FUS, TDP-43, EWSR1)")
        self.id_input.setFixedHeight(90)
        input_layout.addWidget(self.id_input)
        load_btn = QPushButton("Load from File (.csv/.txt)")
        load_btn.clicked.connect(self._load_file)
        input_layout.addWidget(load_btn)
        layout.addWidget(input_group)

        discovery_group = QGroupBox("Interaction Discovery")
        discovery_layout = QVBoxLayout(discovery_group)

        discovery_layout.addWidget(QLabel("Min. STRING confidence score:"))
        self.score_spin = QDoubleSpinBox()
        self.score_spin.setRange(0.0, 1.0)
        self.score_spin.setSingleStep(0.05)
        self.score_spin.setValue(0.4)
        discovery_layout.addWidget(self.score_spin)

        discovery_layout.addWidget(QLabel("Organism:"))
        self.organism_combo = QComboBox()
        self.organism_combo.addItem("Human (9606)", 9606)
        self.organism_combo.addItem("Mouse (10090)", 10090)
        self.organism_combo.addItem("Yeast (4932)", 4932)
        discovery_layout.addWidget(self.organism_combo)

        discovery_layout.addWidget(QLabel("Max expansion depth (hops):"))
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(1, 2)
        self.depth_spin.setValue(1)
        discovery_layout.addWidget(self.depth_spin)

        layout.addWidget(discovery_group)

        context_group = QGroupBox("LLPS Context Modifiers")
        context_layout = QVBoxLayout(context_group)

        self.ptm_checkbox_label = QLabel("PTM (phosphorylation) weight:")
        context_layout.addWidget(self.ptm_checkbox_label)
        self.ptm_weight_spin = QDoubleSpinBox()
        self.ptm_weight_spin.setRange(0.0, 1.0)
        self.ptm_weight_spin.setSingleStep(0.05)
        self.ptm_weight_spin.setValue(0.0)
        self.ptm_weight_spin.setToolTip(
            "Simulates phosphorylation-driven charge patterning that context-modulates S_LLPS "
            "without changing the underlying sequence."
        )
        context_layout.addWidget(self.ptm_weight_spin)

        context_layout.addWidget(QLabel("Multivalency (binding partner) weight:"))
        self.multivalency_weight_spin = QDoubleSpinBox()
        self.multivalency_weight_spin.setRange(0.0, 1.0)
        self.multivalency_weight_spin.setSingleStep(0.05)
        self.multivalency_weight_spin.setValue(0.0)
        self.multivalency_weight_spin.setToolTip(
            "Simulates multivalent scaffold/binding-partner effects that raise effective sticker "
            "valency beyond what the sequence alone predicts."
        )
        context_layout.addWidget(self.multivalency_weight_spin)

        layout.addWidget(context_group)

        sim_group = QGroupBox("Simulation Hyperparameters")
        sim_layout = QVBoxLayout(sim_group)

        sim_layout.addWidget(QLabel("Gamma (amplification/sequestration):"))
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.1, 10.0)
        self.gamma_spin.setValue(2.0)
        sim_layout.addWidget(self.gamma_spin)

        sim_layout.addWidget(QLabel("K_part threshold (S_LLPS):"))
        self.kpart_spin = QDoubleSpinBox()
        self.kpart_spin.setRange(0.0, 1.0)
        self.kpart_spin.setSingleStep(0.05)
        self.kpart_spin.setValue(0.5)
        sim_layout.addWidget(self.kpart_spin)

        sim_layout.addWidget(QLabel("Simulation time T:"))
        self.time_spin = QDoubleSpinBox()
        self.time_spin.setRange(1.0, 500.0)
        self.time_spin.setValue(50.0)
        sim_layout.addWidget(self.time_spin)

        sim_layout.addWidget(QLabel("Noise amplitude:"))
        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(0.0, 5.0)
        self.noise_spin.setSingleStep(0.05)
        self.noise_spin.setValue(0.2)
        sim_layout.addWidget(self.noise_spin)

        sim_layout.addWidget(QLabel("Input node:"))
        self.input_node_combo = QComboBox()
        sim_layout.addWidget(self.input_node_combo)

        sim_layout.addWidget(QLabel("Output node:"))
        self.output_node_combo = QComboBox()
        sim_layout.addWidget(self.output_node_combo)

        layout.addWidget(sim_group)

        self.fetch_btn = QPushButton("Fetch && Build Network")
        self.fetch_btn.clicked.connect(self._on_fetch_network)
        layout.addWidget(self.fetch_btn)

        self.run_btn = QPushButton("Run Pipeline (Layers 1-5)")
        self.run_btn.clicked.connect(self._on_run_pipeline)
        self.run_btn.setEnabled(False)
        layout.addWidget(self.run_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        layout.addStretch(1)
        return sidebar

    def _build_tabs(self) -> QWidget:
        self.tabs = QTabWidget()

        self.graph_view = GraphView()
        self.graph_view.node_clicked.connect(self._on_node_clicked)
        self.tabs.addTab(self.graph_view, "Network Topology")

        self.tabs.addTab(self._build_table_tab(), "Criticality & LLPS Matrix")
        self.tabs.addTab(self._build_kinetics_tab(), "Dynamic Kinetics & Information")
        self.tabs.addTab(self._build_perturbation_tab(), "In-Silico Perturbation")

        return self.tabs

    def _build_table_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        toolbar = QHBoxLayout()
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search / filter nodes...")
        self.search_box.textChanged.connect(self._on_search_changed)
        toolbar.addWidget(self.search_box)
        export_btn = QPushButton("Export to CSV")
        export_btn.clicked.connect(self._export_csv)
        toolbar.addWidget(export_btn)
        layout.addLayout(toolbar)

        self.node_table_model = NodeTableModel()
        self.proxy_model = QSortFilterProxyModel()
        self.proxy_model.setSourceModel(self.node_table_model)
        self.proxy_model.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.proxy_model.setFilterKeyColumn(-1)

        self.table_view = QTableView()
        self.table_view.setModel(self.proxy_model)
        self.table_view.setSortingEnabled(True)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_view.setAlternatingRowColors(True)
        layout.addWidget(self.table_view)
        return widget

    def _build_kinetics_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self.kinetics_plot = KineticsPlotWidget()
        splitter.addWidget(self.kinetics_plot)

        bottom_split = QSplitter(Qt.Orientation.Horizontal)
        self.mi_plot = MutualInfoPlotWidget()
        self.noise_plot = NoiseComparisonPlotWidget()
        bottom_split.addWidget(self.mi_plot)
        bottom_split.addWidget(self.noise_plot)
        splitter.addWidget(bottom_split)

        layout.addWidget(splitter)
        return widget

    def _build_perturbation_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        toolbar = QHBoxLayout()
        knockout_btn = QPushButton("Run Deletion Scan")
        knockout_btn.clicked.connect(self._run_deletion_scan)
        toolbar.addWidget(knockout_btn)
        valency_btn = QPushButton("Run Valency Mutagenesis")
        valency_btn.clicked.connect(self._run_valency_mutagenesis)
        toolbar.addWidget(valency_btn)
        sensitivity_btn = QPushButton("Run Sensitivity Matrix (LHS)")
        sensitivity_btn.clicked.connect(self._run_sensitivity_matrix)
        toolbar.addWidget(sensitivity_btn)
        layout.addLayout(toolbar)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.knockout_heatmap = HeatmapWidget("Deletion / Valency Scan")
        self.sensitivity_heatmap = HeatmapWidget("K_part x Gamma Sensitivity")
        split.addWidget(self.knockout_heatmap)
        split.addWidget(self.sensitivity_heatmap)
        layout.addWidget(split)
        return widget

    def _build_log_panel(self) -> QWidget:
        widget = QGroupBox("Log Console")
        layout = QVBoxLayout(widget)
        self.log_console = QPlainTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setMaximumBlockCount(2000)
        layout.addWidget(self.log_console)
        return widget

    def _wire_logging(self) -> None:
        handler = QtLogHandler()
        handler.emitter.message.connect(self._append_log)
        root_logger = logging.getLogger("phasenet")
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(handler)

    def _append_log(self, level: str, text: str) -> None:
        color = {"ERROR": "#f38ba8", "WARNING": "#f9e2af", "INFO": "#cdd6f4", "DEBUG": "#6c7086"}.get(level, "#cdd6f4")
        self.log_console.appendHtml(f'<span style="color:{color}">{text}</span>')

    # ------------------------------------------------------------ Actions
    def _on_pathway_selected(self, name: str) -> None:
        if name == CUSTOM_PATHWAY_LABEL:
            self._pending_pathway = None
            return
        template = pathways.get_pathway(name)
        self._pending_pathway = template

        self.id_input.setPlainText("\n".join(template.seed_ids))
        self.score_spin.setValue(template.min_score)
        self.depth_spin.setValue(template.expand_depth)
        self.gamma_spin.setValue(template.gamma)
        self.kpart_spin.setValue(template.kpart_threshold)
        self.ptm_weight_spin.setValue(template.ptm_weight)
        self.multivalency_weight_spin.setValue(template.multivalency_weight)
        organism_idx = self.organism_combo.findData(template.organism)
        if organism_idx >= 0:
            self.organism_combo.setCurrentIndex(organism_idx)

        log.info("Loaded pathway template '%s': %s", template.name, template.description)
        if template.notes:
            log.info("Pathway note: %s", template.notes)

    def _load_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Protein IDs", "", "Text/CSV Files (*.csv *.txt)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            ids = [tok.strip() for line in content.splitlines() for tok in line.split(",") if tok.strip()]
            self.id_input.setPlainText("\n".join(ids))
            log.info("Loaded %d identifiers from %s", len(ids), path)
        except OSError as exc:
            QMessageBox.warning(self, "File Load Error", str(exc))
            log.error("Failed to load file %s: %s", path, exc)

    def _collect_seed_ids(self) -> List[str]:
        text = self.id_input.toPlainText()
        return [tok.strip() for line in text.splitlines() for tok in line.split(",") if tok.strip()]

    def _on_fetch_network(self) -> None:
        seed_ids = self._collect_seed_ids()
        if not seed_ids:
            QMessageBox.warning(self, "No Input", "Enter at least one UniProt ID or gene symbol.")
            return

        self.fetch_btn.setEnabled(False)
        self.run_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        log.info("Fetching network for %d seed proteins...", len(seed_ids))

        builder = NetworkBuilder(
            species=self.organism_combo.currentData(),
            min_score=self.score_spin.value(),
            expand_depth=self.depth_spin.value(),
        )
        self._fetch_worker = AsyncWorker(builder.build, seed_ids)
        self._fetch_worker.signals.progress.connect(self._on_progress)
        self._fetch_worker.signals.finished.connect(self._on_network_built)
        self._fetch_worker.signals.error.connect(self._on_worker_error)
        self._fetch_worker.start()

    def _on_progress(self, pct: int, msg: str) -> None:
        self.progress_bar.setValue(pct)
        if msg:
            log.info(msg)

    def _on_worker_error(self, message: str) -> None:
        self.fetch_btn.setEnabled(True)
        self.run_btn.setEnabled(self.graph is not None)
        log.error("Background task failed: %s", message)
        QMessageBox.critical(self, "Task Failed", message.split("\n")[0])

    def _on_network_built(self, graph: nx.Graph) -> None:
        self.graph = graph
        self.fetch_btn.setEnabled(True)
        self.run_btn.setEnabled(True)
        log.info("Network built: %d nodes, %d edges.", graph.number_of_nodes(), graph.number_of_edges())

        self.input_node_combo.clear()
        self.output_node_combo.clear()
        node_ids = list(graph.nodes())
        self.input_node_combo.addItems(node_ids)
        self.output_node_combo.addItems(node_ids)
        if len(node_ids) > 1:
            self.output_node_combo.setCurrentIndex(1)

        if self._pending_pathway is not None:
            template = self._pending_pathway
            in_idx = self.input_node_combo.findText(template.suggested_input)
            out_idx = self.output_node_combo.findText(template.suggested_output)
            if in_idx >= 0:
                self.input_node_combo.setCurrentIndex(in_idx)
            if out_idx >= 0:
                self.output_node_combo.setCurrentIndex(out_idx)

        self.graph_view.update_graph(graph)
        self._refresh_table()

    def _on_run_pipeline(self) -> None:
        if self.graph is None:
            QMessageBox.warning(self, "No Network", "Fetch & build a network first.")
            return

        self.run_btn.setEnabled(False)
        self.fetch_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        log.info("Running Layers 1-5 pipeline...")

        self._pipeline_worker = FunctionWorker(self._run_pipeline_sync, use_progress_cb=True)
        self._pipeline_worker.signals.progress.connect(self._on_progress)
        self._pipeline_worker.signals.finished.connect(self._on_pipeline_done)
        self._pipeline_worker.signals.error.connect(self._on_worker_error)
        self._pipeline_worker.start()

    def _run_pipeline_sync(self, progress_cb=None) -> dict:
        """Runs on the FunctionWorker's background thread."""
        G = self.graph
        assert G is not None

        def scaled(base: int, span: int):
            return lambda pct, msg="": progress_cb(base + int(pct * span / 100), msg) if progress_cb else None

        layer1_topology.compute_topology(G, progress_cb=scaled(0, 30))
        layer2_llps.compute_llps(
            G, ptm_weight=self.ptm_weight_spin.value(), multivalency_weight=self.multivalency_weight_spin.value(),
            progress_cb=scaled(30, 15),
        )
        critical_hubs = layer2_llps.identify_critical_hubs(G)

        input_nodes = [self.input_node_combo.currentText()] if self.input_node_combo.currentText() else [list(G.nodes())[0]]
        output_nodes = [self.output_node_combo.currentText()] if self.output_node_combo.currentText() else [list(G.nodes())[-1]]

        kinetics_kwargs = dict(
            gamma=self.gamma_spin.value(), kpart_threshold=self.kpart_spin.value(),
            sim_time=self.time_spin.value(), noise_amplitude=self.noise_spin.value(),
        )
        comparison = layer4_infotheory.compare_llps_effect(
            G, input_nodes, output_nodes, signal_kind="step", progress_cb=scaled(45, 25), **kinetics_kwargs,
        )
        noise_levels = np.linspace(0.0, 1.5, 10)
        mi_curve = layer4_infotheory.mi_vs_noise_curve(
            G, input_nodes, output_nodes, noise_levels, progress_cb=scaled(70, 30),
            gamma=kinetics_kwargs["gamma"], kpart_threshold=kinetics_kwargs["kpart_threshold"],
            sim_time=kinetics_kwargs["sim_time"],
        )

        if progress_cb:
            progress_cb(100, "Pipeline complete.")

        return {
            "critical_hubs": critical_hubs,
            "input_nodes": input_nodes,
            "output_nodes": output_nodes,
            "comparison": comparison,
            "mi_curve": mi_curve,
        }

    def _on_pipeline_done(self, results: dict) -> None:
        self.run_btn.setEnabled(True)
        self.fetch_btn.setEnabled(True)
        log.info("Pipeline finished. %d PS-critical hubs identified.", len(results["critical_hubs"]))

        self.graph_view.update_graph(self.graph)
        self._refresh_table()

        comparison = results["comparison"]
        self.last_sim_comparison = comparison
        result_on = comparison["result_with_llps"]
        self.kinetics_plot.update_trajectories(
            result_on.times, result_on.trajectories, result_on.condensate_fraction,
            highlight=results["input_nodes"] + results["output_nodes"],
        )
        self.mi_plot.update_curve(results["mi_curve"]["noise_levels"], results["mi_curve"]["mutual_information"])
        self.noise_plot.update_comparison(comparison["with_llps"], comparison["without_llps"])

    def _on_node_clicked(self, node_id: str) -> None:
        if self.graph is None or node_id not in self.graph.nodes:
            return
        data = self.graph.nodes[node_id]
        log.info(
            "Node %s | role=%s | S_LLPS=%.3f | betweenness=%.4f | disorder=%.3f",
            node_id, data.get("role", "?"), data.get("s_llps", 0.0),
            data.get("betweenness_centrality", 0.0), data.get("disorder_score", 0.0),
        )

    def _refresh_table(self) -> None:
        if self.graph is None:
            return
        rows = []
        for n, data in self.graph.nodes(data=True):
            row = {"id": n, "label": data.get("label", n)}
            for key, _ in NODE_COLUMNS[2:]:
                row[key] = data.get(key, 0.0 if key != "role" else "")
            rows.append(row)
        self.node_table_model.set_rows(rows)

    def _on_search_changed(self, text: str) -> None:
        self.proxy_model.setFilterFixedString(text)

    def _export_csv(self) -> None:
        if self.graph is None:
            QMessageBox.warning(self, "No Data", "Nothing to export yet.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "phasenet_nodes.csv", "CSV Files (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([label for _, label in NODE_COLUMNS])
                for row in range(self.node_table_model.rowCount()):
                    writer.writerow([
                        self.node_table_model.data(self.node_table_model.index(row, col))
                        for col in range(self.node_table_model.columnCount())
                    ])
            log.info("Exported node table to %s", path)
        except OSError as exc:
            QMessageBox.warning(self, "Export Error", str(exc))

    # --------------------------------------------------------- Layer 5 UI
    def _pipeline_prereq_nodes(self):
        input_node = self.input_node_combo.currentText()
        output_node = self.output_node_combo.currentText()
        if self.graph is None or not input_node or not output_node:
            QMessageBox.warning(self, "Not Ready", "Build a network and run the pipeline first.")
            return None
        return [input_node], [output_node]

    def _run_deletion_scan(self) -> None:
        nodes = self._pipeline_prereq_nodes()
        if nodes is None:
            return
        input_nodes, output_nodes = nodes
        log.info("Running in-silico deletion scan...")
        worker = FunctionWorker(
            layer5_mutagenesis.deletion_scan, self.graph, input_nodes, output_nodes,
            gamma=self.gamma_spin.value(), kpart_threshold=self.kpart_spin.value(),
            sim_time=self.time_spin.value(), use_progress_cb=True,
        )
        worker.signals.progress.connect(self._on_progress)
        worker.signals.finished.connect(self._on_deletion_scan_done)
        worker.signals.error.connect(self._on_worker_error)
        self._layer5_worker = worker
        worker.start()

    def _on_deletion_scan_done(self, drop_map: dict) -> None:
        log.info("Deletion scan complete for %d nodes.", len(drop_map))
        labels = list(drop_map.keys())
        values = list(drop_map.values())
        self.knockout_heatmap.update_bar_ranking(labels, values, ylabel="I(X;Y) drop on knockout")

    def _run_valency_mutagenesis(self) -> None:
        nodes = self._pipeline_prereq_nodes()
        if nodes is None:
            return
        input_nodes, output_nodes = nodes
        log.info("Running valency mutagenesis scan...")
        worker = FunctionWorker(
            layer5_mutagenesis.valency_mutagenesis, self.graph, input_nodes, output_nodes,
            gamma=self.gamma_spin.value(), kpart_threshold=self.kpart_spin.value(),
            sim_time=self.time_spin.value(), use_progress_cb=True,
        )
        worker.signals.progress.connect(self._on_progress)
        worker.signals.finished.connect(self._on_valency_scan_done)
        worker.signals.error.connect(self._on_worker_error)
        self._layer5_worker = worker
        worker.start()

    def _on_valency_scan_done(self, drop_map: dict) -> None:
        log.info("Valency mutagenesis scan complete for %d nodes.", len(drop_map))
        labels = list(drop_map.keys())
        values = list(drop_map.values())
        self.knockout_heatmap.update_bar_ranking(labels, values, ylabel="I(X;Y) drop (Δv mutation)")

    def _run_sensitivity_matrix(self) -> None:
        nodes = self._pipeline_prereq_nodes()
        if nodes is None:
            return
        input_nodes, output_nodes = nodes
        log.info("Running Latin Hypercube sensitivity sweep...")
        worker = FunctionWorker(
            layer5_mutagenesis.sensitivity_matrix, self.graph, input_nodes, output_nodes,
            sim_time=min(30.0, self.time_spin.value()), use_progress_cb=True,
        )
        worker.signals.progress.connect(self._on_progress)
        worker.signals.finished.connect(self._on_sensitivity_done)
        worker.signals.error.connect(self._on_worker_error)
        self._layer5_worker = worker
        worker.start()

    def _on_sensitivity_done(self, result: dict) -> None:
        log.info("Sensitivity sweep complete (%d samples).", len(result["mutual_information"]))
        self.sensitivity_heatmap.update_scatter_heat(
            result["kpart_thresholds"], result["gammas"], result["mutual_information"],
        )
