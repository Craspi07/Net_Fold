"""Tab 1: Interactive Network Topology viewer (PyVis inside QWebEngineView).

Node size encodes betweenness centrality, node color encodes S_LLPS
propensity (gradient), and edge width encodes STRING confidence. Node
clicks are relayed back into Qt via a QWebChannel bridge.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

import networkx as nx
from PyQt6.QtCore import QObject, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QVBoxLayout, QWidget

log = logging.getLogger("phasenet.graph_view")

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:  # pragma: no cover
    PYVIS_AVAILABLE = False


def _llps_color(score: float) -> str:
    """Blue (low S_LLPS) -> red (high S_LLPS) gradient."""
    score = max(0.0, min(1.0, score))
    r = int(255 * score)
    b = int(255 * (1 - score))
    g = int(60 * (1 - abs(score - 0.5) * 2))
    return f"#{r:02x}{g:02x}{b:02x}"


class GraphBridge(QObject):
    """JS <-> Python bridge exposing node click events."""

    nodeClicked = pyqtSignal(str)

    @pyqtSlot(str)
    def onNodeClick(self, node_id: str) -> None:
        self.nodeClicked.emit(node_id)


class GraphView(QWidget):
    node_clicked = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="phasenet_graph_"))
        self.web_view = QWebEngineView(self)
        self.bridge = GraphBridge()
        self.bridge.nodeClicked.connect(self.node_clicked.emit)

        self.channel = QWebChannel(self.web_view.page())
        self.channel.registerObject("bridge", self.bridge)
        self.web_view.page().setWebChannel(self.channel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.web_view)

        self._render_placeholder()

    def _render_placeholder(self) -> None:
        html = """
        <html><body style="background:#1e1e2e;color:#cdd6f4;font-family:sans-serif;
        display:flex;align-items:center;justify-content:center;height:100vh;margin:0;">
        <h2>Build a network to view the interactive topology graph</h2>
        </body></html>
        """
        self.web_view.setHtml(html)

    def update_graph(self, G: nx.Graph) -> None:
        if not PYVIS_AVAILABLE:
            self.web_view.setHtml("<html><body style='background:#1e1e2e;color:#f38ba8;"
                                   "font-family:sans-serif;'><h3>pyvis is not installed."
                                   " Run: pip install pyvis</h3></body></html>")
            return
        if G.number_of_nodes() == 0:
            self._render_placeholder()
            return

        net = Network(height="100%", width="100%", bgcolor="#1e1e2e", font_color="#cdd6f4",
                       directed=False, notebook=False, cdn_resources="in_line")
        net.force_atlas_2based(gravity=-40, spring_length=120)

        betweenness_vals = [d.get("betweenness_centrality", 0.0) for _, d in G.nodes(data=True)]
        max_bet = max(betweenness_vals) if betweenness_vals else 1.0
        max_bet = max_bet or 1.0

        for node, data in G.nodes(data=True):
            bet = data.get("betweenness_centrality", 0.0)
            s_llps = data.get("s_llps", 0.0)
            size = 10 + 40 * (bet / max_bet)
            color = _llps_color(s_llps)
            role = data.get("role", "")
            label = data.get("label", node)
            title = (
                f"<b>{label}</b> ({node})<br>"
                f"Betweenness: {bet:.4f}<br>"
                f"S_LLPS: {s_llps:.3f}<br>"
                f"Role: {role}<br>"
                f"Disorder: {data.get('disorder_score', 0.0):.3f}"
            )
            net.add_node(node, label=label, title=title, size=size, color=color)

        for u, v, data in G.edges(data=True):
            conf = data.get("confidence", data.get("weight", 0.5))
            net.add_edge(u, v, value=max(0.5, conf * 5), title=f"confidence: {conf:.2f}",
                         color="rgba(180,180,200,0.35)")

        html_path = self._tmp_dir / "network.html"
        net.write_html(str(html_path), notebook=False, open_browser=False)
        self._inject_bridge_script(html_path)
        self.web_view.load(QUrl.fromLocalFile(str(html_path)))

    def _inject_bridge_script(self, html_path: Path) -> None:
        """Append QWebChannel wiring + node click -> Python relay into the PyVis HTML."""
        content = html_path.read_text(encoding="utf-8")
        bridge_script = """
        <script src="qrc:///qtwebchannel/qwebchannel.js"></script>
        <script type="text/javascript">
        document.addEventListener("DOMContentLoaded", function () {
            new QWebChannel(qt.webChannelTransport, function (channel) {
                window.pyBridge = channel.objects.bridge;
                if (typeof network !== "undefined") {
                    network.on("click", function (params) {
                        if (params.nodes.length > 0 && window.pyBridge) {
                            window.pyBridge.onNodeClick(String(params.nodes[0]));
                        }
                    });
                }
            });
        });
        </script>
        </body>
        """
        content = content.replace("</body>", bridge_script)
        html_path.write_text(content, encoding="utf-8")
