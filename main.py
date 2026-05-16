import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QMessageBox, QTabWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from signals_sampled import ReconstructedSignal
from signals_types import Signal
from gui_tabs import GeneratorTab, QuantizationTab, ReconstructionTab, OperationsTab, build_stats_text


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Generator Sygnałów")
        self.signals = []
        self.current_idx = -1
        self.show_comparison = False

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # lewa część (menu)
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.West)
        root.addWidget(self.tabs, 1)

        self.tab_gen   = GeneratorTab(self)
        self.tab_quant = QuantizationTab(self)
        self.tab_recon = ReconstructionTab(self)
        self.tab_ops   = OperationsTab(self)

        self.tabs.addTab(self.tab_gen,   "Generator")
        self.tabs.addTab(self.tab_quant, "Kwantyzacja")
        self.tabs.addTab(self.tab_recon, "Rekonstrukcja")
        self.tabs.addTab(self.tab_ops,   "Operacje")

        self._nav_labels = [t.nav.label for t in (self.tab_gen, self.tab_quant,
                                                    self.tab_recon, self.tab_ops)]
        self._stats_texts = [t.stats for t in (self.tab_gen, self.tab_quant,
                                                self.tab_recon, self.tab_ops)]

        # prawa część (wykresy)
        right = QVBoxLayout()
        root.addLayout(right, 2)

        self.btn_toggle_view = QPushButton("Pokaż porównanie")
        self.btn_toggle_view.clicked.connect(self.toggle_view)
        right.addWidget(self.btn_toggle_view)

        self.fig = Figure(figsize=(6, 6))
        self.canvas = FigureCanvasQTAgg(self.fig)
        right.addWidget(self.canvas)

        self.resize(1100, 750)


    def refresh_all_combos(self):
        items = [f"[{i + 1}] {s.name}" for i, s in enumerate(self.signals)]
        for combo in (
            self.tab_ops.op_a, self.tab_ops.op_b,
            self.tab_quant.source_combo,
            self.tab_recon.source_combo,
        ):
            combo.clear()
            combo.addItems(items)


    def navigate(self, delta):
        if not self.signals:
            return
        self.current_idx = max(0, min(len(self.signals) - 1, self.current_idx + delta))
        self.show_comparison = False
        self.show_current()


    def toggle_view(self):
        if self.current_idx < 0:
            return
        sig = self.signals[self.current_idx]
        if not (hasattr(sig, 'original') and sig.original is not None):
            QMessageBox.information(self, "Info", "Brak kolejnych wykresów dla tego sygnału")
            return
        self.show_comparison = not self.show_comparison
        self.show_current()


    def show_current(self):
        if self.current_idx < 0 or self.current_idx >= len(self.signals):
            return
        sig = self.signals[self.current_idx]
        draw_continuous = getattr(sig, 'draw_continuous', False)
        has_comparison = hasattr(sig, 'original') and sig.original is not None

        nav_text = f"{self.current_idx + 1}/{len(self.signals)}"
        for lbl in self._nav_labels:
            lbl.setText(nav_text)

        if has_comparison:
            self.btn_toggle_view.setEnabled(True)
            self.btn_toggle_view.setText(
                "Pokaż wykres + histogram" if self.show_comparison else "Pokaż porównanie")
        else:
            self.btn_toggle_view.setEnabled(False)
            self.btn_toggle_view.setText("Pokaż porównanie")
            self.show_comparison = False

        bins = self.tab_gen.bins_spin.value()
        self.fig.clear()

        if self.show_comparison and has_comparison:
            ax = self.fig.add_subplot(111)
            original_for_plot = sig.original
            if isinstance(sig, ReconstructedSignal) and sig.original.source is not None:
                try:
                    original_for_plot = sig.original.resample(sig.fs)
                except ValueError:
                    pass
            label_orig = "Oryginał"
            label_trans = "Rekonstrukcja" if isinstance(sig, ReconstructedSignal) else "Po kwantyzacji"
            Signal.plot_comparison(ax, original_for_plot, sig,
                                   orig_label=label_orig, trans_label=label_trans)
        else:
            ax1 = self.fig.add_subplot(211)
            sig.plot_signal(ax1, draw_continuous=draw_continuous)
            ax2 = self.fig.add_subplot(212)
            sig.plot_histogram(ax2, bins=bins)

        self.fig.tight_layout()
        self.canvas.draw()

        stats = build_stats_text(sig)
        for st in self._stats_texts:
            st.setText(stats)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
