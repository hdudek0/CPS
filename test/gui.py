import sys
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QLabel, QLineEdit, QPushButton, QSpinBox, QTextEdit, QFileDialog, QGroupBox)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from signals import (S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11,
                     SampledSignal, Signal)

SIGNAL_DEFS = {
    "Szum o rozkładzie jednostajnym": (S1, ["A", "t1", "d"], False),
    "Szum gaussowski": (S2, ["A", "t1", "d"], False),
    "Sygnał sinusoidalny": (S3, ["A", "T", "t1", "d"], False),
    "Sygnał sinusoidalny wyprostowany jednopołówkowo": (S4, ["A", "T", "t1", "d"], False),
    "Sygnał sinusoidalny wyprostowany dwupołówkowo": (S5, ["A", "T", "t1", "d"], False),
    "Sygnał prostokątny": (S6, ["A", "T", "t1", "d", "kw"], False),
    "Sygnał prostokątny symetryczny": (S7, ["A", "T", "t1", "d", "kw"], False),
    "Sygnał trójkątny": (S8, ["A", "T", "t1", "d", "kw"], False),
    "Skok jednostkowy": (S9, ["A", "t1", "d", "ts"], False),
    "Impuls jednostkowy": (S10, ["A", "ns", "n1", "l", "fs"], True),
    "Szum impulsowy": (S11, ["A", "p", "n1", "l", "fs"], True)
}

def to_sampled(sig):
    X, Y = sig.samples()
    return SampledSignal(X, Y, is_discrete=getattr(sig, "is_discrete", False), name=str(sig))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Generator Sygnałów")
        self.signals = []  # lista typu SampledSignal
        self.current_idx = -1

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # sterowanie po lewej
        left = QVBoxLayout()
        root.addLayout(left, 1)

        # wybór sygnału
        self.type_combo = QComboBox()
        self.type_combo.addItems(SIGNAL_DEFS.keys())
        self.type_combo.currentTextChanged.connect(self.on_type_changed)
        left.addWidget(QLabel("Typ sygnału:"))
        left.addWidget(self.type_combo)

        # parametry sygnału
        self.params_group = QGroupBox("Parametry")
        self.params_layout = QVBoxLayout(self.params_group)
        self.param_inputs = {}
        left.addWidget(self.params_group)

        # beans
        h = QHBoxLayout()
        h.addWidget(QLabel("Binsy histogramu:")) # jak to jest po polsku?
        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(2, 200)
        self.bins_spin.setValue(20)
        h.addWidget(self.bins_spin)
        left.addLayout(h)

        btn_gen = QPushButton("Wygeneruj")
        btn_gen.clicked.connect(self.generate)
        left.addWidget(btn_gen)

        # nawigacja między sygnałami
        nav = QHBoxLayout()
        self.btn_prev = QPushButton("< Poprzedni")
        self.btn_prev.clicked.connect(lambda: self.navigate(-1))
        self.btn_next = QPushButton("Następny >")
        self.btn_next.clicked.connect(lambda: self.navigate(1))
        self.nav_label = QLabel("Brak sygnałów")
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.nav_label)
        nav.addWidget(self.btn_next)
        left.addLayout(nav)

        # operacje
        op_group = QGroupBox("Operacje")
        op_lay = QVBoxLayout(op_group)
        op_row = QHBoxLayout()
        self.op_a = QComboBox()
        self.op_b = QComboBox()
        self.op_type = QComboBox()
        self.op_type.addItems(["+", "-", "*", "/"])
        op_row.addWidget(self.op_a)
        op_row.addWidget(self.op_type)
        op_row.addWidget(self.op_b)
        op_lay.addLayout(op_row)
        btn_op = QPushButton("Oblicz")
        btn_op.clicked.connect(self.do_operation)
        op_lay.addWidget(btn_op)
        left.addWidget(op_group)

        # pliki
        io_row = QHBoxLayout()
        btn_save_bin = QPushButton("Zapisz")
        btn_save_bin.clicked.connect(lambda: self.save_file("bin"))
        btn_save_txt = QPushButton("Eksportuj jako .txt")
        btn_save_txt.clicked.connect(lambda: self.save_file("txt"))
        btn_load = QPushButton("Wczytaj")
        btn_load.clicked.connect(self.load_file)
        io_row.addWidget(btn_save_bin)
        io_row.addWidget(btn_save_txt)
        io_row.addWidget(btn_load)
        left.addLayout(io_row)

        # statystyki
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(120)
        left.addWidget(QLabel("Statystyki:"))
        left.addWidget(self.stats_text)
        left.addStretch()

        # wykresy po prawej
        right = QVBoxLayout()
        root.addLayout(right, 2)

        self.fig = Figure(figsize=(6, 6))
        self.canvas = FigureCanvasQTAgg(self.fig)
        right.addWidget(self.canvas)

        self.on_type_changed(self.type_combo.currentText())
        self.resize(1000, 600)

    def on_type_changed(self, text):
        # wyczyszczenie pól parametrów
        for i in reversed(range(self.params_layout.count())):
            w = self.params_layout.itemAt(i).widget()
            if w:
                w.deleteLater()
        self.param_inputs.clear()

        if text not in SIGNAL_DEFS:
            return
        _, params, _ = SIGNAL_DEFS[text]
        defaults = {"A": "1", "T": "1", "t1": "0", "d": "2", "kw": "0.5",
                    "ts": "1", "n1": "0", "l": "20", "fs": "100", "ns": "5", "p": "0.5"}
        for p in params:
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.addWidget(QLabel(p + ":"))
            le = QLineEdit(defaults.get(p, "1"))
            rl.addWidget(le)
            self.params_layout.addWidget(row)
            self.param_inputs[p] = le

    def get_params(self):
        return {k: float(v.text()) for k, v in self.param_inputs.items()}

    def generate(self):
        text = self.type_combo.currentText()
        cls, param_names, _ = SIGNAL_DEFS[text]
        params = self.get_params()
        args = [params[p] for p in param_names]
        # discrete signals need int for some params <- ?
        sig = cls(*args)
        sampled = to_sampled(sig)
        self.signals.append(sampled)
        self.current_idx = len(self.signals) - 1
        self.refresh_op_combos()
        self.show_current()

    def navigate(self, delta):
        if not self.signals:
            return
        self.current_idx = max(0, min(len(self.signals) - 1, self.current_idx + delta))
        self.show_current()

    def show_current(self):
        if self.current_idx < 0 or self.current_idx >= len(self.signals):
            return
        sig = self.signals[self.current_idx]
        self.nav_label.setText(f"{self.current_idx + 1}/{len(self.signals)}")

        X, Y = sig.samples()
        bins = self.bins_spin.value()

        self.fig.clear()
        ax1 = self.fig.add_subplot(211)
        ax1.axhline(0, color='darkgray', linewidth=1, linestyle="--")
        if sig.is_discrete:
            ax1.scatter(X, Y, s=10)
        else:
            ax1.plot(X, Y, color="darkgreen")
        ax1.set_title(str(sig))
        ax1.set_xlabel("t[s]")
        ax1.set_ylabel("A")

        ax2 = self.fig.add_subplot(212)
        Yarr = np.array(Y)
        if len(Yarr) > 0 and Yarr.max() != Yarr.min():
            ax2.hist(Yarr, bins=bins)
        ax2.set_title("Histogram")
        ax2.set_xlabel("A")
        ax2.set_ylabel("ilość")

        self.fig.tight_layout()
        self.canvas.draw()

        self.stats_text.setText(
            f"Średnia: {sig.mean():.6f}\n"
            f"Średnia bezwzględna: {sig.mean_abs():.6f}\n"
            f"Moc: {sig.power():.6f}\n"
            f"Wariancja: {sig.variance():.6f}\n"
            f"Wartość skuteczna: {sig.rms():.6f}"
        )

    def refresh_op_combos(self):
        for combo in (self.op_a, self.op_b):
            combo.clear()
            for i, s in enumerate(self.signals):
                combo.addItem(f"[{i+1}] {s.name}")

    def do_operation(self):
        if len(self.signals) < 2:
            return
        a = self.signals[self.op_a.currentIndex()]
        b = self.signals[self.op_b.currentIndex()]
        op = self.op_type.currentText()
        ops = {"+": a.__add__, "-": a.__sub__, "*": a.__mul__, "/": a.__truediv__}
        result = ops[op](b)
        self.signals.append(result)
        self.current_idx = len(self.signals) - 1
        self.refresh_op_combos()
        self.show_current()

    def save_file(self, fmt):
        if self.current_idx < 0:
            return
        sig = self.signals[self.current_idx]
        if fmt == "bin":
            path, _ = QFileDialog.getSaveFileName(self, "Zapis binarny", "", "Binary (*.bin)")
            if path:
                sig.save_bin(path)
        else:
            path, _ = QFileDialog.getSaveFileName(self, "Plik tekstowy", "", "Text (*.txt)")
            if path:
                sig.save_txt(path)

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Zapis binarny", "", "Binary (*.bin)")
        if path:
            sig = Signal.load(path)
            self.signals.append(sig)
            self.current_idx = len(self.signals) - 1
            self.refresh_op_combos()
            self.show_current()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
