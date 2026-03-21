import sys
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QLabel, QLineEdit, QPushButton, QSpinBox, QTextEdit, QFileDialog, QGroupBox,
    QMessageBox, QDialog)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from signals import (S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11,
                     SampledSignal, ContinuousSignal, Signal)

SIGNAL_DEFS = {
    "Szum o rozkładzie jednostajnym": (S1, ["A", "t1[s]", "fs[Hz]", "d[s]"]),
    "Szum gaussowski": (S2, ["A", "t1[s]", "fs[Hz]", "d[s]"]),
    "Sygnał sinusoidalny": (S3, ["A", "T[s]", "t1[s]", "fs[Hz]", "d[s]"]),
    "Sygnał sinusoidalny wyprostowany jednopołówkowo": (S4, ["A", "T[s]", "t1[s]", "fs[Hz]", "d[s]"]),
    "Sygnał sinusoidalny wyprostowany dwupołówkowo": (S5, ["A", "T[s]", "t1[s]", "fs[Hz]", "d[s]"]),
    "Sygnał prostokątny": (S6, ["A", "T[s]", "t1[s]", "fs[Hz]", "d[s]", "kw"]),
    "Sygnał prostokątny symetryczny": (S7, ["A", "T[s]", "t1[s]", "fs[Hz]", "d[s]", "kw"]),
    "Sygnał trójkątny": (S8, ["A", "T[s]", "t1[s]", "fs[Hz]", "d[s]", "kw"]),
    "Skok jednostkowy": (S9, ["A", "t1[s]", "fs[Hz]", "d[s]", "ts[s]"]),
    "Impuls jednostkowy": (S10, ["A", "ns", "n1", "l", "fs[Hz]"]),
    "Szum impulsowy": (S11, ["A", "p", "n1", "l", "fs"])
}

# minimum, maksimum, czy całkowity
PARAM_RANGE = {
    "A": (None, None, False),
    "T": (1e-9, None, False), # okres musi być > 0
    "t1": (None, None, False),
    "d": (1e-9, None, False), # czas trwania musi być > 0
    "kw": (0.0, 1.0, False), # współczynnik wypełnienia musi być w [0, 1]
    "ts": (None, None, False),
    "n1": (None, None, True),
    "l": (1, None, True), # liczba próbek musi być >= 1
    "fs": (1e-9, None, False), # częstotliwość próbkowania musi być > 0
    "ns": (None, None, True),
    "p": (0.0, 1.0, False), # prawdopodobieństwo musi być w [0, 1]
}

def to_sampled(sig, is_continuous):
    X, Y = sig.samples()
    fs = sig.fs
    if is_continuous:
        n1 = int(sig.t1 * fs)
        l = int(sig.d * fs)
    else:
        n1 = int(sig.n1)
        l = int(sig.l)
    return SampledSignal(X, Y, str(sig), fs, n1, l)


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
        h.addWidget(QLabel("Liczba przedziałów histogramu:"))
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
        btn_load = QPushButton("Wczytaj")
        btn_load.clicked.connect(self.load_file)
        btn_save_txt = QPushButton("Eksportuj jako .txt")
        btn_save_txt.clicked.connect(lambda: self.save_file("txt"))
        btn_show_txt = QPushButton("Pokaż .txt")
        btn_show_txt.clicked.connect(self.show_txt_file)
        io_row.addWidget(btn_save_bin)
        io_row.addWidget(btn_load)
        io_row.addWidget(btn_save_txt)
        io_row.addWidget(btn_show_txt)
        left.addLayout(io_row)

        # statystyki
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setFixedHeight(100)
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
        _, params = SIGNAL_DEFS[text]
        defaults = {"A": "5", "T[s]": "1", "t1[s]": "0", "d[s]": "5", "kw": "0.5",
                    "ts[s]": "1", "n1": "0", "l": "10", "fs[Hz]": "1000", "ns": "5", "p": "0.5"}
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
        result = {}
        errors = []
 
        for k, widget in self.param_inputs.items():
            raw = widget.text().strip()
            # podano nie-liczbę
            try:
                val = float(raw)
            except ValueError:
                errors.append(f"  - {k}: '{raw}' nie jest liczbą")
                widget.setStyleSheet("border: 1px solid red;")
                continue
            # złe wartości parametrów
            range = PARAM_RANGE.get(k)
            valid = True
            if range:
                lo, hi, is_int = range
                if is_int:
                    val = int(val)
                if lo is not None and val < lo:
                    errors.append(f"  - {k}: minimalna wartość to {lo}")
                    valid = False
                elif hi is not None and val > hi:
                    errors.append(f"  - {k}: maksymalna wartość to {hi}")
                    valid = False
 
            if not valid:
                widget.setStyleSheet("border: 1px solid red;")
                continue
 
            widget.setStyleSheet("") # usuwa czerwoną obramówkę po wcześniejszym błędzie
            result[k] = val
 
        if errors:
            raise ValueError("Błędne parametry:\n" + "\n".join(errors))
        return result

    def generate(self):
        text = self.type_combo.currentText()
        cls, param_names = SIGNAL_DEFS[text]
        try:
            params = self.get_params()
        except ValueError as e:
            QMessageBox.warning(self, "Błąd", str(e))
            return
        args = [params[p] for p in param_names]
        sig = cls(*args)
        is_continuous = isinstance(sig, ContinuousSignal)
        sampled = to_sampled(sig, is_continuous)
        # cls not in (S1, S2) -> szumy również rysujemy punktami
        sampled.draw_continuous = is_continuous and cls not in (S1, S2)
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
        draw_continuous = getattr(sig, 'draw_continuous', False)
        self.nav_label.setText(f"{self.current_idx + 1}/{len(self.signals)}")

        X, Y = sig.samples()
        bins = self.bins_spin.value()

        self.fig.clear()
        ax1 = self.fig.add_subplot(211) # 2 rzędy, 1 kolumna, index od 1
        ax1.axhline(0, color='darkgray', linewidth=1, linestyle="--")
        if draw_continuous:
            ax1.plot(X, Y, color="lightblue", marker='.', markersize=2,
                     markeredgecolor="darkgreen", markerfacecolor="darkgreen")
        else:
            ax1.scatter(X, Y, color="darkgreen", s=9)
        ax1.set_title(str(sig))
        ax1.set_xlabel("t[s]")
        ax1.set_ylabel("A")

        ax2 = self.fig.add_subplot(212)
        Y = np.array(Y)
        bin_size = (Y.max() - Y.min()) / bins
        if bin_size == 0:
            ax2.bar([Y[0]], [len(Y)])
        else:
            edges = np.linspace(Y.min(), Y.max(), bins, endpoint=False)
            centers = edges + bin_size / 2
            classified = np.floor((Y - Y.min()) / bin_size).astype(int).clip(0, bins - 1)
            counts = np.bincount(classified)
            # niewielkie przerwy dla czytelności
            ax2.bar(centers, counts, width=bin_size * 0.9)
        ax2.set_title("Histogram")
        ax2.set_xlabel("A")
        ax2.set_ylabel("liczba próbek")

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
        try:
            if len(self.signals) < 2:
                return
            a = self.signals[self.op_a.currentIndex()]
            b = self.signals[self.op_b.currentIndex()]
            op = self.op_type.currentText()
            ops = {"+": a.__add__, "-": a.__sub__, "*": a.__mul__, "/": a.__truediv__}
            result = ops[op](b)
            if not result:
                raise ValueError("Sygnały muszą mieć tę samą częstotliwość próbkowania, liczbę próbek oraz zaczynać się w tym samym czasie.")
            self.signals.append(result)
            self.current_idx = len(self.signals) - 1
            self.refresh_op_combos()
            self.show_current()
        except ValueError as e:
            QMessageBox.warning(self, "Błąd", str(e))
        
    def add_extension(self, path, ext):
        if not path.lower().endswith(ext.lower()):
            path += ext
        return path

    def save_file(self, fmt):
        if self.current_idx < 0:
            return
        sig = self.signals[self.current_idx]
        if fmt == "bin":
            path, _ = QFileDialog.getSaveFileName(self, "Zapis binarny", "", "Binary (*.bin)")
            if path:
                try:
                    sig.save_bin(self.add_extension(path, ".bin"))
                except FileNotFoundError as e:
                    QMessageBox.warning(self, "Błąd", str(e))
        else:
            path, _ = QFileDialog.getSaveFileName(self, "Plik tekstowy", "", "Signal text (*.sig.txt)")
            if path:
                try:
                    sig.save_txt(self.add_extension(path, ".sig.txt"))
                except FileNotFoundError as e:
                    QMessageBox.warning(self, "Błąd", str(e))

    def show_txt_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Otwórz plik tekstowy", "", "Signal text (*.sig.txt)")
        if not path:
            return
        try:
            with open(path, "r") as f:
                content = f.read()
        except Exception as e:
            QMessageBox.warning(self, "Błąd", str(e))
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(path)
        dlg.resize(600, 400)
        layout = QVBoxLayout(dlg)
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText(content)
        layout.addWidget(text_edit)
        dlg.exec()

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Zapis binarny", "", "Binary (*.bin)")
        if path:
            try:
                sig = Signal.load(path)
            except FileNotFoundError as e:
                QMessageBox.warning(self, "Błąd", str(e))
                return
            self.signals.append(sig)
            self.current_idx = len(self.signals) - 1
            self.refresh_op_combos()
            self.show_current()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
