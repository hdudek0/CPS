import sys
import math
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QLabel, QLineEdit, QPushButton, QSpinBox, QTextEdit, QFileDialog, QGroupBox,
    QMessageBox, QDialog, QTabWidget)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from signals import (S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11,
                     SampledSignal, ContinuousSignal, Signal,
                     QuantizedSignal, ReconstructedSignal,
                     mse, snr, psnr, md)

MAX_SAMPLES = 1_000_000

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

PARAM_RANGE = {
    "A": (1e-9, 1e9, False),
    "T[s]": (1e-9, 1e9, False),
    "t1[s]": (-1e9, 1e9, False),
    "d[s]": (1e-9, 1e9, False),
    "kw": (0.0, 1.0, False),
    "ts[s]": (-1e9, 1e9, False),
    "n1": (-1e9, 1e9, True),
    "l": (1, 1e9, True),
    "fs[Hz]": (1e-9, 1e9, False),
    "ns": (-1e9, 1e9, True),
    "p": (0.0, 1.0, False),
    "fs": (1e-9, 1e9, False),
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
    return SampledSignal(X, Y, str(sig), fs, n1, l, source=sig)


def compute_sample_count(params, param_names):
    if "d[s]" in params and "fs[Hz]" in params:
        return params["fs[Hz]"] * params["d[s]"]
    elif "l" in params and "fs" in params:
        return params["l"]
    elif "l" in params and "fs[Hz]" in params:
        return params["l"]
    return 0


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Generator Sygnałów")
        self.signals = []
        self.current_idx = -1
        self.show_comparison = False
        self.stats_texts = []
        self.nav_labels = []

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.West)
        root.addWidget(tabs, 1)

        # --- Generator ---
        tab_gen = QWidget()
        gen_lay = QVBoxLayout(tab_gen)

        self.type_combo = QComboBox()
        self.type_combo.addItems(SIGNAL_DEFS.keys())
        self.type_combo.currentTextChanged.connect(self.on_type_changed)
        gen_lay.addWidget(QLabel("Typ sygnału:"))
        gen_lay.addWidget(self.type_combo)

        self.params_group = QGroupBox("Parametry")
        self.params_layout = QVBoxLayout(self.params_group)
        self.param_inputs = {}
        gen_lay.addWidget(self.params_group)

        h = QHBoxLayout()
        h.addWidget(QLabel("Przedziały histogramu:"))
        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(2, 200)
        self.bins_spin.setValue(20)
        h.addWidget(self.bins_spin)
        gen_lay.addLayout(h)

        btn_gen = QPushButton("Wygeneruj")
        btn_gen.clicked.connect(self.generate)
        gen_lay.addWidget(btn_gen)

        self._make_signal_nav_widget(gen_lay)
        self._make_file_stats_widget(gen_lay)
        tabs.addTab(tab_gen, "Generator")

        # --- Kwantyzacja ---
        tab_quant = QWidget()
        quant_lay = QVBoxLayout(tab_quant)

        quant_src_row = QHBoxLayout()
        quant_src_row.addWidget(QLabel("Sygnał źródłowy:"))
        self.quant_source = QComboBox()
        quant_src_row.addWidget(self.quant_source)
        quant_lay.addLayout(quant_src_row)

        quant_row = QHBoxLayout()
        quant_row.addWidget(QLabel("Liczba poziomów:"))
        self.quant_levels = QSpinBox()
        self.quant_levels.setRange(2, 2**16)
        self.quant_levels.setValue(16)
        quant_row.addWidget(self.quant_levels)
        quant_lay.addLayout(quant_row)

        btn_quant = QPushButton("Kwantyzuj")
        btn_quant.clicked.connect(self.do_quantize)
        quant_lay.addWidget(btn_quant)

        self._make_signal_nav_widget(quant_lay)
        self._make_file_stats_widget(quant_lay)
        tabs.addTab(tab_quant, "Kwantyzacja")

        # --- Rekonstrukcja ---
        tab_recon = QWidget()
        recon_lay = QVBoxLayout(tab_recon)

        recon_src_row = QHBoxLayout()
        recon_src_row.addWidget(QLabel("Sygnał źródłowy:"))
        self.recon_source = QComboBox()
        recon_src_row.addWidget(self.recon_source)
        recon_lay.addLayout(recon_src_row)

        recon_fs_row = QHBoxLayout()
        recon_fs_row.addWidget(QLabel("Nowe fs [Hz]:"))
        self.recon_fs = QLineEdit("2000")
        recon_fs_row.addWidget(self.recon_fs)
        recon_lay.addLayout(recon_fs_row)

        recon_method_row = QHBoxLayout()
        recon_method_row.addWidget(QLabel("Metoda:"))
        self.recon_method = QComboBox()
        self.recon_method.addItems(["foh", "sinc"])
        recon_method_row.addWidget(self.recon_method)
        recon_lay.addLayout(recon_method_row)

        self.sinc_group = QGroupBox()
        sinc_lay = QVBoxLayout(self.sinc_group)
        sinc_row1 = QHBoxLayout()
        sinc_row1.addWidget(QLabel("Liczba sąsiadów po jednej stronie:"))
        self.sinc_half = QSpinBox()
        self.sinc_half.setRange(1, 10000)
        self.sinc_half.setValue(10)
        sinc_row1.addWidget(self.sinc_half)
        sinc_lay.addLayout(sinc_row1)
        recon_lay.addWidget(self.sinc_group)

        self.recon_method.currentTextChanged.connect(self._update_sinc_group)
        self._update_sinc_group(self.recon_method.currentText())

        btn_recon = QPushButton("Rekonstruuj")
        btn_recon.clicked.connect(self.do_reconstruct)
        recon_lay.addWidget(btn_recon)

        self._make_signal_nav_widget(recon_lay)
        self._make_file_stats_widget(recon_lay)
        tabs.addTab(tab_recon, "Rekonstrukcja")

        # --- Operacje ---
        tab_ops = QWidget()
        op_lay = QVBoxLayout(tab_ops)

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

        self._make_signal_nav_widget(op_lay)
        self._make_file_stats_widget(op_lay)
        tabs.addTab(tab_ops, "Operacje")

        # --- Prawy panel ---
        right = QVBoxLayout()
        root.addLayout(right, 2)

        self.btn_toggle_view = QPushButton("Pokaż porównanie")
        self.btn_toggle_view.clicked.connect(self.toggle_view)
        right.addWidget(self.btn_toggle_view)

        self.fig = Figure(figsize=(6, 6))
        self.canvas = FigureCanvasQTAgg(self.fig)
        right.addWidget(self.canvas)

        self.on_type_changed(self.type_combo.currentText())
        self.resize(1100, 750)

    def _make_signal_nav_widget(self, lay):
        nav = QHBoxLayout()
        btn_prev = QPushButton("< Poprzedni")
        btn_prev.clicked.connect(lambda: self.navigate(-1))
        btn_next = QPushButton("Następny >")
        btn_next.clicked.connect(lambda: self.navigate(1))
        label = QLabel("Brak sygnałów")
        label.setAlignment(Qt.AlignCenter)
        nav.addWidget(btn_prev)
        nav.addWidget(label)
        nav.addWidget(btn_next)
        lay.addLayout(nav)
        self.nav_labels.append(label)

    def _make_file_stats_widget(self, lay):
        io_group = QGroupBox("Plik")
        io_btn_lay = QVBoxLayout(io_group)
        row1 = QHBoxLayout()
        btn_save_bin = QPushButton("Zapisz (.bin)")
        btn_save_bin.clicked.connect(lambda: self.save_file("bin"))
        btn_load = QPushButton("Wczytaj (.bin)")
        btn_load.clicked.connect(self.load_file)
        row1.addWidget(btn_save_bin)
        row1.addWidget(btn_load)
        row2 = QHBoxLayout()
        btn_save_txt = QPushButton("Eksportuj jako .txt")
        btn_save_txt.clicked.connect(lambda: self.save_file("txt"))
        btn_show_txt = QPushButton("Pokaż .txt")
        btn_show_txt.clicked.connect(self.show_txt_file)
        row2.addWidget(btn_save_txt)
        row2.addWidget(btn_show_txt)
        io_btn_lay.addLayout(row1)
        io_btn_lay.addLayout(row2)
        lay.addWidget(io_group)

        lay.addWidget(QLabel("Statystyki:"))
        st = QTextEdit()
        st.setReadOnly(True)
        lay.addWidget(st)
        self.stats_texts.append(st)

    def _update_sinc_group(self, method):
        self.sinc_group.setEnabled(method == "sinc")

    def on_type_changed(self, text):
        for i in reversed(range(self.params_layout.count())):
            w = self.params_layout.itemAt(i).widget()
            if w:
                w.deleteLater()
        self.param_inputs.clear()

        if text not in SIGNAL_DEFS:
            return
        _, params = SIGNAL_DEFS[text]
        defaults = {"A": "5", "T[s]": "1", "t1[s]": "0", "d[s]": "5", "kw": "0.5",
                    "ts[s]": "1", "n1": "0", "l": "10", "fs[Hz]": "1000", "ns": "5",
                    "p": "0.5", "fs": "1000"}
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
            try:
                val = float(raw)
            except ValueError:
                errors.append(f"  - {k}: '{raw}' nie jest liczbą")
                widget.setStyleSheet("border: 1px solid red;")
                continue

            pr = PARAM_RANGE.get(k)
            valid = True
            if pr:
                lo, hi, is_int = pr
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

            widget.setStyleSheet("")
            result[k] = val

        if errors:
            raise ValueError("Błędne parametry:\n" + "\n".join(errors))

        sample_count = compute_sample_count(result, list(self.param_inputs.keys()))
        if sample_count > MAX_SAMPLES:
            raise ValueError(
                f"Zbyt duża liczba próbek: {int(sample_count)}.\n"
                f"Maksymalna dozwolona liczba próbek to {MAX_SAMPLES}.\n"
                f"Zmniejsz fs lub czas trwania / liczbę próbek.")

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
        if is_continuous and int(sig.fs * sig.d) < 1:
            QMessageBox.warning(self, "Błąd",
                "Nie można wygenerować sygnału: w wybranej konfiguracji"
                " parametrów liczba próbek wynosi 0.")
            return
        sampled = to_sampled(sig, is_continuous)
        sampled.draw_continuous = is_continuous and cls not in (S1, S2)
        sampled.no_reconstruction = not is_continuous or cls in (S1, S2)
        self.signals.append(sampled)
        self.current_idx = len(self.signals) - 1
        self.show_comparison = False
        self.refresh_all_combos()
        self.show_current()

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
        has_comparison = hasattr(sig, 'original') and sig.original is not None
        if not has_comparison:
            QMessageBox.information(self, "Info",
                "Brak kolejnych wykresów dla tego sygnału")
            return
        self.show_comparison = not self.show_comparison
        self.show_current()

    def show_current(self):
        if self.current_idx < 0 or self.current_idx >= len(self.signals):
            return
        sig = self.signals[self.current_idx]
        draw_continuous = getattr(sig, 'draw_continuous', False)
        has_comparison = hasattr(sig, 'original') and sig.original is not None
        for lbl in self.nav_labels:
            lbl.setText(f"{self.current_idx + 1}/{len(self.signals)}")

        if has_comparison:
            self.btn_toggle_view.setEnabled(True)
            self.btn_toggle_view.setText(
                "Pokaż wykres + histogram" if self.show_comparison else "Pokaż porównanie")
        else:
            self.btn_toggle_view.setEnabled(False)
            self.btn_toggle_view.setText("Pokaż porównanie")
            self.show_comparison = False

        bins = self.bins_spin.value()
        self.fig.clear()

        if self.show_comparison and has_comparison:
            ax = self.fig.add_subplot(111)
            original_for_plot = sig.original

            if isinstance(sig, ReconstructedSignal):
                if sig.original.source is not None:
                    try:
                        original_for_plot = sig.original.resample(sig.fs)
                    except ValueError:
                        pass
                Signal.plot_comparison(ax, original_for_plot, sig,
                                   orig_label="Oryginał", trans_label="Rekonstrukcja")
            else:
                Signal.plot_comparison(ax, original_for_plot, sig,
                                   orig_label="Oryginał", trans_label="Po kwantyzacji")
        else:
            ax1 = self.fig.add_subplot(211)
            sig.plot_signal(ax1, draw_continuous=draw_continuous)

            ax2 = self.fig.add_subplot(212)
            sig.plot_histogram(ax2, bins=bins)

        self.fig.tight_layout()
        self.canvas.draw()

        stats = (
            f"Średnia: {sig.mean():.6f}\n"
            f"Średnia bezwzględna: {sig.mean_abs():.6f}\n"
            f"Moc: {sig.power():.6f}\n"
            f"Wariancja: {sig.variance():.6f}\n"
            f"Wartość skuteczna: {sig.rms():.6f}"
        )

        if has_comparison:
            original_for_metrics = sig.original
            if isinstance(sig, ReconstructedSignal):
                if sig.original.source is not None:
                    try:
                        original_for_metrics = sig.original.resample(sig.fs)
                    except ValueError:
                        original_for_metrics = sig.original

            m = mse(original_for_metrics, sig)
            s = snr(original_for_metrics, sig)
            p = psnr(original_for_metrics, sig)
            d = md(original_for_metrics, sig)
            stats += f"\nMSE: {m:.6f}" if m is not None else "\nMSE: N/A"
            stats += f"\nSNR: {s:.6f} dB" if s is not None and s != float('inf') else f"\nSNR: {'∞' if s == float('inf') else 'N/A'}"
            stats += f"\nPSNR: {p:.6f} dB" if p is not None else "\nPSNR: N/A"
            stats += f"\nMD: {d:.6f}" if d is not None else "\nMD: N/A"
            if isinstance(sig, QuantizedSignal):
                b = math.log2(sig.levels)
                st = 6.02 * b + 1.76
                enob = (s - 1.76) / 6.02
                stats += f"\nSNR (teoretyczne): {st:.6f} dB" if st is not None else "\nSNR (teoretyczne): N/A"
                stats += f"\nENOB: {enob:.6f}" if s is not None and s != float('inf') else "\nENOB: N/A"

        for st in self.stats_texts:
            st.setText(stats)

    def refresh_all_combos(self):
        for combo in (self.op_a, self.op_b, self.quant_source, self.recon_source):
            combo.clear()
            for i, s in enumerate(self.signals):
                combo.addItem(f"[{i+1}] {s.name}")

    def do_quantize(self):
        if not self.signals:
            QMessageBox.warning(self, "Błąd", "Brak sygnałów do kwantyzacji.")
            return
        idx = self.quant_source.currentIndex()
        if idx < 0 or idx >= len(self.signals):
            return
        source = self.signals[idx]
        levels = self.quant_levels.value()
        try:
            q = QuantizedSignal(source, levels)
        except Exception as e:
            QMessageBox.warning(self, "Błąd", str(e))
            return
        self.signals.append(q)
        self.current_idx = len(self.signals) - 1
        self.show_comparison = False
        self.refresh_all_combos()
        self.show_current()

    def do_reconstruct(self):
        if not self.signals:
            QMessageBox.warning(self, "Błąd", "Brak sygnałów do rekonstrukcji.")
            return
        idx = self.recon_source.currentIndex()
        if idx < 0 or idx >= len(self.signals):
            return
        source = self.signals[idx]
        if getattr(source, 'no_reconstruction', False):
            QMessageBox.warning(self, "Błąd", "Nie można rekonstruować sygnałów dyskretnych i szumów.")
            return
        try:
            fs_new = float(self.recon_fs.text().strip())
            if fs_new <= 0:
                raise ValueError("fs musi być > 0")
        except ValueError as e:
            QMessageBox.warning(self, "Błąd", f"Nieprawidłowe fs: {e}")
            return

        expected_samples = int(source.l * fs_new / source.fs)
        if expected_samples > MAX_SAMPLES:
            QMessageBox.warning(self, "Błąd",
                f"Zbyt duża liczba próbek po rekonstrukcji: {expected_samples}.\n"
                f"Maksymalna dozwolona to {MAX_SAMPLES}.")
            return

        method = self.recon_method.currentText()
        sinc_half = self.sinc_half.value()

        if method == "sinc":
            l_old = source.l
            if sinc_half * 2 + 1 > l_old:
                self.sinc_half.setStyleSheet("border: 1px solid red;")
                QMessageBox.warning(self, "Błąd",
                    f"Liczba sąsiadów ({sinc_half * 2 + 1}) przekracza długość sygnału ({l_old}).")
                return
            self.sinc_half.setStyleSheet("")
        else:
            self.sinc_half.setStyleSheet("")

        try:
            r = ReconstructedSignal(source, fs_new, method=method, sinc_half=sinc_half)
        except ValueError as e:
            QMessageBox.warning(self, "Błąd", str(e))
            return

        self.signals.append(r)
        self.current_idx = len(self.signals) - 1
        self.show_comparison = False
        self.refresh_all_combos()
        self.show_current()

    def do_operation(self):
        try:
            if len(self.signals) < 1:
                return
            a = self.signals[self.op_a.currentIndex()]
            b = self.signals[self.op_b.currentIndex()]
            op = self.op_type.currentText()
            ops = {"+": a.__add__, "-": a.__sub__, "*": a.__mul__, "/": a.__truediv__}
            result = ops[op](b)
            if not result:
                raise ValueError("Sygnały muszą mieć tę samą częstotliwość próbkowania, "
                                 "liczbę próbek oraz zaczynać się w tym samym czasie.")
            self.signals.append(result)
            self.current_idx = len(self.signals) - 1
            self.show_comparison = False
            self.refresh_all_combos()
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
            self.refresh_all_combos()
            self.show_current()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
