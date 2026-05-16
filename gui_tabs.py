import math
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QLineEdit,
    QPushButton, QSpinBox, QGroupBox, QMessageBox, QDialog, QTextEdit, QFileDialog,
)
from PySide6.QtCore import Qt
from signals_funcs import S1, S2
from signals_sampled import QuantizedSignal, ReconstructedSignal, mse, snr, psnr, md
from signals_types import ContinuousSignal, Signal
from gui_config import (
    SIGNAL_DEFS, PARAM_RANGE, PARAM_DEFAULTS, MAX_SAMPLES,
    to_sampled, compute_sample_count,
)


# Elementy wspólne zakładek

class NavWidget(QHBoxLayout):
    def __init__(self, on_prev, on_next):
        super().__init__()
        btn_prev = QPushButton("< Poprzedni")
        btn_prev.clicked.connect(on_prev)
        btn_next = QPushButton("Następny >")
        btn_next.clicked.connect(on_next)
        self.label = QLabel("Brak sygnałów")
        self.label.setAlignment(Qt.AlignCenter)
        self.addWidget(btn_prev)
        self.addWidget(self.label)
        self.addWidget(btn_next)


class FileStatsWidget(QVBoxLayout):
    def __init__(self, on_save_bin, on_load, on_save_txt, on_show_txt):
        super().__init__()
        io_group = QGroupBox("Plik")
        io_lay = QVBoxLayout(io_group)
        row1 = QHBoxLayout()
        btn_save_bin = QPushButton("Zapisz (.bin)")
        btn_save_bin.clicked.connect(on_save_bin)
        btn_load = QPushButton("Wczytaj (.bin)")
        btn_load.clicked.connect(on_load)
        row1.addWidget(btn_save_bin)
        row1.addWidget(btn_load)
        row2 = QHBoxLayout()
        btn_save_txt = QPushButton("Eksportuj jako .txt")
        btn_save_txt.clicked.connect(on_save_txt)
        btn_show_txt = QPushButton("Pokaż .txt")
        btn_show_txt.clicked.connect(on_show_txt)
        row2.addWidget(btn_save_txt)
        row2.addWidget(btn_show_txt)
        io_lay.addLayout(row1)
        io_lay.addLayout(row2)
        self.addWidget(io_group)

        self.addWidget(QLabel("Statystyki:"))
        self.stats = QTextEdit()
        self.stats.setReadOnly(True)
        self.addWidget(self.stats)


# Baza dla wszystkich zakładek (nawigacja, operacje na plikach)

class BaseTab(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app

    def _add_extension(self, path, ext):
        return path if path.lower().endswith(ext.lower()) else path + ext

    def _save_file(self, fmt, parent):
        if self.app.current_idx < 0:
            return
        sig = self.app.signals[self.app.current_idx]
        if fmt == "bin":
            path, _ = QFileDialog.getSaveFileName(parent, "Zapis binarny", "", "Binary (*.bin)")
            if path:
                try:
                    sig.save_bin(self._add_extension(path, ".bin"))
                except FileNotFoundError as e:
                    QMessageBox.warning(parent, "Błąd", str(e))
        else:
            path, _ = QFileDialog.getSaveFileName(parent, "Plik tekstowy", "", "Signal text (*.sig.txt)")
            if path:
                try:
                    sig.save_txt(self._add_extension(path, ".sig.txt"))
                except FileNotFoundError as e:
                    QMessageBox.warning(parent, "Błąd", str(e))

    def _load_file(self, parent):
        path, _ = QFileDialog.getOpenFileName(parent, "Wczytaj sygnał", "", "Binary (*.bin)")
        if path:
            try:
                sig = Signal.load(path)
            except FileNotFoundError as e:
                QMessageBox.warning(parent, "Błąd", str(e))
                return
            self.app.signals.append(sig)
            self.app.current_idx = len(self.app.signals) - 1
            self.app.refresh_all_combos()
            self.app.show_current()

    def _show_txt_file(self, parent):
        path, _ = QFileDialog.getOpenFileName(parent, "Otwórz plik tekstowy", "", "Signal text (*.sig.txt)")
        if not path:
            return
        try:
            with open(path, "r") as f:
                content = f.read()
        except Exception as e:
            QMessageBox.warning(parent, "Błąd", str(e))
            return
        dlg = QDialog(parent)
        dlg.setWindowTitle(path)
        dlg.resize(600, 400)
        lay = QVBoxLayout(dlg)
        te = QTextEdit()
        te.setReadOnly(True)
        te.setPlainText(content)
        lay.addWidget(te)
        dlg.exec()


# Zakładka generatora

class GeneratorTab(BaseTab):
    def __init__(self, app):
        super().__init__(app)
        lay = QVBoxLayout(self)

        self.type_combo = QComboBox()
        self.type_combo.addItems(SIGNAL_DEFS.keys())
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        lay.addWidget(QLabel("Typ sygnału:"))
        lay.addWidget(self.type_combo)

        self.params_group = QGroupBox("Parametry")
        self.params_layout = QVBoxLayout(self.params_group)
        self.param_inputs = {}
        lay.addWidget(self.params_group)

        h = QHBoxLayout()
        h.addWidget(QLabel("Przedziały histogramu:"))
        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(2, 200)
        self.bins_spin.setValue(20)
        h.addWidget(self.bins_spin)
        lay.addLayout(h)

        btn_gen = QPushButton("Wygeneruj")
        btn_gen.clicked.connect(self.generate)
        lay.addWidget(btn_gen)

        self.nav = NavWidget(lambda: app.navigate(-1), lambda: app.navigate(1))
        lay.addLayout(self.nav)

        fs_widget = FileStatsWidget(
            lambda: self._save_file("bin", self),
            lambda: self._load_file(self),
            lambda: self._save_file("txt", self),
            lambda: self._show_txt_file(self),
        )
        lay.addLayout(fs_widget)
        self.stats = fs_widget.stats

        self._on_type_changed(self.type_combo.currentText())

    def _on_type_changed(self, text):
        for i in reversed(range(self.params_layout.count())):
            w = self.params_layout.itemAt(i).widget()
            if w:
                w.deleteLater()
        self.param_inputs.clear()
        if text not in SIGNAL_DEFS:
            return
        _, params = SIGNAL_DEFS[text]
        for p in params:
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.addWidget(QLabel(p + ":"))
            le = QLineEdit(PARAM_DEFAULTS.get(p, "1"))
            rl.addWidget(le)
            self.params_layout.addWidget(row)
            self.param_inputs[p] = le

    def _get_params(self):
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
                f"Maksymalna dozwolona to {MAX_SAMPLES}.\n"
                f"Zmniejsz fs lub czas trwania.")
        return result

    def generate(self):
        text = self.type_combo.currentText()
        cls, param_names = SIGNAL_DEFS[text]
        try:
            params = self._get_params()
        except ValueError as e:
            QMessageBox.warning(self, "Błąd", str(e))
            return
        args = [params[p] for p in param_names]
        sig = cls(*args)
        is_continuous = isinstance(sig, ContinuousSignal)
        if is_continuous and int(sig.fs * sig.d) < 1:
            QMessageBox.warning(self, "Błąd",
                "Nie można wygenerować sygnału: liczba próbek wynosi 0.")
            return
        sampled = to_sampled(sig, is_continuous)
        sampled.draw_continuous = is_continuous and cls not in (S1, S2)
        sampled.no_reconstruction = not is_continuous or cls in (S1, S2)
        self.app.signals.append(sampled)
        self.app.current_idx = len(self.app.signals) - 1
        self.app.show_comparison = False
        self.app.refresh_all_combos()
        self.app.show_current()


# Zakładka kwantyzacji

class QuantizationTab(BaseTab):
    def __init__(self, app):
        super().__init__(app)
        lay = QVBoxLayout(self)

        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Sygnał źródłowy:"))
        self.source_combo = QComboBox()
        src_row.addWidget(self.source_combo)
        lay.addLayout(src_row)

        lvl_row = QHBoxLayout()
        lvl_row.addWidget(QLabel("Liczba poziomów:"))
        self.levels_spin = QSpinBox()
        self.levels_spin.setRange(2, 2 ** 16)
        self.levels_spin.setValue(16)
        lvl_row.addWidget(self.levels_spin)
        lay.addLayout(lvl_row)

        btn = QPushButton("Kwantyzuj")
        btn.clicked.connect(self.quantize)
        lay.addWidget(btn)

        self.nav = NavWidget(lambda: app.navigate(-1), lambda: app.navigate(1))
        lay.addLayout(self.nav)

        fs_widget = FileStatsWidget(
            lambda: self._save_file("bin", self),
            lambda: self._load_file(self),
            lambda: self._save_file("txt", self),
            lambda: self._show_txt_file(self),
        )
        lay.addLayout(fs_widget)
        self.stats = fs_widget.stats

    def quantize(self):
        if not self.app.signals:
            QMessageBox.warning(self, "Błąd", "Brak sygnałów do kwantyzacji.")
            return
        idx = self.source_combo.currentIndex()
        if idx < 0 or idx >= len(self.app.signals):
            return
        try:
            q = QuantizedSignal(self.app.signals[idx], self.levels_spin.value())
        except Exception as e:
            QMessageBox.warning(self, "Błąd", str(e))
            return
        self.app.signals.append(q)
        self.app.current_idx = len(self.app.signals) - 1
        self.app.show_comparison = False
        self.app.refresh_all_combos()
        self.app.show_current()


# Zakładka rekonstrukcji

class ReconstructionTab(BaseTab):
    def __init__(self, app):
        super().__init__(app)
        lay = QVBoxLayout(self)

        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Sygnał źródłowy:"))
        self.source_combo = QComboBox()
        src_row.addWidget(self.source_combo)
        lay.addLayout(src_row)

        fs_row = QHBoxLayout()
        fs_row.addWidget(QLabel("Nowe fs [Hz]:"))
        self.fs_edit = QLineEdit("2000")
        fs_row.addWidget(self.fs_edit)
        lay.addLayout(fs_row)

        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Metoda:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["foh", "sinc"])
        self.method_combo.currentTextChanged.connect(self._update_sinc_group)
        method_row.addWidget(self.method_combo)
        lay.addLayout(method_row)

        self.sinc_group = QGroupBox()
        sinc_lay = QVBoxLayout(self.sinc_group)
        sinc_row = QHBoxLayout()
        sinc_row.addWidget(QLabel("Liczba sąsiadów po jednej stronie:"))
        self.sinc_half = QSpinBox()
        self.sinc_half.setRange(1, 10000)
        self.sinc_half.setValue(10)
        sinc_row.addWidget(self.sinc_half)
        sinc_lay.addLayout(sinc_row)
        lay.addWidget(self.sinc_group)
        self._update_sinc_group(self.method_combo.currentText())

        btn = QPushButton("Rekonstruuj")
        btn.clicked.connect(self.reconstruct)
        lay.addWidget(btn)

        self.nav = NavWidget(lambda: app.navigate(-1), lambda: app.navigate(1))
        lay.addLayout(self.nav)

        fs_widget = FileStatsWidget(
            lambda: self._save_file("bin", self),
            lambda: self._load_file(self),
            lambda: self._save_file("txt", self),
            lambda: self._show_txt_file(self),
        )
        lay.addLayout(fs_widget)
        self.stats = fs_widget.stats

    def _update_sinc_group(self, method):
        self.sinc_group.setEnabled(method == "sinc")

    def reconstruct(self):
        if not self.app.signals:
            QMessageBox.warning(self, "Błąd", "Brak sygnałów do rekonstrukcji.")
            return
        idx = self.source_combo.currentIndex()
        if idx < 0 or idx >= len(self.app.signals):
            return
        source = self.app.signals[idx]
        if getattr(source, 'no_reconstruction', False):
            QMessageBox.warning(self, "Błąd",
                "Nie można rekonstruować sygnałów dyskretnych i szumów.")
            return
        try:
            fs_new = float(self.fs_edit.text().strip())
            if fs_new <= 0:
                raise ValueError("fs musi być > 0")
        except ValueError as e:
            QMessageBox.warning(self, "Błąd", f"Nieprawidłowe fs: {e}")
            return
        expected = int(source.l * fs_new / source.fs)
        if expected > MAX_SAMPLES:
            QMessageBox.warning(self, "Błąd",
                f"Zbyt duża liczba próbek po rekonstrukcji: {expected}.\n"
                f"Maksymalna dozwolona to {MAX_SAMPLES}.")
            return
        method = self.method_combo.currentText()
        sinc_half = self.sinc_half.value()
        if method == "sinc" and sinc_half * 2 + 1 > source.l:
            self.sinc_half.setStyleSheet("border: 1px solid red;")
            QMessageBox.warning(self, "Błąd",
                f"Liczba próbek ({sinc_half * 2 + 1}) przekracza długość sygnału ({source.l}).")
            return
        self.sinc_half.setStyleSheet("")
        try:
            r = ReconstructedSignal(source, fs_new, method=method, sinc_half=sinc_half)
        except ValueError as e:
            QMessageBox.warning(self, "Błąd", str(e))
            return
        self.app.signals.append(r)
        self.app.current_idx = len(self.app.signals) - 1
        self.app.show_comparison = False
        self.app.refresh_all_combos()
        self.app.show_current()


# Zakładka operacji na sygnałach

class OperationsTab(BaseTab):
    def __init__(self, app):
        super().__init__(app)
        lay = QVBoxLayout(self)

        op_row = QHBoxLayout()
        self.op_a = QComboBox()
        self.op_b = QComboBox()
        self.op_type = QComboBox()
        self.op_type.addItems(["+", "-", "*", "/"])
        op_row.addWidget(self.op_a)
        op_row.addWidget(self.op_type)
        op_row.addWidget(self.op_b)
        lay.addLayout(op_row)

        btn = QPushButton("Oblicz")
        btn.clicked.connect(self.compute)
        lay.addWidget(btn)

        self.nav = NavWidget(lambda: app.navigate(-1), lambda: app.navigate(1))
        lay.addLayout(self.nav)

        fs_widget = FileStatsWidget(
            lambda: self._save_file("bin", self),
            lambda: self._load_file(self),
            lambda: self._save_file("txt", self),
            lambda: self._show_txt_file(self),
        )
        lay.addLayout(fs_widget)
        self.stats = fs_widget.stats

    def compute(self):
        if len(self.app.signals) < 1:
            return
        a = self.app.signals[self.op_a.currentIndex()]
        b = self.app.signals[self.op_b.currentIndex()]
        op = self.op_type.currentText()
        ops = {"+": a.__add__, "-": a.__sub__, "*": a.__mul__, "/": a.__truediv__}
        try:
            result = ops[op](b)
            if not result:
                raise ValueError(
                    "Sygnały muszą mieć tę samą częstotliwość próbkowania, "
                    "liczbę próbek oraz zaczynać się w tym samym czasie.")
        except ValueError as e:
            QMessageBox.warning(self, "Błąd", str(e))
            return
        self.app.signals.append(result)
        self.app.current_idx = len(self.app.signals) - 1
        self.app.show_comparison = False
        self.app.refresh_all_combos()
        self.app.show_current()


def build_stats_text(sig):
    stats = (
        f"Średnia: {sig.mean():.6f}\n"
        f"Średnia bezwzględna: {sig.mean_abs():.6f}\n"
        f"Moc: {sig.power():.6f}\n"
        f"Wariancja: {sig.variance():.6f}\n"
        f"Wartość skuteczna: {sig.rms():.6f}"
    )
    has_comparison = hasattr(sig, 'original') and sig.original is not None
    if not has_comparison:
        return stats

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
    if s is None:
        stats += "\nSNR: N/A"
    elif s == float('inf'):
        stats += "\nSNR: ∞"
    else:
        stats += f"\nSNR: {s:.6f} dB"
    stats += f"\nPSNR: {p:.6f} dB" if p is not None else "\nPSNR: N/A"
    stats += f"\nMD: {d:.6f}" if d is not None else "\nMD: N/A"

    from signals_sampled import QuantizedSignal
    if isinstance(sig, QuantizedSignal):
        b = math.log2(sig.levels)
        st = 6.02 * b + 1.76
        enob = (s - 1.76) / 6.02 if s is not None and s != float('inf') else None
        stats += f"\nSNR (teoretyczne): {st:.6f} dB"
        stats += f"\nENOB: {enob:.6f}" if enob is not None else "\nENOB: N/A"

    return stats
