import math
import pickle
from signals_types import Signal, ResultOfOperation


class SampledSignal(Signal):
    def __init__(self, X, Y, name, fs, n1, l, source=None, no_reconstruction=False):
        self.X = list(X)
        self.Y = list(Y)
        self.name = name
        self.fs = fs
        self.n1 = n1
        self.l = l
        self.source = source
        self.no_reconstruction = no_reconstruction

    def value(self, t):
        if self.source is not None:
            return self.source.value(t)
        return None

    def samples(self):
        return self.X, self.Y

    def resample(self, new_fs):
        if self.source is None:
            raise ValueError(
                "Brak odniesienia do sygnału źródłowego - nie można próbkować ponownie.")
        t_start = self.n1 / self.fs
        l_new = round(self.l * new_fs / self.fs)
        n1_new = round(self.n1 * new_fs / self.fs)
        X, Y = [], []
        for i in range(l_new):
            t = t_start + i / new_fs
            X.append(t)
            Y.append(self.source.value(t))
        return SampledSignal(X, Y, self.name, new_fs, n1_new, l_new, source=self.source)

    def __str__(self):
        return self.name

    def save_bin(self, path):
        X, Y = self.samples()
        data = {
            "name": str(self),
            "X": X,
            "Y": Y,
            "fs": self.fs,
            "n1": self.n1,
            "l": self.l,
            "source": self.source,
            "no_reconstruction": getattr(self, 'no_reconstruction', False)
        }
        try:
            with open(path, "wb") as f:
                pickle.dump(data, f)
        except Exception:
            raise FileNotFoundError("Nie udało się zapisać pliku")

    def save_txt(self, path):
        X, Y = self.samples()
        try:
            with open(path, "w") as f:
                f.write(f"nazwa: {str(self)}\n")
                f.write(f"czestotliwosc probkowania (fs): {str(self.fs)}\n")
                f.write(f"numer pierwszej probki (n1): {str(self.n1)}\n")
                f.write(f"liczba probek (l): {str(self.l)}\n")
                if self.source is not None:
                    f.write(f"wzor: {str(self.source)}\n")
                else:
                    f.write("wzor: (brak - nie mozna probkowac ponownie)\n")
                for x, y in zip(X, Y):
                    f.write(f"{x}\t{y}\n")
        except Exception:
            raise FileNotFoundError("Nie udało się zapisać pliku")

    def _operation(self, other, op, symbol):
        X1, Y1 = self.samples()
        X2, Y2 = other.samples()
        if self.fs != other.fs or self.n1 != other.n1 or self.l != other.l:
            return None
        Y = [op(Y1[i], Y2[i]) for i in range(self.l)]
        if symbol == "/":
            good_vals = [y for y in Y if not math.isnan(y)]
            replacement = max(abs(v) for v in good_vals) if good_vals else 0
            Y = [replacement if math.isnan(y) else y for y in Y]
        if self.source is not None and other.source is not None:
            op_source = ResultOfOperation(self.source, other.source, op, symbol)
        else:
            op_source = None
        return SampledSignal(X1, Y, f"({self}){symbol}({other})", self.fs, self.n1, self.l,
                             source=op_source)

    def __add__(self, other):
        return self._operation(other, lambda a, b: a + b, "+")

    def __sub__(self, other):
        return self._operation(other, lambda a, b: a - b, "-")

    def __mul__(self, other):
        return self._operation(other, lambda a, b: a * b, "*")

    def __truediv__(self, other):
        def safe_div(a, b):
            return a / b if b > 1e-9 else math.nan
        return self._operation(other, safe_div, "/")


class QuantizedSignal(SampledSignal):
    def __init__(self, sampled, levels):
        self.original = sampled
        X, Y = sampled.samples()
        self.levels = levels
        ymin, ymax = min(Y), max(Y)
        if ymin != ymax:
            level_size = (ymax - ymin) / (levels - 1)
            quantized_Y = [
                ymin + max(0, min(levels - 1, round((y - ymin) / level_size))) * level_size
                for y in Y
            ]
        else:
            quantized_Y = list(Y)
        super().__init__(X, quantized_Y,
                         f"{str(sampled)} po kwantyzacji (poziomy: {levels})",
                         sampled.fs, sampled.n1, sampled.l, source=sampled.source)


class ReconstructedSignal(SampledSignal):
    def __init__(self, source_sig, fs_new, method="foh", sinc_half=10):
        self.original = source_sig
        self.method = method
        X_old, Y_old = source_sig.samples()
        fs_old = source_sig.fs
        if fs_new <= fs_old:
            raise ValueError(
                "Rekonstrukcja musi mieć większą częstotliwość próbkowania niż sygnał oryginalny")
        l_old = source_sig.l
        if method == "sinc" and sinc_half * 2 + 1 > l_old:
            raise ValueError(
                f"Liczba próbek sinc ({sinc_half * 2 + 1}) przekracza długość sygnału ({l_old}).")
        l_new = int(l_old * fs_new / fs_old)
        X_new, Y_new = [], []
        t_start = X_old[0]
        for i in range(l_new):
            t = t_start + i / fs_new
            X_new.append(t)
            if method == "foh":
                i_left = min(int((t - t_start) * fs_old), l_old - 2)
                i_right = i_left + 1
                X_left, X_right = X_old[i_left], X_old[i_right]
                Y_left, Y_right = Y_old[i_left], Y_old[i_right]
                Y_new.append((Y_right - Y_left) / (X_right - X_left) * (t - X_left) + Y_left)
            else:
                i_center = round((t - t_start) * fs_old)
                i_left = max(0, i_center - sinc_half)
                i_right = min(l_old, i_center + sinc_half + 1)
                Y_new.append(sum(
                    Y_old[k] * self._sinc((t - X_old[k]) * fs_old)
                    for k in range(i_left, i_right)
                ))

        n1_new = round(source_sig.n1 * fs_new / fs_old)
        super().__init__(X_new, Y_new,
                         f"{str(source_sig)} rekonstrukcja ({method}, fs={fs_new})",
                         fs_new, n1_new, l_new, source=source_sig.source)

    def _sinc(self, x):
        if abs(x) < 1e-10:
            return 1.0
        return math.sin(math.pi * x) / (math.pi * x)


# METRYKI JAKOŚCI

def mse(orig, reconstr):
    _, Y1 = orig.samples()
    _, Y2 = reconstr.samples()
    if orig.fs != reconstr.fs or orig.n1 != reconstr.n1 or orig.l != reconstr.l:
        return None
    return sum((y1 - y2) ** 2 for y1, y2 in zip(Y1, Y2)) / len(Y1)


def snr(orig, reconstr):
    _, Y1 = orig.samples()
    _, Y2 = reconstr.samples()
    if orig.fs != reconstr.fs or orig.n1 != reconstr.n1 or orig.l != reconstr.l:
        return None
    sum1 = sum(y1 ** 2 for y1 in Y1)
    sum2 = sum((y1 - y2) ** 2 for y1, y2 in zip(Y1, Y2))
    if sum2 == 0:
        return float('inf')
    return 10 * math.log10(sum1 / sum2)


def psnr(orig, reconstr):
    _, Y1 = orig.samples()
    m = mse(orig, reconstr)
    if not m:
        return None
    return 10 * math.log10(max(Y1) ** 2 / m)


def md(orig, reconstr):
    _, Y1 = orig.samples()
    _, Y2 = reconstr.samples()
    if orig.fs != reconstr.fs or orig.n1 != reconstr.n1 or orig.l != reconstr.l:
        return None
    return max(abs(y1 - y2) for y1, y2 in zip(Y1, Y2))
