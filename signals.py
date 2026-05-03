import math
import random
import pickle
import numpy as np
from abc import ABC, abstractmethod


class Signal(ABC):
    @abstractmethod
    def value(self, x):
        pass

    @abstractmethod
    def samples(self):
        pass

    @staticmethod
    def load(path):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception:
            raise FileNotFoundError("Nie udało się załadować pliku")
        return SampledSignal(data["X"], data["Y"],
                             f"Wczytany({data.get('name', '?')})",
                             data["fs"], data["n1"], data["l"],
                             data.get("source"), data.get("no_reconstruction"))

    def mean(self):
        _, Y = self.samples()
        if not Y:
            return 0
        return sum(Y) / len(Y)

    def mean_abs(self):
        _, Y = self.samples()
        if not Y:
            return 0
        return sum(abs(y) for y in Y) / len(Y)

    def power(self):
        _, Y = self.samples()
        if not Y:
            return 0
        return sum(y ** 2 for y in Y) / len(Y)

    def variance(self):
        _, Y = self.samples()
        if not Y:
            return 0
        m = self.mean()
        return sum((y - m) ** 2 for y in Y) / len(Y)

    def rms(self):
        return math.sqrt(self.power())

    def plot_signal(self, ax, draw_continuous=False):
        X, Y = self.samples()
        ax.axhline(0, color='darkgray', linewidth=1, linestyle="--")
        if draw_continuous:
            ax.plot(X, Y, color="lightblue", marker='.', markersize=2,
                    markeredgecolor="darkgreen", markerfacecolor="darkgreen")
        else:
            ax.scatter(X, Y, color="darkgreen", s=9)
        ax.set_title(str(self))
        ax.set_xlabel("t[s]")
        ax.set_ylabel("A")

    def plot_histogram(self, ax, bins=20):
        _, Y = self.samples()
        Y = np.array(Y)
        bin_size = (Y.max() - Y.min()) / bins
        if bin_size == 0:
            ax.bar([Y[0]], [len(Y)])
        else:
            edges = np.linspace(Y.min(), Y.max(), bins, endpoint=False)
            centers = edges + bin_size / 2
            classified = np.floor((Y - Y.min()) / bin_size).astype(int).clip(0, bins - 1)
            counts = np.bincount(classified)
            ax.bar(centers, counts, width=bin_size * 0.9)
        ax.set_title("Histogram")
        ax.set_xlabel("A")
        ax.set_ylabel("liczba próbek")

    @staticmethod
    def plot_comparison(ax, original, transformed, orig_label="Sygnał wzorcowy",
                        trans_label="Sygnał przetworzony"):
        X1, Y1 = original.samples()
        X2, Y2 = transformed.samples()
        ax.axhline(0, color='darkgray', linewidth=1, linestyle="--")
        ax.scatter(X1, Y1, color="green", s=9, label=orig_label, alpha=0.7)
        ax.scatter(X2, Y2, color="red", s=9, label=trans_label, alpha=0.7)
        ax.legend()
        ax.set_title("Porównanie")
        ax.set_xlabel("t[s]")
        ax.set_ylabel("A")


class ResultOfOperation:
    def __init__(self, a_source, b_source, op_fn, symbol):
        self.a_source = a_source
        self.b_source = b_source
        self.op_fn = op_fn
        self.symbol = symbol

    def value(self, t):
        return self.op_fn(self.a_source.value(t), self.b_source.value(t))

    def __str__(self):
        return f"({self.a_source}){self.symbol}({self.b_source})"


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
            if good_vals:
                replacement = max(abs(v) for v in good_vals)
            else:
                replacement = 0
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
            quantized_Y = [ymin + max(0, min(levels - 1, round((y - ymin) / level_size))) * level_size for y in Y]
        else:
            quantized_Y = list(Y)
        super().__init__(X, quantized_Y, f"{str(sampled)} po kwantyzacji (poziomy: {levels})",
                         sampled.fs, sampled.n1, sampled.l, source=sampled.source)


class ReconstructedSignal(SampledSignal):
    def __init__(self, source_sig, fs_new, method="foh", sinc_half=10):
        self.original = source_sig
        self.method = method
        X_old, Y_old = source_sig.samples()
        fs_old = source_sig.fs
        if fs_new <= fs_old:
            raise ValueError("Rekonstrukcja musi mieć większą częstotliwość próbkowania niż sygnał oryginalny")
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
                i_left = int((t - t_start) * fs_old)
                if i_left >= l_old - 1:
                    i_left = l_old - 2
                i_right = i_left + 1

                X_left = X_old[i_left]
                X_right = X_old[i_right]
                Y_left = Y_old[i_left]
                Y_right = Y_old[i_right]

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


class ContinuousSignal(Signal, ABC):
    @abstractmethod
    def value(self, t):
        pass

    def __init__(self, A, t1, fs, d):
        self.A = A
        self.t1 = t1
        self.d = d
        self.fs = fs

    def samples(self):
        X, Y = [], []
        n = int(self.fs * self.d)
        for i in range(n):
            t = self.t1 + i / self.fs
            X.append(t)
            Y.append(self.value(t))
        return X, Y

    def _t_in_domain(self, t):
        return t >= self.t1 and t <= self.d + self.t1


class DiscreteSignal(Signal, ABC):
    @abstractmethod
    def value(self, n):
        pass

    def __init__(self, A, n1, l, fs):
        self.A = A
        self.n1 = n1
        self.l = l
        self.fs = fs
        self.n2 = l + n1 - 1

    def samples(self):
        X, Y = [], []
        for i in range(self.l):
            n = self.n1 + i
            t = n / self.fs
            X.append(t)
            Y.append(self.value(n))
        return X, Y

    def _n_in_domain(self, n):
        return n >= self.n1 and n <= self.n2


# SYGNAŁY CIĄGŁE

# Szum o rozkładzie jednostajnym
class S1(ContinuousSignal):
    def value(self, t):
        if self._t_in_domain(t):
            return random.uniform(-self.A, self.A)
        else:
            return 0

    def __str__(self):
        return "Szum o rozkładzie jednostajnym"


# Szum gaussowski
class S2(ContinuousSignal):
    def value(self, t):
        if self._t_in_domain(t):
            return random.gauss(0, self.A)
        else:
            return 0

    def __str__(self):
        return "Szum gaussowski"


# Sygnał sinusoidalny
class S3(ContinuousSignal):
    def __init__(self, A, T, t1, fs, d):
        super().__init__(A, t1, fs, d)
        self.T = T

    def value(self, t):
        if self._t_in_domain(t):
            return self.A * math.sin(2 * math.pi / self.T * (t - self.t1))
        else:
            return 0

    def __str__(self):
        return "Sygnał sinusoidalny"


# Sygnał sinusoidalny wyprostowany jednopołówkowo
class S4(ContinuousSignal):
    def __init__(self, A, T, t1, fs, d):
        super().__init__(A, t1, fs, d)
        self.T = T

    def value(self, t):
        if self._t_in_domain(t):
            s = math.sin(2 * math.pi / self.T * (t - self.t1))
            return 0.5 * self.A * (s + abs(s))
        else:
            return 0

    def __str__(self):
        return "Sygnał sinusoidalny wyprostowany jednopołówkowo"


# Sygnał sinusoidalny wyprostowany dwupołówkowo
class S5(ContinuousSignal):
    def __init__(self, A, T, t1, fs, d):
        super().__init__(A, t1, fs, d)
        self.T = T

    def value(self, t):
        if self._t_in_domain(t):
            return self.A * abs(math.sin(2 * math.pi / self.T * (t - self.t1)))
        else:
            return 0

    def __str__(self):
        return "Sygnał sinusoidalny wyprostowany dwupołówkowo"


# Sygnał prostokątny
class S6(ContinuousSignal):
    def __init__(self, A, T, t1, fs, d, kw):
        super().__init__(A, t1, fs, d)
        self.T = T
        self.kw = kw
 
    def value(self, t):
        if self._t_in_domain(t):
            k = math.floor((t - self.t1) / self.T)
            local = t - k * self.T - self.t1
            if local < 0:
                local = 0
            if local < self.kw * self.T:
                return self.A
            return 0
        else:
            return 0
 
    def __str__(self):
        return "Sygnał prostokątny"
 
 
# Sygnał prostokątny symetryczny
class S7(ContinuousSignal):
    def __init__(self, A, T, t1, fs, d, kw):
        super().__init__(A, t1, fs, d)
        self.T = T
        self.kw = kw
 
    def value(self, t):
        if self._t_in_domain(t):
            k = math.floor((t - self.t1) / self.T)
            local = t - k * self.T - self.t1
            if local < 0:
                local = 0
            if local < self.kw * self.T:
                return self.A
            return -self.A
        else:
            return 0
 
    def __str__(self):
        return "Sygnał prostokątny symetryczny"
 
 
# Sygnał trójkątny
class S8(ContinuousSignal):
    def __init__(self, A, T, t1, fs, d, kw):
        super().__init__(A, t1, fs, d)
        self.T = T
        self.kw = kw
 
    def value(self, t):
        if self._t_in_domain(t):
            k = math.floor((t - self.t1) / self.T)
            local = t - k * self.T - self.t1
            if local < 0:
                local = 0
            if local < self.kw * self.T:
                return self.A / (self.kw * self.T) * local
            return -self.A / (self.T * (1 - self.kw)) * local + self.A / (1 - self.kw)
        else:
            return 0
 
    def __str__(self):
        return "Sygnał trójkątny"


# Skok jednostkowy
class S9(ContinuousSignal):
    def __init__(self, A, t1, fs, d, ts):
        super().__init__(A, t1, fs, d)
        self.ts = ts

    def value(self, t):
        if self._t_in_domain(t):
            if t > self.ts:
                return self.A
            elif t == self.ts:
                return 0.5 * self.A
            else:
                return 0
        else:
            return 0

    def __str__(self):
        return "Skok jednostkowy"


# SYGNAŁY DYSKRETNE

# Impuls jednostkowy
class S10(DiscreteSignal):
    def __init__(self, A, ns, n1, l, fs):
        super().__init__(A, n1, l, fs)
        self.ns = ns

    def value(self, n):
        if self._n_in_domain(n):
            if n == self.ns:
                return self.A
            else:
                return 0
        else:
            return 0

    def __str__(self):
        return "Impuls jednostkowy"


# Szum impulsowy
class S11(DiscreteSignal):
    def __init__(self, A, p, n1, l, fs):
        super().__init__(A, n1, l, fs)
        self.p = p

    def value(self, n):
        if self._n_in_domain(n):
            if random.random() < self.p:
                return self.A
            else:
                return 0
        else:
            return 0

    def __str__(self):
        return "Szum impulsowy"


def mse(orig, reconstr):
    _, Y1 = orig.samples()
    _, Y2 = reconstr.samples()
    if orig.fs != reconstr.fs or orig.n1 != reconstr.n1 or orig.l != reconstr.l:
        return None
    return sum((y1 - y2)**2 for y1, y2 in zip(Y1, Y2)) / len(Y1)


def snr(orig, reconstr):
    _, Y1 = orig.samples()
    _, Y2 = reconstr.samples()
    if orig.fs != reconstr.fs or orig.n1 != reconstr.n1 or orig.l != reconstr.l:
        return None
    sum1 = sum(y1**2 for y1 in Y1)
    sum2 = sum((y1 - y2)**2 for y1, y2 in zip(Y1, Y2))
    if sum2 == 0:
        return float('inf')
    return 10 * math.log10(sum1 / sum2)


def psnr(orig, reconstr):
    _, Y1 = orig.samples()
    m = mse(orig, reconstr)
    if not m:
        return None
    return 10 * math.log10(max(Y1)**2 / m)


def md(orig, reconstr):
    _, Y1 = orig.samples()
    _, Y2 = reconstr.samples()
    if orig.fs != reconstr.fs or orig.n1 != reconstr.n1 or orig.l != reconstr.l:
        return None
    return max(abs(y1 - y2) for y1, y2 in zip(Y1, Y2))