import math
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
        from signals_sampled import SampledSignal
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
        return self.t1 <= t <= self.d + self.t1


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
        return self.n1 <= n <= self.n2
