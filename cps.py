import math
import numpy as np
import random
import pickle
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from scipy import integrate

# TODO: pozabezpieczać przed dzieleniem przez 0 i przedziały dla niektórych parametrów (T > 0, kw != 1, itp)
# TODO: co robić z tym errorem z całek - zignorować 
# TODO: GUI -> wyliczać częstotliwość i różne rzeczy na podstawie podanych parametrów ZAMIAST DOMYŚLNEJ
# TODO: załatwić problem z funkcjami opartymi o losowość - S1, S2, S11 TRUDNE SPORE
# pomysł na S1 i S2 - dodać jakąś klasę z zapisywaniem wartości
# a S11 można nawet w niej samej idk
# TODO: zmienić x, y na A i t PÓŹNIEJ

class Signal(ABC):
    @abstractmethod
    def value(self, x):
        pass

    @abstractmethod
    def samples(self):
        pass

    def plot(self):
        X, Y = self.samples()
        plt.axhline(0, color='darkgray', linewidth=1, linestyle="--")
        plt.axvline(0, color='darkgray', linewidth=1, linestyle="--")

        if getattr(self, "is_discrete", False):
            plt.scatter(X, Y)
            plt.xlabel("t[s]", fontsize=12, fontweight="bold")
        else:
            plt.plot(X, Y, color="darkgreen")
            plt.xlabel("t[s]", fontsize=12, fontweight="bold")

        plt.title(str(self), fontsize=14, fontweight="bold")
        plt.ylabel("A", rotation=0, fontsize=12, fontweight="bold")
        plt.show()

    def histogram(self, bins):
        _, Y = self.samples()
        Y = np.array(Y)
        bin_size = (Y.max() - Y.min()) / bins
        edges = np.arange(Y.min(), Y.max(), bin_size)
        classified = np.floor((Y - Y.min()) / bin_size).astype(int).clip(0, bins - 1)
        counts = np.bincount(classified)
        plt.bar(edges, counts, width=bin_size)
        plt.title("Histogram: " + str(self))
        plt.xlabel("A")
        plt.ylabel("liczność")
        plt.show()

    def save_bin(self, path):
        X, Y = self.samples()
        data = {
            "name": str(self),
            "is_discrete": getattr(self, "is_discrete", False),
            "X": X,
            "Y": Y
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    # równie dobrze może być poza klasą, ale dla porządku
    # statyczna w signal
    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return SampledSignal(data["X"], data["Y"], is_discrete=data.get("is_discrete", False),
                             name=f"Loaded({data.get('name', '?')})")

    def save_txt(self, path):
        X, Y = self.samples()
        with open(path, "w") as f:
            f.write(f"name: {str(self)}\n")
            f.write(f"is discrete: {getattr(self, 'is_discrete', False)}\n")
            for x, y in zip(X, Y):
                f.write(f"{x}\t{y}\n")

    def _operation(self, other, op, symbol):
        X1, Y1 = self.samples()
        X2, Y2 = other.samples()
        n = min(len(Y1), len(Y2))
        X = X1[:n]
        Y = [op(Y1[i], Y2[i]) for i in range(n)]
        is_discrete = getattr(self, "is_discrete", False) or getattr(other, "is_discrete", False)
        return SampledSignal(X, Y, is_discrete=is_discrete, name=f"({self}){symbol}({other})")

    def __add__(self, other):
        return self._operation(other, lambda a, b: a + b, "+")

    def __sub__(self, other):
        return self._operation(other, lambda a, b: a - b, "-")

    def __mul__(self, other):
        return self._operation(other, lambda a, b: a * b, "*")
    
    def __truediv__(self, other):
        return self._operation(other, lambda a, b: a / b, "/")
    
    
    @abstractmethod
    def mean(self):
        pass

    @abstractmethod
    def mean_abs(self):
        pass

    @abstractmethod
    def power(self):
        pass

    @abstractmethod
    def variance(self):
        pass

    def rms(self):
        return math.sqrt(self.power())
    

class SampledSignal(Signal):
    def __init__(self, X, Y, is_discrete=False, name="SampledSignal"):
        self.X = X
        self.Y = Y
        self.is_discrete = is_discrete
        self.name = name

    # TODO: chyba nie mieć value, używać samples
    def value(self, x):
        return None

    def samples(self):
        return self.X, self.Y

    def __str__(self):
        return self.name
    
    def mean(self):
        if not self.Y:
            return 0
        return sum(self.Y) / len(self.Y)

    def mean_abs(self):
        if not self.Y:
            return 0
        return sum(abs(y) for y in self.Y) / len(self.Y)

    def power(self):
        if not self.Y:
            return 0
        return sum(y ** 2 for y in self.Y) / len(self.Y)

    def variance(self):
        if not self.Y:
            return 0
        m = self.mean()
        return sum((y - m) ** 2 for y in self.Y) / len(self.Y)
    

class ContinuousSignal(Signal):
    is_discrete = False

    def __init__(self, A, t1, d, f=2000):
        self.A = A
        self.t1 = t1
        self.d = d
        self.f = f
        self.t2 = d + t1

    def samples(self):
        X, Y = [], []
        # jeśli d lub f nie całkowite to liczba próbek zaokrąglona w dółs do jedności
        n = int(self.f * self.d)
        for i in range(n):
            t = self.t1 + i / self.f
            X.append(t)
            Y.append(self.value(t))
        return X, Y
    
    def mean(self):
        integral, _ = integrate.quad(self.value, self.t1, self.t2)
        return integral / (self.t2 - self.t1)
    
    def mean_abs(self):
        integral, _ = integrate.quad(lambda t: abs(self.value(t)), self.t1, self.t2)
        return integral / (self.t2 - self.t1)
    
    def power(self):
        integral, _ = integrate.quad(lambda t: self.value(t)**2, self.t1, self.t2)
        return integral / (self.t2 - self.t1)
    
    def variance(self):
        mean_val = self.mean()
        integral, _ = integrate.quad(lambda t: (self.value(t) - mean_val)**2, self.t1, self.t2)
        return integral / (self.t2 - self.t1)


class DiscreteSignal(Signal):
    is_discrete = True

    def __init__(self, A, n1, l, f=10):
        self.A = A
        self.n1 = n1
        self.l = l #l = n2 - n1 + 1 -> nie musi być osobno n2 na razie
        self.f = f

    def samples(self):
        X, Y = [], []
        for i in range(self.l):
            n = self.n1 + i
            t = n / self.f
            X.append(t)
            Y.append(self.value(n))
        return X, Y
    
    def mean(self):
        return sum(self.value(n) for n in range(self.n1, self.l + self.n1)) / self.l
    
    def mean_abs(self):
        return sum(abs(self.value(n)) for n in range(self.n1, self.l + self.n1)) / self.l
    
    def power(self):
        return sum(self.value(n)**2 for n in range(self.n1, self.l + self.n1)) / self.l
    
    def variance(self):
        mean_val = self.mean()
        return sum((self.value(n) - mean_val)**2 for n in range(self.n1, self.l + self.n1)) / self.l


# SYGNAŁY CIĄGŁE

# Szum o rozkładzie jednostajnym
class S1(ContinuousSignal):
    def value(self, t):
        return random.uniform(-self.A, self.A)

    def __str__(self):
        return "Szum o rozkładzie jednostajnym"


# Szum gaussowski
class S2(ContinuousSignal):
    def value(self, t):
        return random.gauss(0, self.A)

    def __str__(self):
        return "Szum gaussowski"


# Sygnał sinusoidalny
class S3(ContinuousSignal):
    def __init__(self, A, T, t1, d, f=2000):
        super().__init__(A, t1, d, f)
        self.T = T

    def value(self, t):
        return self.A * math.sin(2 * math.pi / self.T * (t - self.t1))

    def __str__(self):
        return "Sygnał sinusoidalny"


# Sygnał sinusoidalny wyprostowany jednopołówkowo
class S4(ContinuousSignal):
    def __init__(self, A, T, t1, d, f=2000):
        super().__init__(A, t1, d, f)
        self.T = T

    # poprawka wzoru
    def value(self, t):
        s = math.sin(2 * math.pi / self.T * (t - self.t1))
        return 0.5 * self.A * (s + abs(s))

    def __str__(self):
        return "Sygnał sinusoidalny wyprostowany jednopołówkowo"


# Sygnał sinusoidalny wyprostowany dwupołówkowo
class S5(ContinuousSignal):
    def __init__(self, A, T, t1, d, f=2000):
        super().__init__(A, t1, d, f)
        self.T = T

    def value(self, t):
        return self.A * abs(math.sin(2 * math.pi / self.T * (t - self.t1)))

    def __str__(self):
        return "Sygnał sinusoidalny wyprostowany dwupołówkowo"


# Sygnał prostokątny
class S6(ContinuousSignal):
    def __init__(self, A, T, t1, d, kw, f=2000):
        super().__init__(A, t1, d, f)
        self.T = T
        self.kw = kw

    def value(self, t):
        k = math.floor((t - self.t1) / self.T)
        if t >= k * self.T + self.t1 and t < self.kw * self.T + k * self.T + self.t1:
            return self.A
        # drugi if się i tak nie wykonywał jeśli pierwszy true, sprawdzić wzór
        return 0

    def __str__(self):
        return "Sygnał prostokątny"


# Sygnał prostokątny symetryczny
class S7(ContinuousSignal):
    def __init__(self, A, T, t1, d, kw, f=2000):
        super().__init__(A, t1, d, f)
        self.T = T
        self.kw = kw

    def value(self, t):
        k = math.floor((t - self.t1) / self.T)
        if t >= k * self.T + self.t1 and t < self.kw * self.T + k * self.T + self.t1:
            return self.A
        # tak samo tutaj
        return -self.A

    def __str__(self):
        return "Sygnał prostokątny symetryczny"


# Sygnał trójkątny
class S8(ContinuousSignal):
    def __init__(self, A, T, t1, d, kw, f=2000):
        super().__init__(A, t1, d, f)
        self.T = T
        self.kw = kw

    def value(self, t):
        k = math.floor((t - self.t1) / self.T)
        if t >= k * self.T + self.t1 and t < self.kw * self.T + k * self.T + self.t1:
            return self.A / (self.kw * self.T) * (t - k * self.T - self.t1)
        # i tu
        return -self.A / (self.T * (1 - self.kw)) * (t - k * self.T - self.t1) + self.A / (1 - self.kw)

    def __str__(self):
        return "Sygnał trójkątny"


# Skok jednostkowy
class S9(ContinuousSignal):
    def __init__(self, A, t1, d, ts, f=2000):
        super().__init__(A, t1, d, f)
        self.ts = ts

    def value(self, t):
        if t > self.ts:
            return self.A
        elif t == self.ts:
            return 0.5 * self.A
        else:
            return 0

    def __str__(self):
        return "Skok jednostkowy"


# SYGNAŁY DYSKRETNE

# Impuls jednostkowy
class S10(DiscreteSignal):
    def __init__(self, A, ns, n1, l, f=10):
        super().__init__(A, n1, l, f)
        self.ns = ns

    def value(self, n):
        if n == self.ns:
            return self.A
        else:
            return 0

    def __str__(self):
        return "Impuls jednostkowy"


# Szum impulsowy
class S11(DiscreteSignal):
    def __init__(self, A, p, n1, l, f=10):
        super().__init__(A, n1, l, f)
        self.p = p

    def value(self, n):
        if random.random() < self.p:
            return self.A
        else:
            return 0

    def __str__(self):
        return "Szum impulsowy"

if __name__ == "__main__":
    szumik = S2(35, 0, 10)
    sinusik = S5(10, 3, 0, 10)
    skoczek = S9(10, -10, 20, 0)

    szumik.plot()
    szumik.histogram(40)

    suma = szumik + sinusik
    roznica = sinusik - skoczek
    iloczyn = sinusik * skoczek

    # TODO: ogarnąć przedtem dzielenie przez 0
    iloraz = None

    suma.plot()

    suma.save_bin("suma.bin")
    wczytany = Signal.load("suma.bin")
    wczytany.plot()

    # dyskretny
    szumik2 = S11(5, 0.2, -5, 40)
    szumik2.save_bin("s11.bin")
    w = Signal.load("s11.bin")
    w.plot()