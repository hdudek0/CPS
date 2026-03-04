import math
import random
import pickle
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

# TODO: zoptymalizować wzory niektórych sygnałów
# TODO: pozabezpieczać przed dzieleniem przez 0

# pythanie: dlaczego częstotliwosc probkowania nie jest parametrem sygnału w sygnałach ciągłych?
# nie jest używana do zdefiniowania sygnału, to jest oddzielny mechanizm który musimy zastosować
# żeby reprezentować go w postaci zbioru punktów zamiast funkcji, nie zmienia to sygnału,
# tylko rozdzielczość tej reprezentacji. tak mi się wydaje, można pytać tak czy siak

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
        plt.hist(Y, bins=bins)
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

    # TODO, Stasiak mówił jak to ma wyglądać ale nie pamiętam :/
    def save_txt(self, path):
        pass

    # TODO: DRY (_operation z lambdą?)
    def __add__(self, other):
        X1, Y1 = self.samples()
        X2, Y2 = other.samples()
        n = min(len(Y1), len(Y2))
        X = X1[:n]
        Y = [Y1[i] + Y2[i] for i in range(n)]
        is_discrete = getattr(self, "is_discrete", False) or getattr(other, "is_discrete", False)
        return SampledSignal(X, Y, is_discrete=is_discrete, name=f"({self})+({other})")

    def __sub__(self, other):
        X1, Y1 = self.samples()
        X2, Y2 = other.samples()
        n = min(len(Y1), len(Y2))
        X = X1[:n]
        Y = [Y1[i] - Y2[i] for i in range(n)]
        is_discrete = getattr(self, "is_discrete", False) or getattr(other, "is_discrete", False)
        return SampledSignal(X, Y, is_discrete=is_discrete, name=f"({self})-({other})")

    def __mul__(self, other):
        X1, Y1 = self.samples()
        X2, Y2 = other.samples()
        n = min(len(Y1), len(Y2))
        X = X1[:n]
        Y = [Y1[i] * Y2[i] for i in range(n)]
        is_discrete = getattr(self, "is_discrete", False) or getattr(other, "is_discrete", False)
        return SampledSignal(X, Y, is_discrete=is_discrete, name=f"({self})*({other})")

    def __truediv__(self, other):
        X1, Y1 = self.samples()
        X2, Y2 = other.samples()
        n = min(len(Y1), len(Y2))
        X = X1[:n]
        Y = [Y1[i] / Y2[i] for i in range(n)]
        is_discrete = getattr(self, "is_discrete", False) or getattr(other, "is_discrete", False)
        return SampledSignal(X, Y, is_discrete=is_discrete, name=f"({self})/({other})")


class SampledSignal(Signal):
    def __init__(self, X, Y, is_discrete=False, name="SampledSignal"):
        self.X = X
        self.Y = Y
        self.is_discrete = is_discrete
        self.name = name

    # TODO
    def value(self, x):
        return None

    def samples(self):
        return self.X, self.Y

    def __str__(self):
        return self.name
    

class ContinuousSignal(Signal):
    is_discrete = False

    def __init__(self, A, t1, d, f=2000):
        self.A = A
        self.t1 = t1
        self.d = d
        self.f = f

    def samples(self):
        X, Y = [], []
        n = self.f * self.d
        for i in range(n):
            t = self.t1 + i / self.f
            X.append(t)
            Y.append(self.value(t))
        return X, Y


class DiscreteSignal(Signal):
    is_discrete = True

    def __init__(self, A, n1, d, f=10):
        self.A = A
        self.n1 = n1
        self.d = d
        self.f = f

    def samples(self):
        X, Y = [], []
        for i in range(self.d):
            n = self.n1 + i
            t = n / self.f
            X.append(t)
            Y.append(self.value(n))
        return X, Y

# SYGNAŁY CIĄGŁE

# Szum o rozkładzie jednostajnym
class S1(ContinuousSignal):
    def value(self, t):
        return random.uniform(-self.A, self.A)

    def __str__(self):
        return "S1 Szum jednostajny"


# Szum gaussowski
class S2(ContinuousSignal):
    # TODO: poprawic żeby był rozkład normalny
    def value(self, t):
        return min(self.A, max(-self.A, random.gauss()))  #domyślnie mean=0, std=1

    def __str__(self):
        return "S2 Szum gaussowski"


# Sygnał sinusoidalny
class S3(ContinuousSignal):
    def __init__(self, A, T, t1, d, f=2000):
        super().__init__(A, t1, d, f)
        self.T = T

    def value(self, t):
        return self.A * math.sin(2 * math.pi / self.T * (t - self.t1))

    def __str__(self):
        return "S3 Sinusoida"


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
        return "S4 Sinusoida jednopołówkowo"


# Sygnał sinusoidalny wyprostowany dwupołówkowo
class S5(ContinuousSignal):
    def __init__(self, A, T, t1, d, f=2000):
        super().__init__(A, t1, d, f)
        self.T = T

    def value(self, t):
        return self.A * abs(math.sin(2 * math.pi / self.T * (t - self.t1)))

    def __str__(self):
        return "S5 Sinusoida dwupołówkowo"


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
        return "S6 Prostokątny"


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
        return "S7 Prostokątny symetryczny"


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
        return "S8 Trójkątny"


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
        return "S9 Skok jednostkowy"


# SYGNAŁY DYSKRETNE

# wydaje mi się że parametr l to miało być d
# d - długość l - length wygląda jak pomyłka

# Impuls jednostkowy
class S10(DiscreteSignal):
    def __init__(self, A, ns, n1, d, f=10):
        super().__init__(A, n1, d, f)
        self.ns = ns

    def value(self, n):
        if n == self.ns:
            return self.A
        else:
            return 0

    def __str__(self):
        return "S10 Impuls jednostkowy"


# Szum impulsowy
class S11(DiscreteSignal):
    def __init__(self, A, p, n1, d, f=10):
        super().__init__(A, n1, d, f)
        self.p = p

    def value(self, n):
        if random.random() < self.p:
            return self.A
        else:
            return 0

    def __str__(self):
        return "S11 Szum impulsowy"

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
    wczytany = Signal.load_bin("suma.bin")
    wczytany.plot()

    # dyskretny
    szumik2 = S11(5, 0.2, 0, 100, 10)
    szumik2.save_bin("s11.bin")
    w = Signal.load_bin("s11.bin")
    w.plot()