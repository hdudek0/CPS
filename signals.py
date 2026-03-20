import math
import random
import pickle
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

# TODO: zmienić x, y na A i t -> jeśli ci się chce
# TODO: dopasować domyślne parametry (defaults) w GUI -> też opcjonalnie one są jakby ok

class Signal(ABC):
    @abstractmethod
    def value(self, x):
        pass

    @abstractmethod
    def samples(self):
        pass

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return SampledSignal(data["X"], data["Y"],
                             f"Wczytany({data.get('name', '?')})",
                             data["fs"], data["n1"], data["l"])
    
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
    

class SampledSignal(Signal):
    def __init__(self, X, Y, name, fs, n1, l):
        self.X = list(X)
        self.Y = list(Y)
        self.name = name
        self.fs = fs
        self.n1 = n1
        self.l = l

    def value(self, x):
        return None

    def samples(self):
        return self.X, self.Y

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
            "l": self.l
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
            
    def save_txt(self, path):
        X, Y = self.samples()
        with open(path, "w") as f:
            f.write(f"name: {str(self)}\n")
            for x, y in zip(X, Y):
                f.write(f"{x}\t{y}\n")

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
        return SampledSignal(X1, Y, f"({self}){symbol}({other})", self.fs, self.n1, self.l)

    def __add__(self, other):
        return self._operation(other, lambda a, b: a + b, "+")

    def __sub__(self, other):
        return self._operation(other, lambda a, b: a - b, "-")

    def __mul__(self, other):
        return self._operation(other, lambda a, b: a * b, "*")
    
    def __truediv__(self, other):
        def safe_div(a, b):
            return a / b if b != 0 else math.nan
        return self._operation(other, safe_div, "/")
        

class ContinuousSignal(Signal):
    def __init__(self, A, t1, fs, d):
        self.A = A
        self.t1 = t1
        self.d = d
        self.fs = fs
    
    def samples(self):
        X, Y = [], []
        # jeśli d * fs nie całkowite to liczba próbek zaokrąglona w dół do jedności
        n = int(self.fs * self.d)
        for i in range(n):
            t = self.t1 + i / self.fs
            X.append(t)
            Y.append(self.value(t))
        return X, Y
    
    def _t_in_domain(self, t):
        return t >= self.t1 and t <= self.d + self.t1


class DiscreteSignal(Signal):
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