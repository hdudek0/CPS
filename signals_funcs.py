import math
import random
from signals_types import ContinuousSignal, DiscreteSignal


# SYGNAŁY CIĄGŁE

# Szum o rozkładzie jednostajnym
class S1(ContinuousSignal):
    def value(self, t):
        if self._t_in_domain(t):
            return random.uniform(-self.A, self.A)
        return 0

    def __str__(self):
        return "Szum o rozkładzie jednostajnym"


# Szum gaussowski
class S2(ContinuousSignal):
    def value(self, t):
        if self._t_in_domain(t):
            return random.gauss(0, self.A)
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
            local = max(0, t - k * self.T - self.t1)
            return self.A if local < self.kw * self.T else 0
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
            local = max(0, t - k * self.T - self.t1)
            return self.A if local < self.kw * self.T else -self.A
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
            local = max(0, t - k * self.T - self.t1)
            if local < self.kw * self.T:
                return self.A / (self.kw * self.T) * local
            return -self.A / (self.T * (1 - self.kw)) * local + self.A / (1 - self.kw)
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
            return self.A if n == self.ns else 0
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
            return self.A if random.random() < self.p else 0
        return 0

    def __str__(self):
        return "Szum impulsowy"
