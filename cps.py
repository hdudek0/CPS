import math
import matplotlib.pyplot as plt
import random

# TODO: zoptymalizować wzory niektórych sygnałów
# TODO: pozabezpieczać przed dzieleniem przez 0

# pythanie: dlaczego częstotliwosc probkowania nie jest parametrem sygnału w sygnałach ciągłych?

# SYGNAŁY CIĄGŁE

# Szum o rozkładzie jednostajnym
class S1():
    def __init__(self, A, t1, d):
        self.A = A
        self.t1 = t1
        self.d = d
    
    def function(self, t):
        return random.uniform(-self.A, self.A)


# Szum gaussowski
class S2():
    def __init__(self, A, t1, d):
        self.A = A
        self.t1 = t1
        self.d = d
    
    # TODO: poprawic żeby był rozkład normalny
    def function(self, t):
        return min(self.A, max(-self.A, random.gauss()))   #domyślnie mean=0, std=1
    

# Sygnał sinusoidalny
class S3():
    def __init__(self, A, T, t1, d):
        self.A = A
        self.T = T
        self.t1 = t1
        self.d = d
    
    def function(self, t):
        return self.A * math.sin(2 * math.pi / self.T * (t - self.t1))
    

# Sygnał sinusoidalny wyprostowany jednopołówkowo
class S4():
    def __init__(self, A, T, t1, d):
        self.A = A
        self.T = T
        self.t1 = t1
        self.d = d
    
    def function(self, t):
        return 0.5 * self.A * (math.sin(2 * math.pi / self.T * (t - self.t1)) + abs(math.sin(2 * math.pi / self.T * (t - self.t1))))
    

# Sygnał sinusoidalny wyprostowany dwupołówkowo
class S5():
    def __init__(self, A, T, t1, d):
        self.A = A
        self.T = T
        self.t1 = t1
        self.d = d
    
    def function(self, t):
        return self.A * abs(math.sin(2 * math.pi / self.T * (t - self.t1)))


# Sygnał prostokątny
class S6():
    def __init__(self, A, T, t1, d, kw):
        self.A = A
        self.T = T
        self.t1 = t1
        self.d = d
        self.kw = kw
    
    def function(self, t):
        k = math.floor((t - self.t1) / self.T)
        if t >= k * self.T + self.t1 and t < self.kw * self.T + k * self.T + self.t1:
            return self.A
        if t >= self.kw * self.T - k * self.T + self.t1 and t < self.T + k * self.T + self.t1:
            return 0


# Sygnał prostokątny symetryczny
class S7():
    def __init__(self, A, T, t1, d, kw):
        self.A = A
        self.T = T
        self.t1 = t1
        self.d = d
        self.kw = kw
    
    def function(self, t):
        k = math.floor((t - self.t1) / self.T)
        if t >= k * self.T + self.t1 and t < self.kw * self.T + k * self.T + self.t1:
            return self.A
        if t >= self.kw * self.T + self.t1 + k * self.T and t < self.T + k * self.T + self.t1:
            return -self.A
    

# Sygnał trójkątny
class S8():
    def __init__(self, A, T, t1, d, kw):
        self.A = A
        self.T = T
        self.t1 = t1
        self.d = d
        self.kw = kw

    def __str__(self):
        return "Sygnał trójkątny"
    
    def function(self, t):
        k = math.floor((t - self.t1) / self.T)
        if t >= k * self.T + self.t1 and t < self.kw * self.T + k * self.T + self.t1:
            return self.A / (self.kw * self.T) * (t - k * self.T - self.t1)
        if t >= self.kw * self.T + self.t1 + k * self.T and t < self.T + k * self.T + self.t1:
            return -self.A / (self.T * (1 - self.kw)) * (t - k * self.T - self.t1) + self.A / (1 - self.kw)


# Skok jednostkowy
class S9():
    def __init__(self, A, t1, d, ts):
        self.A = A
        self.t1 = t1
        self.d = d
        self.ts = ts
    
    def function(self, t):
        if t > self.ts:
            return self.A
        elif t == self.ts:
            return 0.5 * self.A
        else:
            return 0


def make_plot(signal):
    tabA = []
    tabt = []
    f = 100000
    number_of_samples = f * signal.d
    for s in range(number_of_samples):
        t = signal.t1 + s/f
        tabt.append(t)
        tabA.append(signal.function(t))
    plt.axhline(0, color='darkgray', linewidth=1, linestyle="--")
    plt.axvline(0, color='darkgray', linewidth=1, linestyle="--")
    plt.plot(tabt, tabA, color="darkgreen")
    # TODO: title w zależności od sygnału (wstępny pomysł z __str__ ale tego chyba się nie używa do tego bo raczej ten str powinien byc rozny dla roznych instancji tej samej klasy)
    plt.title(str(signal), fontsize=14, fontweight="bold")
    plt.xlabel("t[s]", fontsize=12, fontweight="bold")
    plt.ylabel("A", rotation=0, fontsize=12, fontweight="bold")
    plt.show()


szumik = S2(35, 0, 10)
sinusik = S5(10, 3, 0, 10)
skoczek = S9(10,-10, 20, 0)
prostokacik = S7(10, 2, 0, 10, 0.5)
trojkacik = S8(10, 2, 0, 10, 0.5)
#make_plot(trojkacik)

# SYGNAŁY DYSKRETNE

# Impuls jednostkowy
class S10():
    def __init__(self, A, ns, n1, l, f):
        self.A = A
        self.ns = ns
        self.n1 = n1
        self.l = l
        self.f = f
    
    def function(self, n):
        if n == self.ns:
            return self.A
        else:
            return 0
        

# Szum impulsowy
class S11():
    def __init__(self, A, t1, d, f, p):
        self.A = A
        self.t1 = t1
        self.d = d
        self.f = f
        self.p = p
    
    def function(self, t):
        return

# czy ja dobrze rozumiem n1 jako analogie t1??
def make_plot2(signal):
    tabA = []
    tabt = []
    for i in range(signal.l):
        n = signal.n1 + i
        t = n / signal.f
        tabt.append(t)
        tabA.append(signal.function(n))
    plt.axhline(0, color='darkgray', linewidth=1, linestyle="--")
    plt.axvline(0, color='darkgray', linewidth=1, linestyle="--")
    plt.scatter(tabt, tabA, color="darkgreen")
    # TODO: title w zależności od sygnału (wstępny pomysł z __str__ ale tego chyba się nie używa do tego bo raczej ten str powinien byc rozny dla roznych instancji tej samej klasy)
    plt.title(str(signal), fontsize=14, fontweight="bold")
    plt.xlabel("t[s]", fontsize=12, fontweight="bold")
    plt.ylabel("A", rotation=0, fontsize=12, fontweight="bold")
    plt.show()

impulsik = S10(1, 0, -5, 11, 100)
make_plot2(impulsik)