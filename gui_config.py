from signals_funcs import S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11
from signals_sampled import SampledSignal

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

PARAM_DEFAULTS = {
    "A": "5", "T[s]": "1", "t1[s]": "0", "d[s]": "5", "kw": "0.5",
    "ts[s]": "1", "n1": "0", "l": "10", "fs[Hz]": "1000", "ns": "5",
    "p": "0.5", "fs": "1000",
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
    if "l" in params:
        return params["l"]
    return 0
