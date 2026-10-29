import math
import random
import struct
import unicodedata

SR = 44100

_F0MUL = 1.0
_FS = 1.0

VOICES = {
    "male": {"f0": 1.0, "fs": 1.0},
    "girl": {"f0": 1.65, "fs": 1.2},
    "female": {"f0": 1.65, "fs": 1.2},
    "woman": {"f0": 1.65, "fs": 1.2},
}


def _norm_voice(voice):
    v = (voice or "male").strip().lower()
    if v in ("ж", "жен", "женский", "девочка", "девушка", "fem"):
        v = "female"
    elif v in ("м", "муж", "мужской", "мужчина", "папа"):
        v = "male"
    return v if v in VOICES else "male"


class Resonator:
    __slots__ = ("a1", "a2", "g", "y1", "y2")

    def __init__(self, sr, freq, bw):
        r = math.exp(-math.pi * bw / sr)
        arg = 2.0 * math.pi * freq / sr
        self.a1 = 2.0 * r * math.cos(arg)
        self.a2 = -r * r
        self.g = 1.0 - r
        self.y1 = 0.0
        self.y2 = 0.0

    def run(self, samples):
        a1 = self.a1
        a2 = self.a2
        g = self.g
        y1 = self.y1
        y2 = self.y2
        n = len(samples)
        out = [0.0] * n
        for i in range(n):
            y = g * samples[i] + a1 * y1 + a2 * y2
            y2 = y1
            y1 = y
            out[i] = y
        self.y1 = y1
        self.y2 = y2
        return out


class ZeroFilter:
    __slots__ = ("a", "x1", "x2")

    def __init__(self, sr, freq):
        self.a = 2.0 * math.cos(2.0 * math.pi * freq / sr)
        self.x1 = 0.0
        self.x2 = 0.0

    def run(self, samples):
        a = self.a
        x1 = self.x1
        x2 = self.x2
        n = len(samples)
        out = [0.0] * n
        for i in range(n):
            s = samples[i]
            out[i] = s - a * x1 + x2
            x2 = x1
            x1 = s
        self.x1 = x1
        self.x2 = x2
        return out


def lowpass1(samples, sr, fc):
    k = 1.0 - math.exp(-2.0 * math.pi * fc / sr)
    out = [0.0] * len(samples)
    s = 0.0
    for i in range(len(samples)):
        s += k * (samples[i] - s)
        out[i] = s
    return out


def highpass1(samples, sr, fc):
    lp = lowpass1(samples, sr, fc)
    return [x - y for x, y in zip(samples, lp)]


def envelope(n, attack, release):
    if n <= 0:
        return []
    attack = max(0, min(n, int(attack)))
    release = max(0, min(n - attack, int(release)))
    out = [0.0] * n
    for i in range(n):
        if attack > 0 and i < attack:
            out[i] = 0.5 - 0.5 * math.cos(math.pi * i / attack)
        elif release > 0 and i >= n - release:
            rel_i = i - (n - release)
            out[i] = 0.5 + 0.5 * math.cos(math.pi * rel_i / release)
        else:
            out[i] = 1.0
    return out


def white_noise(n):
    return [random.uniform(-1.0, 1.0) for _ in range(n)]


def _scale_rms(sig, target):
    n = len(sig)
    if n == 0:
        return sig
    r = math.sqrt(sum(s * s for s in sig) / n)
    if r < 1e-9:
        return sig
    k = target / r
    return [s * k for s in sig]


def _peaking(samples, sr, freq, q, db):
    a = 10.0 ** (db / 40.0)
    w0 = 2.0 * math.pi * freq / sr
    alpha = math.sin(w0) * math.sinh(math.log(2.0) / 2.0 * q * w0 / math.sin(w0))
    b0 = 1.0 + alpha * a
    b1 = -2.0 * math.cos(w0)
    b2 = 1.0 - alpha * a
    a0 = 1.0 + alpha / a
    a1 = -2.0 * math.cos(w0)
    a2 = 1.0 - alpha / a
    inv = 1.0 / a0
    x1 = x2 = y1 = y2 = 0.0
    out = []
    for s in samples:
        y = b0 * inv * s + b1 * inv * x1 + b2 * inv * x2 - a1 * inv * y1 - a2 * inv * y2
        x2 = x1
        x1 = s
        y2 = y1
        y1 = y
        out.append(y)
    return out


def _presence(samples, sr):
    boost = _peaking(samples, sr, 2600.0, 0.9, 6.0)
    return [x + (b - x) * 0.4 for x, b in zip(samples, boost)]


def _dc_block(samples, sr):
    k = 1.0 - math.exp(-2.0 * math.pi * 30.0 / sr)
    out = []
    s = 0.0
    for x in samples:
        s += k * (x - s)
        out.append(x - s)
    return out


def glottal(f0s, sr, phase0=0.0, q=0.62, breath=0.02, jitter=0.005):
    out = []
    phase = phase0
    for f0 in f0s:
        f0 = f0 * (1.0 + jitter * (random.random() * 2.0 - 1.0))
        p = phase - math.floor(phase)
        if p < q:
            out.append(0.5 * (1.0 - math.cos(math.pi * p / q)))
        else:
            out.append(0.0)
        phase += f0 / sr
    if breath > 0.0:
        for i in range(len(out)):
            out[i] += (random.random() * 2.0 - 1.0) * breath * 0.03
    return phase, out


def _vow(F, dur, amp, B=(75, 90, 120)):
    return {"type": "vowel", "formants": list(F), "bws": list(B), "dur": dur, "amp": amp}


def _cons(kind, dur, amp=0.5, **kw):
    d = {"type": kind, "dur": dur, "amp": amp}
    d.update(kw)
    return d


PHONES = {
    "а": _vow((750, 1240, 2600), 0.17, 0.95),
    "о": _vow((500, 850, 2250), 0.15, 0.9),
    "у": _vow((320, 700, 2250), 0.16, 0.9),
    "ы": _vow((390, 1450, 2200), 0.15, 0.9),
    "э": _vow((500, 1700, 2600), 0.15, 0.9),
    "е": _vow((450, 1850, 2450), 0.16, 0.95),
    "и": _vow((300, 2180, 2950), 0.14, 0.9),
    "я": _vow((700, 1330, 2450), 0.17, 0.95),
    "ё": _vow((460, 1100, 2350), 0.15, 0.9),
    "ю": _vow((330, 820, 2300), 0.15, 0.9),
    "п": _cons("plosive", 0.070, 0.5, closure=0.045, burst=0.012, bfilter="п"),
    "т": _cons("plosive", 0.070, 0.5, closure=0.040, burst=0.012, bfilter="т"),
    "к": _cons("plosive", 0.080, 0.55, closure=0.050, burst=0.016, bfilter="к"),
    "б": _cons("plosive", 0.080, 0.5, closure=0.050, burst=0.010, bfilter="п", voiced=True),
    "д": _cons("plosive", 0.080, 0.5, closure=0.050, burst=0.010, bfilter="т", voiced=True),
    "г": _cons("plosive", 0.090, 0.55, closure=0.055, burst=0.012, bfilter="к", voiced=True),
    "ф": _cons("fric", 0.070, 0.4, fric=[(900, 300)]),
    "с": _cons("fric", 0.100, 0.5, fric=[(5600, 800), (7500, 1400)]),
    "ш": _cons("fric", 0.130, 0.5, fric=[(2100, 550), (2600, 800)]),
    "х": _cons("fric", 0.090, 0.32, fric=[(1200, 500), (2200, 800)]),
    "щ": _cons("fric", 0.160, 0.55, fric=[(2400, 600), (3200, 800)]),
    "з": _cons("vfric", 0.080, 0.4, fric=[(5400, 800), (7200, 1400)], formants=(330, 1400, 2300)),
    "ж": _cons("vfric", 0.120, 0.4, fric=[(2100, 600), (2600, 800)], formants=(250, 1300, 2200)),
    "в": _cons("vfric", 0.070, 0.4, fric=[(1200, 400)], formants=(300, 1100, 2100)),
    "ц": _cons("affricate", 0.140, 0.5, bfilter="т", fric=[(5500, 900)]),
    "ч": _cons("affricate", 0.110, 0.5, bfilter="т", fric=[(2400, 700), (3000, 1000)]),
    "м": _cons("nasal", 0.070, 0.5, formants=(250, 1150, 2150), bws=(70, 180, 250), zero=800),
    "н": _cons("nasal", 0.070, 0.5, formants=(250, 1400, 2300), bws=(70, 180, 250), zero=850),
    "л": _cons("liquid", 0.080, 0.55, formants=(350, 1200, 2500), bws=(60, 100, 140)),
    "р": _cons("trill", 0.110, 0.6, formants=(490, 1450, 2400), bws=(70, 100, 140)),
    "й": _cons("glide", 0.060, 0.5, formants=(300, 2100, 2900)),
    " ": _cons("pause", 0.120),
}

VOWELS = set("аоуыэеияёю")

IOTATED = {"я": "а", "ё": "о", "е": "э", "ю": "у", "є": "э"}

SOFTABLE = set("пткбдвфсзмнлр")

LETTER_MAP = {
    "а": "а", "б": "б", "в": "в", "г": "г", "д": "д", "е": "е", "ё": "ё",
    "ж": "ж", "з": "з", "и": "и", "й": "й", "к": "к", "л": "л", "м": "м",
    "н": "н", "о": "о", "п": "п", "р": "р", "с": "с", "т": "т", "у": "у",
    "ф": "ф", "х": "х", "ц": "ц", "ч": "ч", "ш": "ш", "щ": "щ", "ы": "ы",
    "э": "э", "ю": "ю", "я": "я",
    "і": "и", "ї": "и", "є": "е", "ґ": "г", "ў": "у",
    "α": "а", "β": "б", "γ": "г", "δ": "д", "ε": "э", "ζ": "з", "η": "и",
    "θ": "с", "ι": "и", "κ": "к", "λ": "л", "μ": "м", "ν": "н", "ξ": "с",
    "ο": "о", "π": "п", "ρ": "р", "σ": "с", "ς": "с", "τ": "т", "υ": "у",
    "φ": "ф", "χ": "х", "ψ": "п", "ω": "о",
}

WORDS = {
    "привет": "п'|р'|и|в'|е|т'",
    "как": "к|а|к",
    "у": "у",
    "тебя": "т'|и|б'|я",
    "дела": "д'|и|л'|а",
}

STRESS = {"привет": 4, "как": 1, "у": 0, "тебя": 3, "дела": 3}


class Token:
    __slots__ = ("phone", "stressed", "soft", "type", "dur", "start", "emph")

    def __init__(self, phone, stressed, soft):
        self.phone = phone
        self.stressed = stressed
        self.soft = soft
        self.emph = 0.0
        info = PHONES[phone]
        self.type = info["type"]
        dur = info["dur"]
        if info["type"] == "vowel":
            dur = dur * (1.15 if stressed else 0.82)
        self.dur = dur


LETTER_NAMES = {
    "а": "а", "б": "бэ", "в": "вэ", "г": "гэ", "д": "дэ", "е": "е", "ё": "ё",
    "ж": "жэ", "з": "зэ", "и": "и", "й": "й", "к": "ка", "л": "эль", "м": "эм",
    "н": "эн", "о": "о", "п": "пэ", "р": "эр", "с": "эс", "т": "тэ", "у": "у",
    "ф": "эф", "х": "ха", "ц": "цэ", "ч": "че", "ш": "ша", "щ": "ща", "ъ": "твёрдый знак",
    "ы": "ы", "ь": "мягкий знак", "э": "э", "ю": "ю", "я": "я",
    "і": "и", "ї": "и", "є": "е", "ґ": "гэ", "ў": "у",
}

EN_DIGRAPHS = sorted([
    ("tion", ["ш", "и", "н"]),
    ("ture", ["ч", "э", "р"]),
    ("sh", ["ш"]),
    ("ch", ["ч"]),
    ("th", ["с"]),
    ("ph", ["ф"]),
    ("ck", ["к"]),
    ("wh", ["в"]),
    ("qu", ["к", "в"]),
    ("ee", ["и"]),
    ("ea", ["и"]),
    ("oo", ["у"]),
    ("ou", ["а", "у"]),
    ("ow", ["о"]),
    ("oi", ["о", "й"]),
    ("oy", ["о", "й"]),
    ("ai", ["э", "й"]),
    ("ay", ["э", "й"]),
    ("au", ["о"]),
    ("aw", ["о"]),
    ("er", ["э", "р"]),
    ("ir", ["э", "р"]),
    ("ur", ["э", "р"]),
    ("ar", ["а", "р"]),
    ("or", ["о", "р"]),
    ("ng", ["н", "г"]),
    ("wr", ["р"]),
    ("kn", ["н"]),
    ("gn", ["н"]),
    ("mb", ["м"]),
    ("mn", ["м"]),
    ("gh", []),
], key=lambda e: -len(e[0]))

EN_SINGLE = {
    "a": ["э"], "b": ["б"], "c": ["к"], "d": ["д"], "e": ["э"], "f": ["ф"],
    "g": ["г"], "h": ["х"], "i": ["и"], "j": ["д", "ж"], "k": ["к"],
    "l": ["л"], "m": ["м"], "n": ["н"], "o": ["о"], "p": ["п"], "q": ["к"],
    "r": ["р"], "s": ["с"], "t": ["т"], "u": ["у"], "v": ["в"], "w": ["в"],
    "x": ["к", "с"], "y": ["и"], "z": ["з"],
}

UNITS = ["ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь",
         "восемь", "девять", "десять", "одиннадцать", "двенадцать",
         "тринадцать", "четырнадцать", "пятнадцать", "шестнадцать",
         "семнадцать", "восемнадцать", "девятнадцать"]
TENS = ["", "десять", "двадцать", "тридцать", "сорок", "пятьдесят",
        "шестьдесят", "семьдесят", "восемьдесят", "девяносто"]
HUNDREDS = ["", "сто", "двести", "триста", "четыреста", "пятьсот",
            "шестьсот", "семьсот", "восемьсот", "девятьсот"]
DIGIT_WORDS = ["ноль", "один", "два", "три", "четыре", "пять", "шесть",
               "семь", "восемь", "девять"]


def russian_g2p(word):
    letters = [c for c in word if c in LETTER_MAP or c in "ьъ"]
    vids = [i for i, c in enumerate(letters) if LETTER_MAP.get(c, c) in VOWELS]
    if len(vids) > 1:
        si = vids[len(vids) - 2]
    elif vids:
        si = vids[0]
    else:
        si = -1
    raw = []
    i = 0
    while i < len(letters):
        ch = letters[i]
        nxt = letters[i + 1] if i + 1 < len(letters) else ""
        if ch == "ь":
            if raw and raw[-1][0] in SOFTABLE and not raw[-1][2]:
                raw[-1] = (raw[-1][0], raw[-1][1], True)
            i += 1
            continue
        if ch == "ъ":
            if nxt and nxt in "яёюе":
                raw.append(("й", False, True))
                raw.append(({"я": "а", "ё": "о", "ю": "у", "е": "э"}[nxt],
                            i + 1 == si, False))
                i += 2
            else:
                i += 1
            continue
        if ch == "ї":
            raw.append(("й", False, True))
            raw.append(("и", i == si, False))
            i += 1
            continue
        if ch in IOTATED:
            prev = letters[i - 1] if i > 0 else ""
            prevbase = LETTER_MAP.get(prev, prev)
            if i == 0 or prevbase in VOWELS or prev in "ьъ":
                raw.append(("й", False, True))
                raw.append((IOTATED[ch], i == si, False))
                i += 1
                continue
        base = LETTER_MAP.get(ch)
        if base is None:
            i += 1
            continue
        soft = False
        if ch in SOFTABLE and nxt and nxt in "яюеёиьієї":
            soft = True
        raw.append((base, i == si, soft))
        i += 1
    return [Token(ph, st, sf) for ph, st, sf in raw]


def _single_letter_phones(ch):
    if ch in "ьъ":
        return []
    base = LETTER_MAP.get(ch, ch)
    if base in PHONES:
        return [base]
    return []


EN_EXCEPTIONS = {
    "the": ["з", "э"],
    "a": ["э"],
    "i": ["а", "й"],
}


def english_g2p(word):
    plain_ascii = all(ord(c) < 128 for c in word)
    word = "".join(c for c in unicodedata.normalize("NFKD", word)
                   if not unicodedata.combining(c))
    flat = []
    if word in EN_EXCEPTIONS:
        flat = list(EN_EXCEPTIONS[word])
    i = 0
    while word not in EN_EXCEPTIONS and i < len(word):
        hit = False
        for pat, phs in EN_DIGRAPHS:
            if word.startswith(pat, i):
                flat.extend(phs)
                i += len(pat)
                hit = True
                break
        if hit:
            continue
        ch = word[i]
        nxt = word[i + 1] if i + 1 < len(word) else ""
        if ch == "c" and nxt in "iey":
            phs = ["с"]
        elif ch == "g" and nxt in "iey":
            phs = ["д", "ж"]
        else:
            phs = list(EN_SINGLE[ch]) if ch in EN_SINGLE else _single_letter_phones(ch)
            if ch == "y" and not flat:
                phs = ["й"]
        flat.extend(phs)
        i += 1
    collapsed = []
    for p in flat:
        if collapsed and p == collapsed[-1] and p in SOFTABLE:
            continue
        collapsed.append(p)
    flat = collapsed
    if plain_ascii and word not in EN_EXCEPTIONS and len(word) > 1 \
            and word[-1] == "e" and flat and flat[-1] == "э":
        if word[-2] not in "aeiouy":
            flat = flat[:-1]
    vids = [i for i, p in enumerate(flat) if p in VOWELS]
    if len(vids) > 1:
        si = vids[-1]
    elif vids:
        si = vids[0]
    else:
        si = -1
    toks = []
    for i, p in enumerate(flat):
        if p in PHONES:
            toks.append(Token(p, i == si, False))
    return toks


def word_to_tokens(word, charMode=False):
    if word in WORDS:
        phs = WORDS[word].split("|")
        stress = STRESS.get(word, 0)
        toks = []
        for i, pho in enumerate(phs):
            base = pho
            soft = False
            if pho.endswith("'") and len(pho) > 1:
                base = pho[:-1]
                soft = True
            toks.append(Token(base, i == stress, soft))
        return toks
    if charMode and len(word) == 1 and word in LETTER_NAMES:
        word = LETTER_NAMES[word]
    if any("a" <= c <= "z" for c in word):
        return english_g2p(word)
    if any("\u0530" <= c <= "\u058F" for c in word):
        return armenian_g2p(word)
    return russian_g2p(word)


AM_MAP = {
    "ա": "а", "բ": "б", "գ": "г", "դ": "д", "ե": "э", "զ": "з", "է": "э",
    "ը": "ы", "թ": "т", "ժ": "ж", "ի": "и", "լ": "л", "խ": "х", "ծ": "ц",
    "կ": "к", "հ": "х", "ձ": "дз", "ղ": "г", "ճ": "ч", "մ": "м", "յ": "й",
    "ն": "н", "շ": "ш", "ո": "о", "չ": "ч", "պ": "п", "ջ": "дж", "ռ": "р",
    "ս": "с", "վ": "в", "տ": "т", "ր": "р", "ց": "ц", "փ": "п", "ք": "к",
    "օ": "о", "ֆ": "ф",
}

AM_UNITS = {1: "մեկ", 2: "երկու", 3: "երեք", 4: "չորս", 5: "հինգ", 6: "վեց",
            7: "յոթ", 8: "ութ", 9: "ինը"}
AM_TEENS = {10: "տասը", 11: "տասնմեկ", 12: "տասներկու", 13: "տասներեք",
            14: "տասնչորս", 15: "տասնհինգ", 16: "տասնվեց", 17: "տասնյոթ",
            18: "տասնութ", 19: "տասնինը"}
AM_TENS = {20: "քսան", 30: "երեսուն", 40: "քառասուն", 50: "հիսուն",
           60: "վաթսուն", 70: "յոթանասուն", 80: "ութսուն", 90: "իննսուն"}
AM_HUNDRED = "հարյուր"
AM_THOUSAND = "հազար"
AM_MILLION = "միլիոն"
AM_DIGIT_WORDS = ["զրո", "մեկ", "երկու", "երեք", "չորս", "հինգ", "վեց",
                  "յոթ", "ութ", "ինը"]


def _am_tens_units(v):
    if v < 10:
        return AM_UNITS[v]
    if v in AM_TEENS:
        return AM_TEENS[v]
    t = (v // 10) * 10
    u = v % 10
    if u == 0:
        return AM_TENS[t]
    return AM_TENS[t] + AM_UNITS[u]


def _am_under1000(v):
    parts = []
    h = v // 100
    rem = v % 100
    if h:
        if h == 1:
            parts.append(AM_HUNDRED)
        else:
            parts.append(AM_UNITS[h] + " " + AM_HUNDRED)
    if rem:
        parts.append(_am_tens_units(rem))
    return " ".join(parts)


def armenian_number_to_words(n):
    n = int(n)
    if n == 0:
        return "զրո"
    groups = []
    rank = 0
    while n > 0:
        groups.append((n % 1000, rank))
        n //= 1000
        rank += 1
    parts = []
    for g, rank in reversed(groups):
        if g == 0:
            continue
        w = _am_under1000(g)
        if rank == 1:
            w += " հազար"
        elif rank == 2:
            if g == 1:
                w = AM_MILLION
            else:
                w = _am_under1000(g) + " " + AM_MILLION
        parts.append(w)
    return " ".join(parts)


def armenian_g2p(word):
    flat = []
    i = 0
    while i < len(word):
        c = word[i]
        if c == "և":
            flat.append(["э", "в"])
            i += 1
            continue
        if c == "ո" and i + 1 < len(word) and word[i + 1] == "ւ":
            flat.append(["у"])
            i += 2
            continue
        if c in "եո" and i == 0:
            flat.append(["й", "э"] if c == "ե" else ["в", "о"])
            i += 1
            continue
        ph = AM_MAP.get(c)
        if ph is None:
            i += 1
            continue
        flat.append(list(ph) if len(ph) > 1 else [ph])
        i += 1
    phones = [p for seg in flat for p in seg]
    vids = [idx for idx, p in enumerate(phones) if p in VOWELS]
    si = vids[-1] if vids else -1
    toks = []
    for idx, p in enumerate(phones):
        if p in PHONES:
            toks.append(Token(p, idx == si, False))
    return toks


def number_to_russian(n):
    n = int(n)
    if n == 0:
        return "ноль"

    def num_under100(v, fem=False):
        if v < 20:
            if fem and v == 1:
                return "одна"
            if fem and v == 2:
                return "две"
            return UNITS[v]
        t = TENS[v // 10]
        u = v % 10
        if u == 0:
            return t
        uw = UNITS[u]
        if fem and u == 1:
            uw = "одна"
        elif fem and u == 2:
            uw = "две"
        return t + " " + uw

    def num_to1000(v, fem=False):
        parts = []
        h = v // 100
        if h:
            parts.append(HUNDREDS[h])
        rem = v % 100
        if rem:
            parts.append(num_under100(rem, fem))
        return " ".join(parts)

    def plural(v, one, few, many):
        x = v % 100
        if 11 <= x <= 14:
            return many
        x %= 10
        if x == 1:
            return one
        if x in (2, 3, 4):
            return few
        return many

    groups = []
    rank = 0
    while n > 0:
        groups.append((n % 1000, rank))
        n //= 1000
        rank += 1
    parts = []
    for g, rank in reversed(groups):
        if g == 0:
            continue
        words = num_to1000(g, fem=(rank == 1)).split()
        if rank == 1:
            words.append(plural(g, "тысяча", "тысячи", "тысяч"))
        elif rank == 2:
            words.append(plural(g, "миллион", "миллиона", "миллионов"))
        elif rank == 3:
            words.append(plural(g, "миллиард", "миллиарда", "миллиардов"))
        parts.extend(words)
    return " ".join(parts)


SYMBOL_NAMES = {
    "+": "плюс", "±": "плюс минус",
    "=": "равно", "*": "звёздочка", "/": "слэш", "\\": "обратный слэш",
    "%": "процент", "^": "крышка", "&": "амперсанд", "~": "тильда",
    "#": "решётка", "№": "номер", "@": "собака", "$": "доллар",
    "_": "нижнее подчеркивание", "<": "меньше", ">": "больше",
    "×": "умножить", "÷": "разделить",
    "€": "евро", "£": "фунт", "¥": "йена", "¢": "цент",
    "°": "градус", "²": "в квадрате", "³": "в кубе",
    "½": "одна вторая", "⅓": "одна треть", "⅔": "две трети",
    "¼": "одна четверть", "¾": "три четверти",
    "∞": "бесконечность", "≠": "не равно", "≤": "меньше или равно",
    "≥": "больше или равно", "≈": "приблизительно",
    "√": "корень", "∑": "сумма", "∏": "произведение",
    "→": "стрелка вправо", "←": "стрелка влево",
    "↑": "стрелка вверх", "↓": "стрелка вниз",
    "™": "торговая марка", "©": "копирайт", "®": "зарегистрированная марка",
    "§": "параграф", "¶": "параграф", "•": "точка", "·": "точка",
    "♀": "женский знак", "♂": "мужской знак",
    "π": "пи", "µ": "микро", "Ω": "ом",
}

_PAUSE_PUNCT = {
    ".": 0.30, "։": 0.30, "․": 0.30, "…": 0.34, "。": 0.30,
    ":": 0.22, ";": 0.22, "؛": 0.22, "：": 0.22, "；": 0.22,
    "-": 0.18, "–": 0.20, "—": 0.22, "֊": 0.18,
}

_SKIP_PUNCT = set("()[]{}«»„“”‘’\"'`")


EMOJI = {
    "😀": "улыбка", "😃": "улыбка", "😄": "смех", "😁": "улыбка",
    "😆": "смех", "😅": "смех", "🤣": "смех", "😂": "смех",
    "🙂": "улыбка", "🙃": "смайлик", "😉": "подмигивание", "😊": "улыбка",
    "😇": "ангел", "🥰": "влюблённость", "😍": "влюблённость", "🤩": "восторг",
    "😘": "поцелуй", "😗": "поцелуй", "😋": "вкусно", "😛": "язык",
    "😜": "шалость", "🤪": "сумасшествие", "🤗": "объятие", "🤭": "смешок",
    "🤫": "тише", "🤔": "задумчивость", "😐": "спокойствие", "😑": "спокойствие",
    "😶": "молчание", "😏": "хитрость", "😒": "недовольство", "😞": "грусть",
    "😔": "грусть", "😟": "беспокойство", "😕": "недоумение", "🙁": "грусть",
    "😣": "расстройство", "😖": "расстройство", "😫": "усталость", "😩": "усталость",
    "🥺": "грусть", "😢": "слёзы", "😭": "плач", "😤": "возмущение",
    "😠": "злость", "😡": "злость", "🤬": "ругань", "🤯": "шок",
    "😳": "смущение", "🥵": "жарко", "🥶": "холодно", "😱": "испуг",
    "😨": "испуг", "😰": "испуг", "😥": "грусть", "😓": "печаль",
    "🤒": "болезнь", "🤕": "травма", "🤢": "тошнота", "🤧": "чихание",
    "😷": "маска", "🤠": "ковбой", "🥳": "праздник", "🤤": "слюни",
    "😴": "сон", "😪": "сон",
    "❣": "сердечки", "💔": "разбитое сердце", "💕": "сердца", "💞": "сердца",
    "💓": "сердце", "💗": "сердце", "💖": "сердце", "💘": "сердце",
    "💝": "сердце", "❤": "сердце", "🧡": "сердце", "💛": "сердце",
    "💚": "сердце", "💙": "сердце", "💜": "сердце", "🖤": "сердце",
    "🤍": "сердце", "🤎": "сердце", "♥": "сердце",
    "👍": "одобрение", "👎": "неодобрение", "👌": "ок", "✌": "победа",
    "🤞": "удача", "✊": "кулак", "👊": "кулак", "🤛": "кулак",
    "🤜": "кулак", "🤝": "рукопожатие", "🙏": "благодарность", "💪": "сила",
    "🙌": "радость", "👏": "аплодисменты", "👐": "ладони", "🤲": "ладони",
    "👋": "приветствие", "🤚": "ладонь", "✋": "ладонь",
    "🐶": "собака", "🐱": "кошка", "🐭": "мышь", "🐹": "хомяк",
    "🐰": "кролик", "🦊": "лиса", "🐻": "медведь", "🐼": "панда",
    "🐨": "коала", "🐯": "тигр", "🦁": "лев", "🐮": "корова",
    "🐷": "свинья", "🐸": "лягушка", "🐵": "обезьяна", "🐔": "курица",
    "🐧": "пингвин", "🐦": "птица", "🦆": "утка", "🦉": "сова",
    "🐺": "волк", "🐗": "кабан", "🐴": "лошадь", "🦄": "единорог",
    "🐝": "пчела", "🦋": "бабочка", "🐌": "улитка", "🐞": "божья коровка",
    "🐢": "черепаха", "🐍": "змея", "🐙": "осьминог", "🦑": "кальмар",
    "🦀": "краб", "🐬": "дельфин", "🐳": "кит", "🐊": "крокодил",
    "🦈": "акула", "🐘": "слон", "🦒": "жираф", "🦘": "кенгуру",
    "🦜": "попугай", "🦔": "ёжик",
    "🍎": "яблоко", "🍐": "груша", "🍊": "апельсин", "🍋": "лимон",
    "🍌": "банан", "🍉": "арбуз", "🍇": "виноград", "🍓": "клубника",
    "🍒": "вишня", "🍑": "персик", "🥭": "манго", "🍍": "ананас",
    "🥥": "кокос", "🥝": "киви", "🍅": "помидор", "🥕": "морковь",
    "🌽": "кукуруза", "🥦": "брокколи", "🥔": "картофель",
    "🍔": "бургер", "🍕": "пицца", "🍟": "фри", "🌭": "хот дог",
    "🥪": "бутерброд", "🍝": "спагетти", "🍜": "лапша", "🍰": "торт",
    "🍪": "печенье", "🍩": "пончик", "🍫": "шоколад", "🍦": "мороженое",
    "🍧": "мороженое", "🍨": "мороженое", "🍬": "конфета", "🍭": "леденец",
    "☕": "кофе", "🍵": "чай", "🥛": "молоко", "🧀": "сыр", "🥚": "яйцо",
    "🍗": "курица", "🥩": "мясо", "🍞": "хлеб", "🥐": "круассан",
    "🍯": "мёд",
    "📱": "телефон", "💻": "компьютер", "⌨": "клавиатура", "🖥": "монитор",
    "📷": "фотоаппарат", "📺": "телевизор", "📞": "телефон", "☎": "телефон",
    "⏰": "будильник", "⌚": "часы", "🎮": "игра", "🎲": "кубик",
    "📚": "книги", "📖": "книга", "📕": "книга", "✏": "карандаш",
    "✂": "ножницы", "🔑": "ключ", "🔒": "замок", "🔓": "замок",
    "💡": "лампочка", "🔍": "лупа", "🔔": "звонок", "🎁": "подарок",
    "🎈": "шарик", "🎉": "праздник", "🎊": "праздник", "🎂": "торт",
    "🧸": "мишка", "🎵": "музыка", "🎶": "музыка", "🎤": "микрофон",
    "🎧": "наушники", "🎬": "кино", "🎥": "камера", "🏆": "кубок",
    "🏅": "медаль", "🥇": "золотая медаль", "🥈": "серебряная медаль",
    "🥉": "бронзовая медаль",
    "☀": "солнце", "🌞": "солнце", "🌝": "луна", "🌙": "луна",
    "☁": "облако", "⛅": "облако", "🌤": "солнце", "🌧": "дождь",
    "🌦": "дождь", "🌨": "снег", "🌩": "гроза", "⛈": "гроза",
    "❄": "снежинка", "☃": "снеговик", "⛄": "снеговик", "🌈": "радуга",
    "🌪": "ураган", "💨": "ветер", "🌊": "волна", "💧": "капля",
    "💦": "вода", "☔": "зонт", "⚡": "молния", "🌋": "вулкан",
    "🌲": "дерево", "🌳": "дерево", "🌹": "роза", "🌻": "подсолнух",
    "🌸": "цветок", "🌷": "цветок", "🍀": "клевер", "🌿": "трава",
    "🌍": "планета", "🌎": "планета", "🌏": "планета",
    "🚀": "ракета", "✈": "самолёт", "🚁": "вертолёт", "🚗": "машина",
    "🚕": "такси", "🚌": "автобус", "🚲": "велосипед", "🚂": "поезд",
    "🚒": "пожарная машина", "🚑": "скорая помощь", "🚓": "полиция",
    "🚜": "трактор", "🛸": "нло", "⛵": "яхта", "🚢": "корабль",
    "👦": "мальчик", "👧": "девочка", "👨": "мужчина", "👩": "женщина",
    "👶": "младенец", "👴": "дедушка", "👵": "бабушка", "👮": "полицейский",
    "💂": "гвардеец", "👷": "строитель", "🧑": "человек", "👤": "человек",
    "👥": "люди",
    "💯": "сто", "🔥": "огонь", "✨": "искры", "⭐": "звезда",
    "🌟": "звезда", "💫": "звезда", "💥": "взрыв", "✅": "галочка",
    "❌": "крестик", "❗": "восклицательный знак", "❕": "восклицательный знак",
    "❓": "вопросительный знак", "❔": "вопросительный знак",
    "💤": "сон", "💬": "разговор", "💭": "мысль", "⚠": "предупреждение",
    "💋": "поцелуй", "👀": "глаза", "👁": "глаз", "👂": "ухо",
    "👃": "нос", "👄": "рот", "🦷": "зуб", "🦴": "кость",
    "🧠": "мозг", "❤": "сердце", "🫀": "сердце", "🫁": "лёгкие",
}

FLAGS = {
    "🇷🇺": ("флаг", "россии"), "🇺🇸": ("флаг", "америки"),
    "🇺🇦": ("флаг", "украины"), "🇧🇾": ("флаг", "беларуси"),
    "🇰🇿": ("флаг", "казахстана"), "🇦🇲": ("флаг", "армении"),
    "🇬🇧": ("флаг", "англии"), "🇩🇪": ("флаг", "германии"),
    "🇫🇷": ("флаг", "франции"), "🇨🇳": ("флаг", "китая"),
    "🇯🇵": ("флаг", "японии"), "🇰🇷": ("флаг", "кореи"),
    "🇮🇹": ("флаг", "италии"), "🇪🇸": ("флаг", "испании"),
}


def _emoji_words(text, i):
    j = i
    while j < len(text):
        ch = text[j]
        if unicodedata.category(ch) == "So" or ch in "\ufe0f\ufe0e\u200d\u20e3" \
                or 0x1F3FB <= ord(ch) <= 0x1F3FF:
            j += 1
        else:
            break
    if j == i:
        return None
    run = [ch for ch in text[i:j]
           if ch not in "\ufe0f\ufe0e\u200d\u20e3"
           and not 0x1F3FB <= ord(ch) <= 0x1F3FF]
    words = []
    k = 0
    while k < len(run):
        ch = run[k]
        if 0x1F1E6 <= ord(ch) <= 0x1F1FF and k + 1 < len(run) \
                and 0x1F1E6 <= ord(run[k + 1]) <= 0x1F1FF:
            names = FLAGS.get(ch + run[k + 1])
            words.extend(names if names else ("флаг",))
            k += 2
        else:
            names = EMOJI.get(ch)
            words.extend(names.split() if names else ("эмодзи",))
            k += 1
    return words, j


def tokenize(text):
    items = []
    i = 0
    while i < len(text):
        c = text[i]
        if c.isspace():
            i += 1
            continue
        if c.isdigit() and unicodedata.category(c) == "Nd":
            j = i
            while j < len(text):
                ch = text[j]
                if ch.isdigit() and unicodedata.category(ch) == "Nd":
                    j += 1
                elif ch in ".," and j + 1 < len(text) and text[j + 1].isdigit() \
                        and unicodedata.category(text[j + 1]) == "Nd":
                    j += 1
                else:
                    break
            chunk = unicodedata.normalize("NFKC", text[i:j])
            sepidx = [k for k, ch in enumerate(chunk) if ch in ".,"]
            parts = []
            last = 0
            for k in sepidx:
                parts.append(chunk[last:k])
                last = k + 1
            parts.append(chunk[last:])
            items.append(("num", parts, [chunk[k] for k in sepidx]))
            i = j
            continue
        if c.isalpha():
            j = i
            while j < len(text) and text[j].isalpha():
                j += 1
            items.append(("word", text[i:j].lower()))
            i = j
            continue
        if c in ",，、՝":
            items.append(("comma",))
        elif c in "!！‼՜":
            items.append(("excl",))
        elif c in "?？՞":
            items.append(("quest",))
        elif c in _PAUSE_PUNCT:
            items.append(("pause", _PAUSE_PUNCT[c]))
        elif c in _SKIP_PUNCT:
            pass
        else:
            name = SYMBOL_NAMES.get(c)
            if name is not None:
                items.append(("sym", name))
            else:
                emo = _emoji_words(text, i)
                if emo is not None:
                    for w in emo[0]:
                        items.append(("word", w))
                    i = emo[1]
                    continue
        i += 1
    return items


def mark_emph(toks, level):
    for t in reversed(toks):
        if t.type == "vowel":
            t.emph = max(t.emph, level)
            break


def build_tokens(text, charMode=False):
    toks = []
    prev_word = False
    am = any("\u0530" <= c <= "\u058F" for c in text)
    for it in tokenize(text):
        kind = it[0]
        if kind == "word":
            if prev_word:
                toks.append(Token(" ", False, False))
            toks.extend(word_to_tokens(it[1], charMode))
            prev_word = True
        elif kind == "num":
            if prev_word:
                toks.append(Token(" ", False, False))
            parts = it[1]
            seps = it[2]
            numstr = armenian_number_to_words if am else number_to_russian
            digit_words = AM_DIGIT_WORDS if am else DIGIT_WORDS

            def sep_word(ch):
                if am:
                    return "կետ" if ch == "." else "ստորակետ"
                return "точка" if ch == "." else "запятая"

            if not seps:
                toks.extend(word_to_tokens(numstr(int(parts[0])), charMode))
            elif len(seps) == 1:
                toks.extend(word_to_tokens(numstr(int(parts[0])), charMode))
                toks.append(Token(" ", False, False))
                toks.extend(word_to_tokens(sep_word(seps[0]), charMode))
                for d in parts[1]:
                    toks.append(Token(" ", False, False))
                    toks.extend(word_to_tokens(digit_words[int(d)], charMode))
            else:
                for idx, p in enumerate(parts):
                    if p and p.isdigit():
                        toks.extend(word_to_tokens(numstr(int(p)), charMode))
                    if idx < len(seps):
                        toks.append(Token(" ", False, False))
                        toks.extend(word_to_tokens(sep_word(seps[idx]), charMode))
            prev_word = True
        elif kind == "comma":
            t = Token(" ", False, False)
            t.dur = 0.16
            toks.append(t)
            prev_word = False
        elif kind == "pause":
            t = Token(" ", False, False)
            t.dur = it[1]
            toks.append(t)
            prev_word = False
        elif kind == "sym":
            if prev_word:
                toks.append(Token(" ", False, False))
            for w in it[1].split():
                toks.extend(word_to_tokens(w, charMode))
                toks.append(Token(" ", False, False))
            prev_word = True
        elif kind == "excl":
            mark_emph(toks, 1.0)
            t = Token(" ", False, False)
            t.dur = 0.35
            toks.append(t)
            prev_word = False
        elif kind == "quest":
            mark_emph(toks, 0.5)
            t = Token(" ", False, False)
            t.dur = 0.35
            toks.append(t)
            prev_word = False
    final = Token(" ", False, False)
    final.dur = 0.28
    toks.append(final)
    t0 = 0.0
    for t in toks:
        t.start = t0
        t0 += t.dur
    return toks


def build_f0(tokens, sr):
    total = sum(t.dur for t in tokens)
    n_total = int(round(total * sr))
    cps = [(0.0, 128.0)]
    weight = {"vowel": 1.2, "nasal": 0.6, "liquid": 0.7, "trill": 0.7,
              "glide": 0.5, "vfric": 0.4, "plosive": 0.3}
    for t in tokens:
        if t.type in weight:
            c = t.start + t.dur / 2.0
            frac = c / total if total > 0 else 0.0
            base = 128.0 - 26.0 * frac
            bump = 20.0 if t.stressed else 0.0
            emph = t.emph * 30.0
            f0 = base + bump * 0.7 + weight[t.type] * 4.0 + emph
            f0 = max(75.0, min(210.0, f0))
            cps.append((c, f0))
    last_emph = max((t.emph for t in tokens if t.type == "vowel"), default=0.0)
    if last_emph >= 1.0:
        end_f0 = 118.0
    elif last_emph >= 0.5:
        end_f0 = 150.0
    else:
        end_f0 = 96.0
    cps.append((total, end_f0))
    cps = sorted(set(cps), key=lambda p: p[0])
    out = [0.0] * n_total
    k = 0
    for i in range(n_total):
        tt = i / sr
        while k + 1 < len(cps) - 1 and cps[k + 1][0] <= tt:
            k += 1
        t0, v0 = cps[k]
        t1, v1 = cps[k + 1]
        span = t1 - t0
        f = v0 + (v1 - v0) * ((tt - t0) / span) if span > 0 else v0
        f += 2.0 * math.sin(2.0 * math.pi * 5.0 * tt)
        out[i] = f
    return out


def _vowel(tok, info, f0s, sr, phase0):
    n = len(f0s)
    F = [f * _FS for f in info["formants"]]
    B = list(info["bws"])
    if tok.soft:
        F[1] += 280 * _FS
        F[2] += 140 * _FS
    phase, sig = glottal(f0s, sr, phase0, breath=0.012)
    for f, b in zip(F, B):
        sig = Resonator(sr, f, b).run(sig)
    sig = _scale_rms(sig, 0.36)
    env = envelope(n, 0.012 * sr, 0.16 * n)
    amp = info["amp"] * (1.0 if tok.stressed else 0.70)
    return [s * e * amp for s, e in zip(sig, env)], phase


def _sonorant(tok, info, f0s, sr, phase0):
    n = len(f0s)
    F = [f * _FS for f in info["formants"]]
    B = list(info["bws"]) if "bws" in info else [70, 90, 120]
    if tok.soft and info["type"] == "liquid":
        F[1] += 150 * _FS
    phase, sig = glottal(f0s, sr, phase0, breath=0.010, q=0.55)
    for f, b in zip(F, B):
        sig = Resonator(sr, f, b).run(sig)
    if "zero" in info:
        sig = ZeroFilter(sr, info["zero"] * _FS).run(sig)
    sig = _scale_rms(sig, 0.19)
    env = envelope(n, 0.012 * sr, 0.2 * n)
    amp = info["amp"] * (1.05 if tok.stressed else 0.85)
    return [s * e * amp for s, e in zip(sig, env)], phase


def _trill(tok, info, f0s, sr, phase0):
    n = len(f0s)
    F = [f * _FS for f in info["formants"]]
    B = info["bws"]
    phase, sig = glottal(f0s, sr, phase0, breath=0.015, q=0.5)
    for f, b in zip(F, B):
        sig = Resonator(sr, f, b).run(sig)
    sig = _scale_rms(sig, 0.24)
    env = envelope(n, 0.012 * sr, 0.2 * n)
    out = []
    for i, (s, e) in enumerate(zip(sig, env)):
        t = i / sr
        gate = 0.55 + 0.45 * max(0.0, math.sin(2.0 * math.pi * 26.0 * t))
        out.append(s * e * info["amp"] * gate)
    return out, phase


def _plosive(tok, info, f0s, sr, phase0):
    n = len(f0s)
    voiced = info.get("voiced", False)
    closure = min(int(info["closure"] * sr), n - 2)
    burst = min(int(info["burst"] * sr), n - closure - 1)
    remain = n - closure - burst
    out = []
    phase = phase0

    if voiced:
        phase, vb = glottal(f0s[:closure], sr, phase, breath=0.002, q=0.45)
        for f, b in ((180 * _FS, 60), (950 * _FS, 250)):
            vb = Resonator(sr, f, b).run(vb)
        vb = _scale_rms(vb, 0.10)
        fenv = envelope(closure, 0.3 * closure, 0.1 * closure)
        out.extend(s * e for s, e in zip(vb, fenv))
    else:
        out.extend([0.0] * closure)
        for f0 in f0s[:closure]:
            phase += f0 / sr

    bn = white_noise(burst)
    f = info["bfilter"]
    if f == "п":
        bn = lowpass1(bn, sr, 650 * _FS)
    elif f == "т":
        bn = lowpass1(bn, sr, 2600 * _FS)
    else:
        bn = highpass1(bn, sr, 1200 * _FS)
    bn = _scale_rms(bn, 0.10)
    benv = envelope(burst, 0.006 * sr, burst)
    out.extend(bn[i] * benv[i] * info["amp"] * 1.7 for i in range(burst))

    if voiced:
        out.extend([0.0] * remain)
    else:
        asp = int(min(0.03 * sr, remain))
        an = white_noise(asp)
        an = Resonator(sr, 1600 * _FS if f in ("п", "т") else 2600 * _FS, 500).run(an)
        an = _scale_rms(an, 0.09)
        aenv = envelope(asp, 0.004 * sr, asp)
        for k in range(asp):
            dec = math.exp(-4.0 * k / asp)
            out.append(an[k] * aenv[k] * dec)
        out.extend([0.0] * (remain - asp))
    return out, phase


def _fric(tok, info, f0s, sr, phase0, voiced):
    n = len(f0s)
    fr = [(f * _FS, b) for f, b in info["fric"]]
    if tok.soft:
        fr = [(f + 400 * _FS, b) for f, b in fr]
    noise = white_noise(n)
    for f, b in fr:
        noise = Resonator(sr, f, b).run(noise)
    phase = phase0
    if voiced:
        phase, buzz = glottal(f0s, sr, phase, breath=0.012)
        for f, b in zip([x * _FS for x in info["formants"]], [70, 120, 180]):
            buzz = Resonator(sr, f, b).run(buzz)
        buzz = _scale_rms(buzz, 0.16)
    noise = _scale_rms(noise, 0.16)
    env = envelope(n, 0.006 * sr, 0.2 * n)
    amp = info["amp"]
    if voiced:
        return [(b * 0.7 + ns * 0.3) * e * amp for b, ns, e in zip(buzz, noise, env)], phase
    return [ns * e * amp * 0.85 for ns, e in zip(noise, env)], phase


def _affricate(tok, info, f0s, sr, phase0):
    n = len(f0s)
    closure = int(0.03 * sr)
    burst = int(0.012 * sr)
    out = [0.0] * closure

    bn = white_noise(burst)
    bn = lowpass1(bn, sr, 2200 * _FS)
    benv = envelope(burst, 0.005 * sr, burst)
    out.extend(bn[i] * benv[i] * 0.30 for i in range(burst))

    remain = n - closure - burst
    ns = white_noise(remain)
    for f, b in info["fric"]:
        ns = Resonator(sr, f * _FS, b).run(ns)
    ns = _scale_rms(ns, 0.12)
    nenv = envelope(remain, 0.005 * sr, 0.2 * remain)
    out.extend(s * e * 1.8 for s, e in zip(ns, nenv))

    phase = phase0
    for f0 in f0s:
        phase += f0 / sr
    return out, phase


def synth_phone(tok, f0s, sr, phase0):
    info = PHONES[tok.phone]
    typ = info["type"]
    if typ == "pause":
        return [0.0] * len(f0s), phase0
    if typ == "vowel":
        return _vowel(tok, info, f0s, sr, phase0)
    if typ in ("nasal", "liquid", "glide"):
        return _sonorant(tok, info, f0s, sr, phase0)
    if typ == "trill":
        return _trill(tok, info, f0s, sr, phase0)
    if typ == "plosive":
        return _plosive(tok, info, f0s, sr, phase0)
    if typ == "fric":
        return _fric(tok, info, f0s, sr, phase0, voiced=False)
    if typ == "vfric":
        return _fric(tok, info, f0s, sr, phase0, voiced=True)
    if typ == "affricate":
        return _affricate(tok, info, f0s, sr, phase0)
    return [0.0] * len(f0s), phase0


def _build_audio(text, sr, seed, voice, f0mul, rate=1.0, charMode=False, abort=None):
    global _F0MUL, _FS
    voice = _norm_voice(voice)
    _F0MUL = max(0.1, VOICES[voice]["f0"] * f0mul)
    _FS = VOICES[voice]["fs"]
    if seed is not None:
        random.seed(seed)
    tokens = build_tokens(text, charMode)
    plan = [(t.phone, round(t.dur, 3), "удар" if t.stressed else "")
            for t in tokens]
    if not tokens:
        return b"", 0, 0.0, 0.0, plan

    rate = max(0.3, min(3.0, rate))
    if rate != 1.0:
        t0 = 0.0
        for t in tokens:
            t.dur = t.dur / rate
            t.start = t0
            t0 += t.dur

    f0 = [v * _F0MUL for v in build_f0(tokens, sr)]

    audio = []
    pos = 0
    phase = 0.0
    for tok in tokens:
        if abort is not None and abort():
            return b"", 0, 0.0, 0.0, plan
        n = int(round(tok.dur * sr))
        seg = f0[pos:pos + n]
        if len(seg) < n:
            seg = list(seg) + [seg[-1] if seg else 130.0] * (n - len(seg))
        buf, phase = synth_phone(tok, seg, sr, phase)
        audio.extend(buf)
        pos += n

    cleaned = _dc_block(audio, sr)
    cleaned = _presence(cleaned, sr)

    peak = max(1e-9, max(abs(v) for v in cleaned))
    g = 0.95 / peak
    th = math.tanh(1.4)
    limited = [math.tanh(v * g * 1.4) / th for v in cleaned]
    rms_lev = math.sqrt(sum(s * s for s in limited) / len(limited))
    data = b"".join(struct.pack("<h", int(v * 32767)) for v in limited)
    return data, pos, (0.95 * th / 1.4), rms_lev, plan


def render_pcm(text, sr=SR, seed=None, voice="male", f0mul=1.0,
               rate=1.0, charMode=False, abort=None):
    """Синтез текста в байты PCM (моно 16 бит)."""
    data, _pos, _pk, _rms, _plan = _build_audio(
        text, sr, seed, voice, f0mul, rate, charMode, abort,
    )
    return data

import os
import sys
import wave


def _app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


DEFAULT_OUT = os.path.join(_app_dir(), "voices", "speech.wav")


def synthesize_speech(text, filename=None, sr=SR, seed=None, voice="male"):
    """Синтез текста в WAV-файл. Создаёт DEFAULT_OUT, если filename не задан."""
    data, pos, pk, rms_lev, plan = _build_audio(text, sr, seed, voice, 1.0)
    print("Токены:", plan)
    if not data:
        print("Готово: пустой текст.")
        return
    filename = filename or DEFAULT_OUT
    outdir = os.path.dirname(os.path.abspath(filename))
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with wave.open(filename, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data)
    print(f"Готово: {filename}, {pos / sr:.2f} с, {sr} Гц / 16 бит / моно, "
          f"{sr * 16 // 1000} кбит/с, пик {pk:.2f}, RMS {rms_lev:.3f}")


def _readline(prompt="> "):
    sys.stdout.write(prompt)
    sys.stdout.flush()
    try:
        raw = sys.stdin.buffer.readline()
    except AttributeError:
        return input(prompt)
    if not raw:
        return ""
    for enc in ("utf-8", "cp1251", "cp866"):
        try:
            return raw.decode(enc).strip("\r\n")
        except Exception:
            continue
    return raw.decode("utf-8", "replace").strip("\r\n")


def console_loop(sr=SR):
    print("SuperSpeak — интерактивный синтез.")
    print("Вводите текст (Enter) — каждый раз сохраняется speech.wav.")
    print("После каждого текста укажите голос: male / girl (введите Enter — мужской).")
    print("Пустой ввод текста или 'выход' — завершение.")
    while True:
        try:
            text = _readline("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text.lower() in ("выход", "exit", "quit", "q", "stop"):
            break
        v = _readline("голос [male]: ").strip() or "male"
        synthesize_speech(text, None, sr, voice=v)


if __name__ == "__main__":
    args = sys.argv[1:]
    rate = SR
    out = None
    voice = "male"
    words = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--rate"):
            rate = int(arg.split("=")[1])
        elif arg.startswith("--out"):
            out = arg.split("=")[1]
        elif arg.startswith("--voice"):
            voice = arg.split("=")[1]
        else:
            words.append(arg)
        i += 1
    if words:
        synthesize_speech(" ".join(words), out or DEFAULT_OUT, rate, voice=voice)
    else:
        console_loop(rate)
