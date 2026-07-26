"""Curriculum / zorluk-agirlikli konfig sampler — PLAN §Asama 6.

PLAN §0.3: uniform sampling'de konfiglerin %70.8'i "kolay" (A1'in secimi
onemsiz), yani egitimde gorulen sinyalin cogu bedava geliyor. En zor %10'luk
dilimde zarar orani %69.2 (uniform ortalamada %13.3) — zor konfiglere agirlik
vermek sinyali ~5 kat yogunlastiriyor.

d(s1,g)=1 olan konfigler HER ZAMAN "kolay" (A1'in tek bir optimal yolu var,
secme sansi yok — PLAN §0.4) VE sifir koordinasyon icerigi tasir; bunlari
"kolay" havuzu icinde de agirlikli sekilde kisiyoruz (TRIVIAL_WEIGHT).

IKI MOD:
  - kucuk N (5x5 gibi): `csv_path` verilirse Asama 2'nin TAM tarama
    (`difficulty.csv`) dosyasindan KESIN kovalar okunur.
  - buyuk N (100x100 gibi): `grid_n` verilirse tam tarama IMKANSIZ
    (14.400 konfig -> 10^12'ye cikiyor), bunun yerine `n_samples` kadar
    rastgele konfig uretilip ayni esik-tabanli kurala gore (grid_env.py'nin
    `_terminal_info()`'undaki `is_hard` ile AYNI mantik) siniflandirilir.
    Kesin degil (enumerate yerine esik), ama enumerate GEREKTIRMEZ.
"""
import csv

import numpy as np

from config import DIFFICULTY_CSV, P_HARD_CAP, P_HARD_END, P_HARD_START

Cell = tuple[int, int]
Config = tuple[Cell, Cell, Cell]

# easy havuzu icinde trivial (d1=1) konfiglere ayrilan pay — tamamen atmiyoruz
# (agirligi 0 yaparsan agin "d1=1 diye bir sey var" bilgisini de kaybeder),
# ama PLAN'in "kisildi" talimatina uyacak sekilde agir bastiriyoruz.
TRIVIAL_WEIGHT = 0.1

# grid_env.py'nin _terminal_info()'undaki is_hard esigiyle AYNI olmali —
# ikisi ayri yerlerde tanimli oldugu icin degistirirsen ikisini de guncelle.
HARD_DISTANCE_FRAC = 0.6


def _manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _sample_one_config(n: int, rng: np.random.Generator) -> Config:
    cells_hi = n
    while True:
        s1 = (int(rng.integers(0, cells_hi)), int(rng.integers(0, cells_hi)))
        s2 = (int(rng.integers(0, cells_hi)), int(rng.integers(0, cells_hi)))
        g = (int(rng.integers(0, cells_hi)), int(rng.integers(0, cells_hi)))
        if s1 != g and s2 != g:
            return s1, s2, g


class CurriculumSampler:
    """sample(episode) her cagrida p_hard(episode) olasiligiyla hard kovasindan,
    geri kalaninda agirlikli sekilde easy kovalarindan secer.
    """

    def __init__(self, csv_path: str = DIFFICULTY_CSV, seed: int | None = None,
                 total_episodes: int = 1,
                 p_start: float = P_HARD_START, p_end: float = P_HARD_END,
                 p_cap: float = P_HARD_CAP,
                 grid_n: int | None = None, n_samples: int = 50_000):
        self.rng = np.random.default_rng(seed)
        self.total_episodes = max(1, total_episodes)
        self.p_start, self.p_end, self.p_cap = p_start, p_end, p_cap

        self.hard: list[Config] = []
        self.easy_nontrivial: list[Config] = []
        self.easy_trivial: list[Config] = []

        if grid_n is not None:
            self._build_pools_by_sampling(grid_n, n_samples)
        else:
            self._build_pools_from_csv(csv_path)

        # BUYUK N NOTU: d(s1,g)=1 olan "trivial" konfigler 5x5'te 1920/14400
        # (%13.3) — anlamli bir dilimdi. 100x100'de rastgele orneklemede
        # goruntme olasiligi ~4/10000 — n_samples=1000'de beklenen sayi <1,
        # neredeyse hic cikmaz. Bu kova BOS olabilir; sample()'da bos ise
        # agirligini easy_nontrivial'a devrediyoruz (asagida), hata FIRLATMIYORUZ.
        if not (self.hard and self.easy_nontrivial):
            raise RuntimeError(
                f"hard/easy_nontrivial kovalari olusmadi (hard={len(self.hard)}, "
                f"easy_nontrivial={len(self.easy_nontrivial)}) — grid_n cok kucuk "
                f"olabilir, n_samples'i artir ya da difficulty.csv'yi kontrol et.")

    def _build_pools_from_csv(self, csv_path: str):
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                s1 = (int(row["s1_r"]), int(row["s1_c"]))
                s2 = (int(row["s2_r"]), int(row["s2_c"]))
                g = (int(row["g_r"]), int(row["g_c"]))
                cfg = (s1, s2, g)
                if row["difficulty"] == "hard":
                    self.hard.append(cfg)
                elif int(row["d1"]) == 1:
                    self.easy_trivial.append(cfg)
                else:
                    self.easy_nontrivial.append(cfg)

    def _build_pools_by_sampling(self, grid_n: int, n_samples: int):
        """Buyuk N icin: tam tarama yerine n_samples rastgele konfig uret,
        grid_env.py'nin is_hard esigiyle AYNI kuralla siniflandir."""
        max_man = 2 * (grid_n - 1)
        for _ in range(n_samples):
            s1, s2, g = _sample_one_config(grid_n, self.rng)
            d1 = _manhattan(s1, g)
            is_hard = bool(max_man > 0 and d1 / max_man >= HARD_DISTANCE_FRAC)
            if is_hard:
                self.hard.append((s1, s2, g))
            elif d1 == 1:
                self.easy_trivial.append((s1, s2, g))
            else:
                self.easy_nontrivial.append((s1, s2, g))

    def p_hard(self, episode: int) -> float:
        """PLAN §Asama 6: p_hard = min(cap, start + (end-start) * episode/total)."""
        frac = min(1.0, episode / self.total_episodes)
        return min(self.p_cap, self.p_start + (self.p_end - self.p_start) * frac)

    def sample(self, episode: int) -> Config:
        p_h = self.p_hard(episode)
        u = self.rng.random()
        if u < p_h:
            pool = self.hard
        elif u < p_h + (1 - p_h) * (1 - TRIVIAL_WEIGHT):
            pool = self.easy_nontrivial
        else:
            # buyuk N'de bu kova sik sik BOS olur (bkz. __init__ notu) —
            # bos ise agirligini easy_nontrivial'a devret.
            pool = self.easy_trivial or self.easy_nontrivial
        return pool[int(self.rng.integers(0, len(pool)))]
