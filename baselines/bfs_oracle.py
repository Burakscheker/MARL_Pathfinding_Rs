"""BFS tabanli EXACT oracle — projenin dogruluk zemini (PLAN §Asama 2).

Sirali modda oracle tam kesindir: A1 tum yolunu bitirdikten sonra yasak bolge
sabitlenir, dolayisiyla A2'nin optimumu A1'in secimine kosullu olarak kesin
hesaplanabilir. Butun RL sonuclari buna gore olculur.
"""
from collections import deque
from functools import lru_cache
from itertools import combinations
from math import comb
from typing import NamedTuple, Optional

from config import DIRS, GRID_N

Cell = tuple[int, int]
Path = tuple[Cell, ...]


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def in_bounds(c: Cell, n: int = GRID_N) -> bool:
    return 0 <= c[0] < n and 0 <= c[1] < n


def neighbors(c: Cell, n: int = GRID_N):
    for dr, dc in DIRS:
        nb = (c[0] + dr, c[1] + dc)
        if in_bounds(nb, n):
            yield nb


# --------------------------------------------------------------------- BFS

def bfs_distance_map(source: Cell, blocked=frozenset(), n: int = GRID_N) -> dict:
    """source'tan TUM ulasilabilir hucrelere BFS mesafesi (tek-kaynak tam
    flood-fill), sozluk olarak.

    Duvar-farkinda reward shaping icin: hedeften TEK BFS ile grid'in
    tamamina mesafe cikarilir, sonra HER ADIMDA sozlukten O(1) okunur —
    ayni (nokta,hedef) cifti icin defalarca bfs_dist() cagirmaktan
    (her biri kendi ayri BFS'i) cok daha ucuz. Bir episode'da en fazla
    2 kez cagrilir (FAZ A basinda A1 icin, FAZ B basinda A2 icin) —
    duvarlar/yasak bolge o fazin SUresi boyunca sabit kaldigindan.
    """
    if source in blocked:
        return {source: 0}
    dist = {source: 0}
    q = deque([source])
    while q:
        cur = q.popleft()
        d = dist[cur]
        for nb in neighbors(cur, n):
            if nb in blocked or nb in dist:
                continue
            dist[nb] = d + 1
            q.append(nb)
    return dist


def bfs_dist(start: Cell, goal: Cell, blocked=frozenset(), n: int = GRID_N) -> Optional[int]:
    """En kisa mesafe, ulasilamiyorsa None."""
    if start in blocked or goal in blocked:
        return None
    if start == goal:
        return 0
    seen = {start}
    q = deque([(start, 0)])
    while q:
        cur, d = q.popleft()
        for nb in neighbors(cur, n):
            if nb in blocked or nb in seen:
                continue
            if nb == goal:
                return d + 1
            seen.add(nb)
            q.append((nb, d + 1))
    return None


def bfs_path(start: Cell, goal: Cell, blocked=frozenset(), n: int = GRID_N) -> Optional[Path]:
    """En kisa yollardan biri (deterministik), ulasilamiyorsa None."""
    if start in blocked or goal in blocked:
        return None
    if start == goal:
        return (start,)
    prev: dict[Cell, Cell] = {start: start}
    q = deque([start])
    while q:
        cur = q.popleft()
        for nb in neighbors(cur, n):
            if nb in blocked or nb in prev:
                continue
            prev[nb] = cur
            if nb == goal:
                out = [goal]
                while out[-1] != start:
                    out.append(prev[out[-1]])
                return tuple(reversed(out))
            q.append(nb)
    return None


def sample_random_monotonic_path(start: Cell, goal: Cell, rng) -> Path:
    """UNIFORM rastgele TEK monoton en kisa yol — enumerate ETMEDEN.

    all_shortest_paths() TUM C(dr+dc,dr) yolu materyalize eder (5x5'te 70,
    100x100'de kose-kose C(198,99) — trilyonlarca, imkansiz). Bu fonksiyon
    AYNI dagilimdan (hangi adimlarin "satir adimi" oldugu uniform rastgele
    bir alt kume) TEK bir yolu O(adim sayisi) zamanda uretir, hic listelemez.
    Buyuk N'de random_shortest_policy'nin kullandigi budur.
    """
    if start == goal:
        return (start,)
    dr, dc = goal[0] - start[0], goal[1] - start[1]
    step_r = 1 if dr > 0 else -1
    step_c = 1 if dc > 0 else -1
    n_r, n_c = abs(dr), abs(dc)
    total = n_r + n_c
    row_steps = set(rng.choice(total, size=n_r, replace=False).tolist()) if n_r else set()
    r, c = start
    path = [(r, c)]
    for i in range(total):
        if i in row_steps:
            r += step_r
        else:
            c += step_c
        path.append((r, c))
    return tuple(path)


# ------------------------------------------------- tum en kisa yollar (monoton)

# C(dr+dc, dr) yol sayisi n_r~n_c (KOSEGEN) olunca PATLAR (100x100'de
# C(198,99) ~ 10^58), ama DUZ CIZGIDE (n_r=0 veya n_c=0) mesafe ne olursa
# olsun HEP 1 yoldur — o yuzden esik HAM MESAFEYE degil, GERCEK YOL SAYISINA
# (math.comb) gore konuluyor. guard'li cagiranlar (eval.py, train.py'nin
# load_all_configs'u) GRID_N<=20 kontroluyle zaten bu fonksiyona hic girmiyor,
# ama demo.py/scan.py gibi guard'siz cagiranlar buyuk N'de yanlislikla
# cagirirsa burada acikca patlasin — sessizce saatlerce takilmak yerine.
# Esik: 2M yol — saniyeler icinde biter, ustu pratik degil. Buyuk
# mesafelerde sample_random_monotonic_path() (asagida) kullanilmali —
# O(path-length), enumerate GEREKTIRMEZ.
MAX_ENUMERABLE_PATHS = 2_000_000


@lru_cache(maxsize=None)
def all_shortest_paths(start: Cell, goal: Cell) -> tuple[Path, ...]:
    """start -> goal arasindaki TUM en kisa yollar.

    Engelsiz gridde en kisa yol her zaman monotondur (sadece hedefe yaklastiran
    iki yonde hareket), dolayisiyla adim sirasi bir kombinasyon secimidir:
    C(dr+dc, dr) tane yol. (0,0)->(4,4) icin C(8,4)=70.
    """
    if start == goal:
        return ((start,),)
    dr, dc = goal[0] - start[0], goal[1] - start[1]
    step_r = 1 if dr > 0 else -1
    step_c = 1 if dc > 0 else -1
    n_r, n_c = abs(dr), abs(dc)
    total = n_r + n_c
    n_paths = comb(total, n_r)
    if n_paths > MAX_ENUMERABLE_PATHS:
        raise ValueError(
            f"all_shortest_paths({start},{goal}): C({total},{n_r})="
            f"{n_paths} yol, {MAX_ENUMERABLE_PATHS} sinirinin uzerinde, "
            f"enumerate etmek pratik degil. Buyuk N'de oracle()/"
            f"all_shortest_paths() yerine sample_random_monotonic_path() "
            f"kullan.")

    out: list[Path] = []
    for row_steps in combinations(range(total), n_r):
        rs = set(row_steps)
        r, c = start
        path = [(r, c)]
        for i in range(total):
            if i in rs:
                r += step_r
            else:
                c += step_c
            path.append((r, c))
        out.append(tuple(path))
    return tuple(out)


def forbidden_from(path1: Path, s1: Cell, s2: Cell, goal: Cell) -> frozenset:
    """A1'in izinden yasak bolgeyi uret.

    KRITIK (PLAN §1): s1, s2 ve goal MUAF. Aksi halde s1==s2 olan 600 konfigde
    A2 kendi baslangicinda yasak hucrede kalir ve problem tanim geregi
    cozumsuzlesir. Ortamin 1 numarali birim testi budur.
    """
    return frozenset(path1) - {s1, s2, goal}


# ------------------------------------------------------------------- oracle

class OracleResult(NamedTuple):
    len1: int                  # A1'in optimal yol uzunlugu = manhattan(s1, goal)
    best_len2: Optional[int]   # A2'nin ulasabilecegi en kisa uzunluk (None = hic yol yok)
    best_path1: Optional[Path] # o sonucu veren A1 yolu
    n_paths: int               # A1'in toplam optimal yol sayisi
    n_blocking: int            # bunlardan kacinin A2'yi TAMAMEN kilitledigi
    n_detour: int              # kacinin A2'yi uzattigi (kilitlemeden)
    free_len2: int             # A2'nin engelsiz mesafesi = manhattan(s2, goal)

    @property
    def is_hard(self) -> bool:
        """A1'in secimi onemli mi? (PLAN §0.3 — konfiglerin %29.2'si)"""
        return (self.n_blocking + self.n_detour) > 0

    @property
    def block_rate(self) -> float:
        """A1 optimal yollarindan rastgele birini secerse kilitleme olasiligi."""
        return self.n_blocking / self.n_paths if self.n_paths else 0.0


@lru_cache(maxsize=None)
def oracle(s1: Cell, s2: Cell, goal: Cell) -> OracleResult:
    """A1'in TUM optimal yollarini tarayarak kesin optimumu bul.

    A1 her zaman optimal (len1 = manhattan) kalmak zorunda; serbestligi sadece
    hangi optimal yolu sectigi. Bu fonksiyon o secimin A2 uzerindeki etkisini
    tam olarak olcer.
    """
    paths = all_shortest_paths(s1, goal)
    free_len2 = manhattan(s2, goal)

    best_len2: Optional[int] = None
    best_path1: Optional[Path] = None
    n_blocking = n_detour = 0

    for p in paths:
        blocked = forbidden_from(p, s1, s2, goal)
        d = bfs_dist(s2, goal, blocked)
        if d is None:
            n_blocking += 1
            continue
        if d > free_len2:
            n_detour += 1
        if best_len2 is None or d < best_len2:
            best_len2, best_path1 = d, p

    return OracleResult(
        len1=manhattan(s1, goal),
        best_len2=best_len2,
        best_path1=best_path1,
        n_paths=len(paths),
        n_blocking=n_blocking,
        n_detour=n_detour,
        free_len2=free_len2,
    )


# ------------------------------------------------------- yardimci: gorsellestirme

def render_paths(s1: Cell, s2: Cell, goal: Cell, path1: Path,
                 path2: Optional[Path] = None, n: int = GRID_N) -> str:
    """ASCII: # yasak bolge, 1/2 baslangiclar, G hedef, o A2'nin yolu."""
    forb = forbidden_from(path1, s1, s2, goal)
    p2 = set(path2 or ())
    rows = []
    for r in range(n):
        row = []
        for c in range(n):
            cell = (r, c)
            if cell == goal:
                row.append("G")
            elif cell == s1 == s2:
                row.append("*")
            elif cell == s1:
                row.append("1")
            elif cell == s2:
                row.append("2")
            elif cell in forb:
                row.append("#")
            elif cell in p2:
                row.append("o")
            else:
                row.append(".")
        rows.append(" ".join(row))
    return "\n".join(rows)
