"""BFS oracle birim testleri — PLAN §Asama 2 kabul kriterleri.

Calistir:  .venv\\Scripts\\python.exe -m tests.test_oracle
"""
from itertools import product

from baselines.bfs_oracle import (all_shortest_paths, bfs_dist, bfs_path,
                                  forbidden_from, manhattan, oracle)
from config import GRID_N

# BILEREK GRID_N'e DEGIL sabit kucuk bir altkumeye bagli: bu test algoritma
# DOGRULUGUNU (all_shortest_paths/oracle) sinar, olcek testi degil. GRID_N=50
# ile CELLS = 50*50 hucre olsaydi product(CELLS,CELLS) 6.25M cift dener,
# bircogu buyuk mesafeli olur ve MemoryError'a kadar gider (olculdu) —
# GRID_N buyudukce bu test SESSIZCE 5x5'ten milyonlarca kombinasyona
# sicramamali.
CELLS = [(r, c) for r in range(5) for c in range(5)]


def test_bfs_basics():
    assert bfs_dist((0, 0), (0, 0)) == 0
    assert bfs_dist((0, 0), (4, 4)) == 8
    assert bfs_dist((0, 0), (0, 4)) == 4
    # tam duvar: 1. satir kapali -> alt yariya gecis yok
    wall = frozenset((1, c) for c in range(GRID_N))
    assert bfs_dist((0, 0), (4, 4), wall) is None
    # duvarda tek delik acilinca gecis var
    wall_hole = wall - {(1, 2)}
    assert bfs_dist((0, 0), (4, 4), wall_hole) == 8
    # bfs_path uzunlugu bfs_dist ile tutarli
    for s, g in [((0, 0), (4, 4)), ((2, 1), (0, 3)), ((4, 0), (0, 4))]:
        assert len(bfs_path(s, g)) - 1 == bfs_dist(s, g) == manhattan(s, g)
    print("  test_bfs_basics OK")


def test_all_shortest_paths():
    assert len(all_shortest_paths((0, 0), (4, 4))) == 70      # C(8,4)
    assert len(all_shortest_paths((0, 0), (0, 4))) == 1       # duz cizgi -> tek yol
    assert len(all_shortest_paths((0, 0), (0, 0))) == 1
    assert len(all_shortest_paths((4, 4), (0, 0))) == 70      # ters yon de calismali
    for s, g in product(CELLS, CELLS):
        for p in all_shortest_paths(s, g):
            assert p[0] == s and p[-1] == g
            assert len(p) - 1 == manhattan(s, g)
            assert len(set(p)) == len(p)                      # kendini kesmiyor
            for a, b in zip(p, p[1:]):
                assert manhattan(a, b) == 1                   # gercekten komsu
    print("  test_all_shortest_paths OK")


def test_exemption_rule():
    """PLAN §1'in kritik kurali: s1, s2, goal yasak bolgeden MUAF."""
    s1 = s2 = (0, 0)
    g = (4, 4)
    for p in all_shortest_paths(s1, g):
        forb = forbidden_from(p, s1, s2, g)
        assert s1 not in forb and s2 not in forb and g not in forb
    # farkli start, s2 A1'in yolunun uzerinde
    s1, s2, g = (0, 0), (0, 2), (0, 4)
    p = all_shortest_paths(s1, g)[0]          # duz cizgi, (0,2)'den geciyor
    assert s2 in p
    assert s2 not in forbidden_from(p, s1, s2, g)
    print("  test_exemption_rule OK")


def test_closed_form_blocking_rule():
    """PLAN §0.3: s1==s2==(0,0), g=(4,4) icin
    A1 kilitler <=> ILK hamlesi ile SON hamlesi ayni yonde.

    KRITIK: bu kural ORJINAL PLAN turetmesinde 5x5'in KOSE-DUVARLARINA
    dayanir (A2, (4,*)/(  *,4) satir-sutununda baska cikis olmadigi icin
    kosede sikisir). bfs_dist n=GRID_N (artik 50) VARSAYILANINI kullanirsa
    A2 buyuk gridin acik alanindan dolasip kilitlenmez, kural GECERSIZ
    gorunur. Bu YERELDIR (5x5'e ozgu, KANITLANMIS bir teorem), o yuzden
    n=5 ACIKCA sabitleniyor — GRID_N ne olursa olsun bu testin anlami
    degismemeli."""
    s1 = s2 = (0, 0)
    g = (4, 4)
    n_block = 0
    for p in all_shortest_paths(s1, g):
        first = (p[1][0] - p[0][0], p[1][1] - p[0][1])
        last = (p[-1][0] - p[-2][0], p[-1][1] - p[-2][1])
        blocked = bfs_dist(s2, g, forbidden_from(p, s1, s2, g), n=5) is None
        assert blocked == (first == last), f"kapali form kurali bozuldu: {p}"
        n_block += blocked
    assert n_block == 30, n_block          # 15 (D..D) + 15 (R..R)
    print("  test_closed_form_blocking_rule OK  (30/70 kilitliyor)")


def test_oracle_consistency():
    """oracle() dogrudan tarama ile ayni sonucu vermeli."""
    cases = [((0, 0), (0, 0), (4, 4)), ((0, 0), (4, 0), (4, 4)),
             ((2, 1), (3, 3), (0, 4)), ((0, 0), (0, 0), (0, 2))]
    for s1, s2, g in cases:
        orc = oracle(s1, s2, g)
        paths = all_shortest_paths(s1, g)
        dists = [bfs_dist(s2, g, forbidden_from(p, s1, s2, g)) for p in paths]
        alive = [d for d in dists if d is not None]
        assert orc.n_paths == len(paths)
        assert orc.n_blocking == sum(d is None for d in dists)
        assert orc.best_len2 == (min(alive) if alive else None)
        assert orc.len1 == manhattan(s1, g)
        if orc.best_path1 is not None:
            d = bfs_dist(s2, g, forbidden_from(orc.best_path1, s1, s2, g))
            assert d == orc.best_len2
    # PLAN §0.2'deki +2 ornegi: ayni start, duz cizgi, A1'in alternatifi yok
    orc = oracle((0, 0), (0, 0), (0, 2))
    assert orc.n_paths == 1 and orc.best_len2 == 4 and orc.free_len2 == 2
    print("  test_oracle_consistency OK")


def test_no_config_is_dead():
    """PLAN §0.2: hicbir konfigurasyon tamamen cozumsuz degil."""
    dead = [(s1, s2, g) for s1, s2, g in product(CELLS, CELLS, CELLS)
            if s1 != g and s2 != g and oracle(s1, s2, g).best_len2 is None]
    assert dead == [], f"{len(dead)} cozumsuz konfig bulundu: {dead[:3]}"
    print("  test_no_config_is_dead OK  (14400/14400 cozulebilir)")


if __name__ == "__main__":
    print("tests/test_oracle.py")
    test_bfs_basics()
    test_all_shortest_paths()
    test_exemption_rule()
    test_closed_form_blocking_rule()
    test_oracle_consistency()
    test_no_config_is_dead()
    print("TUM ORACLE TESTLERI GECTI")
