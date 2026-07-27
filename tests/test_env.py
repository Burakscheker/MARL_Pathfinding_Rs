"""Ortam birim + istatistik testleri — PLAN §Asama 1 kabul kriterleri.

Calistir:  .venv\\Scripts\\python.exe -m tests.test_env
"""
import numpy as np

from baselines.policies import random_shortest_policy, random_walk_act
from config import (AGENT_1, AGENT_2, DOWN, MAX_STEPS_PER_PHASE, NOOP,
                    OBS_DIM, RIGHT, LEFT, STATE_DIM)
from env.grid_env import MARLGridEnv


def _run(env, policy, config=None, seed=None):
    env.reset(config=config, seed=seed)
    policy.reset()
    done = False
    info = {}
    while not done:
        _, _, done, info = env.step(policy.act(env))
    return info


def test_shapes_and_masks():
    env = MARLGridEnv(seed=0)
    obs = env.reset(config=((0, 0), (4, 0), (4, 4)))
    # Boyutlar GRID_N'e degil PATCH_SIZE'a bagli (yerel-pencere gozlem) —
    # 5x5 doneminden kalma sabit 129/102 degerleri artik gecerli degil.
    assert obs[AGENT_1].shape == (OBS_DIM,)
    assert env.state().shape == (STATE_DIM,)

    # FAZ A: aktif = A1 (NOOP kapali), pasif = A2 (SADECE NOOP)
    m1, m2 = env.action_mask(AGENT_1), env.action_mask(AGENT_2)
    assert m1[NOOP] == 0.0, "aktif ajanda NOOP acik kalmis"
    assert m2.tolist() == [0, 0, 0, 0, 1], "pasif ajanda NOOP disinda aksiyon acik"
    # (0,0) kosede: sadece SAG ve ASAGI gecerli
    assert m1.tolist() == [0, 1, 1, 0, 0]
    print("  test_shapes_and_masks OK")


def test_exemption_same_start():
    """PLAN §1 kritik kural: s1 == s2 olan 600 konfigde A2 kendi
    baslangicinda yasak hucrede kalmamali."""
    env = MARLGridEnv(seed=0)
    pol = random_shortest_policy(np.random.default_rng(0))
    for g in [(4, 4), (0, 4), (2, 3), (4, 0)]:
        for s in [(0, 0), (1, 1), (2, 2)]:
            if s == g:
                continue
            info = _run(env, pol, config=(s, s, g))
            assert s not in env.forbidden, "s1==s2 yasak bolgeye dusmus"
            assert g not in env.forbidden, "goal yasak bolgeye dusmus"
            assert info["success"] or info["blocked"], info
    print("  test_exemption_same_start OK")


def test_forbidden_respected():
    """A2 hicbir kosulda yasak hucreye giremez."""
    env = MARLGridEnv(seed=1)
    pol = random_shortest_policy(np.random.default_rng(1))
    for ep in range(500):
        env.reset(seed=None)
        pol.reset()
        done = False
        while not done:
            if env.phase == 1:
                assert env.pos[AGENT_2] not in env.forbidden
            _, _, done, info = env.step(pol.act(env))
        if info["success"]:
            assert not (set(info["path2"]) & env.forbidden), "A2 yasak bolgeden gecmis"
            # kesisim sadece muaf hucrelerde olabilir
            common = set(info["path1"]) & set(info["path2"])
            assert common <= {env.s1, env.s2, env.goal}, common
    print("  test_forbidden_respected OK")


def test_phase_transition_and_early_stop():
    """A1 hedefe varinca yasak bolge sabitlenir; kilitliyse episode HEMEN biter."""
    env = MARLGridEnv(seed=2)
    # PLAN §0.3 kapali form: ilk hamle = son hamle = ASAGI -> kilitler
    env.reset(config=((0, 0), (0, 0), (4, 4)))
    blocking = [DOWN, DOWN, DOWN, RIGHT, RIGHT, RIGHT, RIGHT, DOWN]
    done = False
    for a in blocking:
        assert not done
        _, _, done, info = env.step(a)
    assert done and info["blocked"], info
    assert env.phase == 0, "kilitlenmede FAZ B'ye gecilmemeli"
    assert info["len2"] is None and not info["success"]
    assert not info["block_unavoidable"], "bu konfigde A1'in iyi alternatifi vardi"

    # ayni konfig, kibar yol: D D D D R R R R -> A2 gidebilmeli
    env.reset(config=((0, 0), (0, 0), (4, 4)))
    done = False
    for a in [DOWN, DOWN, DOWN, DOWN, RIGHT, RIGHT, RIGHT, RIGHT]:
        _, _, done, info = env.step(a)
    assert not done, "kibar yolda faz sinirinda bitmemeliydi"
    assert env.phase == 1 and env.forbidden
    print("  test_phase_transition_and_early_stop OK")


def test_timeout():
    """Faz basina adim limiti calisiyor.

    YENI SOZLESME: A1 FAZ A'yi hedefe varmadan tuketirse episode ARTIK BITMEZ.
    Yasak bolge A1'in KISMI izinden sabitlenir ve A2 yine de oynar. Eski
    davranista (episode hemen biterdi) A2, A1 basarisiz oldugu her episode'da
    tek bir egitim adimi bile goremiyordu.
    """
    env = MARLGridEnv(seed=3)
    env.reset(config=((0, 0), (0, 0), (4, 4)))
    done = False
    steps = 0
    while not done:                      # sag-sol gidip gel, hedefe hic varma
        _, _, done, info = env.step(RIGHT if steps % 2 == 0 else LEFT)
        steps += 1
    assert info["timeout"] and info["timeout1"], info
    assert steps > MAX_STEPS_PER_PHASE, (
        f"FAZ A timeout'unda episode bitmis, A2 hic oynayamamis (steps={steps})")
    print("  test_timeout OK")


def test_passive_obs_updates():
    """PLAN §2.2 golge NOOP'un can damari: FAZ A boyunca A2'nin gozlemi
    DEGISMELI. Degismezse VDN'in kredi kanali olusmaz ve VDN sessizce
    IQL'e dejenere olur — projedeki en sinsi bug."""
    env = MARLGridEnv(seed=4)
    env.reset(config=((0, 0), (4, 0), (4, 4)))
    seen = [env.observe(AGENT_2).copy()]
    for a in [RIGHT, RIGHT, RIGHT]:      # A1 saga ilerliyor
        env.step(a)
        seen.append(env.observe(AGENT_2).copy())
    diffs = [not np.array_equal(seen[i], seen[i + 1]) for i in range(len(seen) - 1)]
    assert all(diffs), "FAZ A'da A2'nin gozlemi guncellenmiyor!"
    # ozellikle yasak bolge kanali (3) buyumeli
    forb_counts = [o[3 * 25:4 * 25].sum() for o in seen]
    assert forb_counts == sorted(forb_counts) and forb_counts[-1] > forb_counts[0]
    print("  test_passive_obs_updates OK")


def test_random_walk_no_crash():
    """Maskeye uyan tamamen rastgele politika ile ortam patlamamali."""
    env = MARLGridEnv(seed=5)
    rng = np.random.default_rng(5)
    for _ in range(2000):
        env.reset()
        done = False
        while not done:
            _, _, done, info = env.step(random_walk_act(env, rng))
        assert info["success"] or info["blocked"] or info["timeout"]
    print("  test_random_walk_no_crash OK")


def test_random_shortest_baseline(n_ep=20_000):
    """ISTATISTIK KABUL KRITERI (PLAN §0.3).

    A1 optimal yollarindan uniform rastgele birini secerse, konfig-agirlikli
    beklenti: kilitlenme %0.82, A2'nin zarar gormesi (kilit VEYA uzama) %13.3.
    baselines/scan.py bu sayilari tam olarak hesapliyor; ortam onlari
    simulasyonla yeniden uretmeli.
    """
    env = MARLGridEnv(seed=7)
    pol = random_shortest_policy(np.random.default_rng(7))
    blocked = harmed = success = gap1_bad = 0
    for _ in range(n_ep):
        info = _run(env, pol)
        blocked += info["blocked"]
        harmed += info["harmed"]
        success += info["success"]
        if info["gap1"] not in (None, 0):
            gap1_bad += 1
    p_block = 100 * blocked / n_ep
    p_harm = 100 * harmed / n_ep
    print(f"    {n_ep} episode: kilit %{p_block:.2f} (bekl. 0.82), "
          f"zarar %{p_harm:.2f} (bekl. 13.28), basari %{100*success/n_ep:.2f}")
    assert gap1_bad == 0, "random-shortest politikasi A1'i optimal tutmali"
    assert success + blocked == n_ep, "timeout olmamali"
    assert abs(p_block - 0.82) < 0.35, f"kilitleme orani sapti: {p_block}"
    assert abs(p_harm - 13.28) < 1.2, f"zarar orani sapti: {p_harm}"
    print("  test_random_shortest_baseline OK")


if __name__ == "__main__":
    print("tests/test_env.py")
    test_shapes_and_masks()
    test_exemption_same_start()
    test_forbidden_respected()
    test_phase_transition_and_early_stop()
    test_timeout()
    test_passive_obs_updates()
    test_random_walk_no_crash()
    test_random_shortest_baseline()
    print("TUM ORTAM TESTLERI GECTI ✓")
