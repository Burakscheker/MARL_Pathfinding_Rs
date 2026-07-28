"""DQN bilesenleri birim testleri — PLAN §Asama 3.

Calistir:  .venv\\Scripts\\python.exe -m tests.test_agents
"""
import numpy as np
import torch

from agents.buffer import ReplayBuffer
from agents.dqn import DQNAgent
from agents.networks import NEG_INF, MLP, masked_q
from config import DQN_LEARN_START, N_ACTIONS, OBS_DIM
from env.single_agent import SingleAgentEnv, all_start_goal_pairs


def test_mlp_shape():
    """MLP artik ana egitimde KULLANILMIYOR (build_qnet() CNNQNet donduruyor,
    bkz. agents/networks.py) — bu test sadece sinifin sekli/ileri-gecisi
    bozulmadi diye kalan bir dead-code duman testi. Parametre araligi
    eskiden 5x5'in kucuk OBS_DIM'ine (~129) gore 30-50k'ydi; OBS_DIM artik
    PATCH_SIZE tabanli (893), bu yuzden gercek sayi ~131k. Sabit bir alt/ust
    sinir yerine HIDDEN'dan turetilen dogru degerle kiyaslaniyor."""
    net = MLP(OBS_DIM, N_ACTIONS)
    assert net(torch.zeros(7, OBS_DIM)).shape == (7, N_ACTIONS)
    n_params = sum(p.numel() for p in net.parameters())
    from config import HIDDEN
    expected = ((OBS_DIM + 1) * HIDDEN + (HIDDEN + 1) * HIDDEN
               + (HIDDEN + 1) * N_ACTIONS)
    assert n_params == expected, (n_params, expected)
    print(f"  test_mlp_shape OK  ({n_params} parametre)")


def test_masked_q_no_nan():
    """Maskeli max NaN uretmemeli — NEG_INF yerine gercek -inf kullanilsaydi
    terminal gecislerde (-inf * 0) NaN cikar ve gradyan sessizce bozulurdu."""
    q = torch.randn(4, N_ACTIONS)
    mask = torch.zeros(4, N_ACTIONS)
    mask[:, 1] = 1.0                                  # sadece tek aksiyon acik
    m = masked_q(q, mask)
    assert (m[:, 0] == NEG_INF).all() and (m[:, 1] == q[:, 1]).all()
    assert m.argmax(dim=1).tolist() == [1, 1, 1, 1]

    # tamamen sifir maske: hepsi NEG_INF ama sonlu, (1-done) ile carpim guvenli
    allzero = masked_q(q, torch.zeros(4, N_ACTIONS))
    assert torch.isfinite(allzero).all()
    assert torch.isfinite(allzero.max(dim=1).values * 0.0).all()
    print("  test_masked_q_no_nan OK")


def test_buffer():
    buf = ReplayBuffer(capacity=10, obs_dim=4, n_actions=N_ACTIONS,
                       rng=np.random.default_rng(0))
    assert len(buf) == 0
    zero_mask = np.zeros(N_ACTIONS, dtype=np.float32)
    for i in range(15):                               # halka tamponu tasir
        buf.push(np.full(4, i), i % N_ACTIONS, 0.5, np.full(4, i + 1),
                 done=(i % 3 == 0), next_mask=zero_mask)
    assert len(buf) == 10, "kapasite asilinca len sabit kalmali"

    # done olan gecislerde next_mask 1'lerle doldurulmali
    done_rows = buf.next_mask[buf.done == 1.0]
    assert done_rows.size and (done_rows == 1.0).all(), \
        "terminal gecisin next_mask'i 1'lenmemis — masked max NEG_INF dondurur"

    obs, act, rew, nobs, done, nmask = buf.sample(6)
    assert obs.shape == (6, 4) and nmask.shape == (6, N_ACTIONS)
    assert act.dtype == np.int64
    print("  test_buffer OK")


def test_act_respects_mask():
    agent = DQNAgent(seed=0)
    obs = np.random.default_rng(0).random(OBS_DIM).astype(np.float32)
    mask = np.array([0, 1, 0, 1, 0], dtype=np.float32)
    for eps in (0.0, 1.0, 0.5):
        for _ in range(50):
            assert agent.act(obs, mask, eps=eps) in (1, 3), f"eps={eps}"
    print("  test_act_respects_mask OK")


def test_learn_runs_and_stays_finite():
    """push() her cagrida self.steps'i ilerletir, learn() ise SADECE
    steps % LEARN_EVERY == 0 iken gercekten ogrenir (agents/dqn.py). Bu
    yuzden push() ve learn()'u ARALIKLI cagirmak lazim — push()'lari once
    hepsini bitirip steps'i SABITLEYIP sonra learn()'u N kere cagirmak,
    steps LEARN_EVERY'ye bolunmuyorsa (ki push sayisina gore degisir) HER
    seferinde None doner ve test sessizce yanlis pozitif/negatif verir."""
    agent = DQNAgent(seed=0)
    rng = np.random.default_rng(0)
    assert agent.learn() is None, "LEARN_START'tan once ogrenmemeli"

    mask = np.ones(N_ACTIONS, dtype=np.float32)
    for i in range(DQN_LEARN_START):
        agent.push(rng.random(OBS_DIM).astype(np.float32), int(rng.integers(0, N_ACTIONS)),
                   float(rng.normal()), rng.random(OBS_DIM).astype(np.float32),
                   bool(i % 7 == 0), mask)

    losses = []
    for i in range(200):    # LEARN_EVERY ne olursa olsun en az birkac kez tetiklenir
        agent.push(rng.random(OBS_DIM).astype(np.float32), int(rng.integers(0, N_ACTIONS)),
                   float(rng.normal()), rng.random(OBS_DIM).astype(np.float32),
                   bool(i % 7 == 0), mask)
        losses.append(agent.learn())
    fired = [l for l in losses if l is not None]
    assert len(fired) > 0, "learn() bir kez bile gercek gradyan adimi atmadi"
    assert all(np.isfinite(l) for l in fired), fired[:5]
    assert all(torch.isfinite(p).all() for p in agent.online.parameters()), \
        "ogrenmeden sonra agirliklarda NaN/Inf"
    print(f"  test_learn_runs_and_stays_finite OK  ({len(fired)}/200 adimda ogrenildi)")


def test_single_agent_wrapper():
    env = SingleAgentEnv(seed=0)
    obs = env.reset(config=((0, 0), (0, 0), (0, 2)))
    assert obs.shape == (OBS_DIM,)
    # RIGHT, RIGHT -> hedefte, 2 adim, gap 0
    _, r1, done, _ = env.step(1)
    assert not done
    _, r2, done, info = env.step(1)
    assert done and info["reached"] and info["len1"] == 2 and info["gap1"] == 0
    assert info["truncated"] is False, "hedefe varis kesilme sayilmamali"
    assert r2 > r1, "hedef odulu son adimda gelmeli"
    # A2 hic devreye girmemeli: FAZ B'ye gecilmis olsa bile wrapper bitirir
    assert len(all_start_goal_pairs()) == 600
    print("  test_single_agent_wrapper OK")


def test_wrapper_timeout():
    env = SingleAgentEnv(seed=0)
    env.reset(config=((0, 0), (0, 0), (4, 4)))
    done, steps = False, 0
    while not done:                                   # sag-sol salinim
        _, _, done, info = env.step(1 if steps % 2 == 0 else 3)
        steps += 1
    assert info["timeout"] and not info["reached"], info
    # Kesilme gercek terminalden ayrilmali — train.py bunu bootstrap icin okur
    assert info["truncated"] is True, "timeout 'truncated' olarak isaretlenmemis"
    # maske episode bitse de fiziksel olarak gecerli kalmali (bootstrap icin)
    assert env.action_mask().sum() >= 2, "kesilmede maske degenere olmus"
    print("  test_wrapper_timeout OK")


if __name__ == "__main__":
    print("tests/test_agents.py")
    test_mlp_shape()
    test_masked_q_no_nan()
    test_buffer()
    test_act_respects_mask()
    test_learn_runs_and_stays_finite()
    test_single_agent_wrapper()
    test_wrapper_timeout()
    print("TUM AJAN TESTLERI GECTI")
