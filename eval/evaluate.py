"""Final karsilastirma tablosu — PLAN §Asama 8.

TAM 14.400 konfigde: Random-shortest, Bencil BFS, Oracle (ust sinir), IQL
(Asama 4), VDN+curriculum (Asama 5-6). Ayni konfig uzayi, ayni metrikler —
adil karsilastirma. Ogrenmeyen (scripted) politikalar ile ogrenen ajanlari
TEK ortak run_episode() dongusunde calistiriyoruz, boylece iki ayri kod yolu
sapip yanlis kiyaslamaya yol acmaz.

Calistir:  .venv\\Scripts\\python.exe -m eval.evaluate
"""
import os

from agents.dqn import DQNAgent
from agents.qmix import QMixAgent
from agents.vdn import VDNAgent
from baselines.policies import random_shortest_policy, selfish_bfs_policy
from config import (AGENT_1, AGENT_2, GRID_N, IQL_BATCH, IQL_BUFFER,
                    IQL_EPS_DECAY_STEPS, IQL_LEARN_START, IQL_LR,
                    IQL_TARGET_UPDATE, RUNS_DIR)
from env.grid_env import MARLGridEnv
from train import load_all_configs   # kucuk/buyuk N icin tek yer (train.py)


def run_episode(env: MARLGridEnv, act_fn, config) -> dict:
    """act_fn(env, obs) -> aktif ajanin aksiyonu. TUM politika turleri
    (scripted + ogrenen) bu TEK dongudeen gecer."""
    obs = env.reset(config=config)
    done = False
    info: dict = {}
    while not done:
        a = act_fn(env, obs)
        obs, _, done, info = env.step(a)
    return info


def evaluate_policy(name: str, act_fn, env: MARLGridEnv, configs: list,
                    difficulty: list, on_reset=None) -> dict:
    n = len(configs)
    reached1 = reached2 = blocked = harmed = gap1_bad = success = 0
    gap1_sum = gap2_sum = 0.0
    hard_harmed = hard_n = easy_harmed = easy_n = 0

    for i, cfg in enumerate(configs):
        if on_reset:
            on_reset()
        info = run_episode(env, act_fn, cfg)
        if info.get("gap1") is not None:
            reached1 += 1
            gap1_sum += info["gap1"]
            gap1_bad += info["gap1"] != 0
        if info.get("gap2") is not None:
            reached2 += 1
            gap2_sum += info["gap2"]
        blocked += bool(info.get("blocked"))
        harmed += bool(info.get("harmed"))
        success += bool(info.get("success"))
        if difficulty[i] == "hard":
            hard_n += 1
            hard_harmed += bool(info.get("harmed"))
        else:
            easy_n += 1
            easy_harmed += bool(info.get("harmed"))

    return {
        "name": name, "n": n,
        "success_rate": success / n,
        "reached1_frac": reached1 / n,
        "reached2_frac": reached2 / n,
        "block_rate": blocked / n,
        "harm_rate": harmed / n,
        "easy_harm_rate": easy_harmed / max(easy_n, 1),
        "hard_harm_rate": hard_harmed / max(hard_n, 1),
        "gap1_bad": gap1_bad,
        # "dolanma" olcusu: hedefe VARDIKTAN SONRA ortalama kac adim fazla
        # atildi (0 = tam optimal). gap1_bad sadece ikili (optimal mi degil
        # mi) veriyordu, BUYUKLUK vermiyordu.
        "mean_gap1": gap1_sum / max(reached1, 1),
        "mean_gap2": gap2_sum / max(reached2, 1),
    }


def _make_iql_agent(seed: int) -> DQNAgent:
    return DQNAgent(seed=seed, buffer_size=IQL_BUFFER, batch_size=IQL_BATCH,
                    lr=IQL_LR, eps_decay_steps=IQL_EPS_DECAY_STEPS,
                    learn_start=IQL_LEARN_START, target_update=IQL_TARGET_UPDATE)


def main(vdn_tag: str = "vdn_final", qmix_tag: str = "qmix_final",
        iql_tag: str = "iql", n_eval: int = 2_000) -> list[dict]:
    env = MARLGridEnv(seed=0)
    configs, difficulty = load_all_configs()
    if len(configs) > n_eval:
        idx = __import__("numpy").random.default_rng(0).choice(
            len(configs), size=n_eval, replace=False)
        configs = [configs[i] for i in idx]
        difficulty = [difficulty[i] for i in idx]
    results = []

    scripted = [("Random-shortest", random_shortest_policy),
               ("Bencil BFS", selfish_bfs_policy)]
    if GRID_N <= 20:
        # Oracle (ust sinir) TUM optimal yollari enumerate eder — buyuk N'de
        # (100x100) imkansiz (bkz. baselines/bfs_oracle.py), sadece kucuk
        # gridlerde dahil edilir.
        from baselines.policies import oracle_policy
        scripted.append(("Oracle (ust sinir)", oracle_policy))

    for name, make_policy in scripted:
        policy = make_policy()
        act_fn = lambda env, obs, policy=policy: policy.act(env)
        r = evaluate_policy(name, act_fn, env, configs, difficulty, on_reset=policy.reset)
        results.append(r)
        print(f"  {name}: bitti")

    ckpt1 = f"{RUNS_DIR}/ckpt/{iql_tag}_agent1.pt"
    ckpt2 = f"{RUNS_DIR}/ckpt/{iql_tag}_agent2.pt"
    try:
        agents = {AGENT_1: _make_iql_agent(0), AGENT_2: _make_iql_agent(1)}
        agents[AGENT_1].load(ckpt1)
        agents[AGENT_2].load(ckpt2)
        act_fn = lambda env, obs, agents=agents: agents[env.active].act(
            obs[env.active], env.action_mask(env.active), eps=0.0)
        r = evaluate_policy(f"IQL ({iql_tag})", act_fn, env, configs, difficulty)
        results.append(r)
        print("  IQL: bitti")
    except FileNotFoundError:
        print(f"  ATLANDI: {ckpt1} yok")

    vdn_ckpt = f"{RUNS_DIR}/ckpt/{vdn_tag}.pt"
    try:
        vagent = VDNAgent(seed=0)
        vagent.load(vdn_ckpt)
        act_fn = lambda env, obs, vagent=vagent: vagent.act(
            env.active, obs[env.active], env.action_mask(env.active), eps=0.0)
        r = evaluate_policy(f"VDN+curriculum ({vdn_tag})", act_fn, env, configs, difficulty)
        results.append(r)
        print("  VDN: bitti")
    except FileNotFoundError:
        print(f"  ATLANDI: {vdn_ckpt} yok")

    qmix_ckpt = f"{RUNS_DIR}/ckpt/{qmix_tag}.pt"
    try:
        qagent = QMixAgent(seed=0)
        qagent.load(qmix_ckpt)
        act_fn = lambda env, obs, qagent=qagent: qagent.act(
            env.active, obs[env.active], env.action_mask(env.active), eps=0.0)
        r = evaluate_policy(f"QMIX ({qmix_tag})", act_fn, env, configs, difficulty)
        results.append(r)
        print("  QMIX: bitti")
    except FileNotFoundError:
        print(f"  ATLANDI: {qmix_ckpt} yok")

    lines = [
        "| Politika | Başarı | A1 vardı | A2 vardı | Kilitleme | Zarar (genel) | Zarar (kolay) | Zarar (zor) | A1 ort. gap | A2 ort. gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['name']} | %{100*r['success_rate']:.2f} "
            f"| %{100*r['reached1_frac']:.2f} | %{100*r['reached2_frac']:.2f} "
            f"| %{100*r['block_rate']:.2f} "
            f"| %{100*r['harm_rate']:.2f} | %{100*r['easy_harm_rate']:.2f} "
            f"| %{100*r['hard_harm_rate']:.2f} "
            f"| {r['mean_gap1']:+.3f} | {r['mean_gap2']:+.3f} |")
    table = "\n".join(lines)
    print("\n" + table)

    os.makedirs(RUNS_DIR, exist_ok=True)
    header = (f"# Asama 8 — Final Karşılaştırma ({len(configs)} örneklem konfig, "
             f"GRID_N={GRID_N})\n\n")
    with open(f"{RUNS_DIR}/eval_report.md", "w", encoding="utf-8") as f:
        f.write(header + table + "\n")
    print(f"\nyazildi: {RUNS_DIR}/eval_report.md")
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--vdn-tag", default="vdn_final")
    p.add_argument("--qmix-tag", default="qmix_final")
    p.add_argument("--iql-tag", default="iql")
    p.add_argument("--n-eval", type=int, default=2_000)
    args = p.parse_args()
    main(vdn_tag=args.vdn_tag, qmix_tag=args.qmix_tag, iql_tag=args.iql_tag,
        n_eval=args.n_eval)
