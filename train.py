"""Egitim giris noktasi.

Asama 3:  .venv\\Scripts\\python.exe train.py --algo dqn
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import deque

import numpy as np

# Windows'ta stdout bir dosyaya/boruya yonlendirilince (>, |) konsol degil
# sistem ANSI codepage'i (cp1252) kullaniliyor; Turkce karakterler orada
# gecersiz. UTF-8'e sabitleyip bu siniftaki tum betiklerde tekrarini onle.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from agents.dqn import DQNAgent
from agents.qmix import QMixAgent
from agents.vdn import VDNAgent
from config import (AGENT_1, AGENT_2, DEMO_EPISODES, DEMO_SEED,
                    DIFFICULTY_CSV, DQN_EPISODES, DQN_EVAL_EVERY, GRID_N,
                    IQL_BATCH, IQL_BUFFER, IQL_EPISODES, IQL_EPS_DECAY_STEPS,
                    IQL_EVAL_EVERY, IQL_LEARN_START, IQL_LR,
                    IQL_TARGET_UPDATE, QMIX_EPISODES, QMIX_EVAL_EVERY,
                    RUNS_DIR, SEED, TRAIN_HARM_LOG_EVERY, TRAIN_HARM_WINDOW,
                    VDN_EVAL_EVERY, VDN_EPISODES)
from env.grid_env import MARLGridEnv
from env.sampler import CurriculumSampler
from env.single_agent import SingleAgentEnv, all_start_goal_pairs
from env.two_agent import play_episode, play_episode_qmix, play_episode_vdn


# --------------------------------------------------------------------- eval

def evaluate_dqn(agent: DQNAgent, env: SingleAgentEnv) -> dict:
    """600 (start, goal) ciftinin TAMAMINDA deterministik greedy degerlendirme.

    Orneklem degil tam tarama: "gap 0.0" iddiasi boylece kanit olur.
    """
    gaps, fails = [], []
    for s, g in all_start_goal_pairs():
        obs = env.reset(config=(s, s, g))      # s2 = s1 (tek ajan icin ilgisiz)
        done = False
        while not done:
            a = agent.act(obs, env.action_mask(), eps=0.0)
            obs, _, done, info = env.step(a)
        if not info["reached"] or info["gap1"] != 0:
            fails.append((s, g, info["gap1"] if info["reached"] else "timeout"))
        gaps.append(info["gap1"] if info["reached"] else None)

    ok = [x for x in gaps if x is not None]
    return {
        "n": len(gaps),
        "reached": len(ok),
        "mean_gap": float(np.mean(ok)) if ok else float("nan"),
        "optimal_frac": sum(x == 0 for x in ok) / len(gaps),
        "fails": fails,
    }


# -------------------------------------------------------------------- train

def train_dqn(episodes: int = DQN_EPISODES, seed: int = SEED,
              log_every: int = 1_000, eval_every: int = DQN_EVAL_EVERY) -> DQNAgent:
    env = SingleAgentEnv(seed=seed)
    agent = DQNAgent(seed=seed)

    os.makedirs(f"{RUNS_DIR}/ckpt", exist_ok=True)
    log_path = f"{RUNS_DIR}/dqn_train_log.csv"
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    logger = csv.DictWriter(log_file, fieldnames=[
        "episode", "steps", "eps", "return", "len", "gap", "loss",
        "eval_optimal_frac", "eval_mean_gap"])
    logger.writeheader()

    ret_w, gap_w, len_w, loss_w = (deque(maxlen=500) for _ in range(4))
    t0 = time.time()

    for ep in range(1, episodes + 1):
        obs, done, ep_ret = env.reset(), False, 0.0
        while not done:
            mask = env.action_mask()
            a = agent.act(obs, mask)
            next_obs, r, done, info = env.step(a)
            next_mask = env.action_mask()
            # Zaman limitinde episode biter ama bootstrap DEVAM etmeli.
            # done=True yazmak "15. adimda hedefe uzaksan degerin 0" demektir;
            # gercek deger ~0.7'dir ve bu hata tam da ajanin kayboldugu
            # durumlara enjekte edilir. Klasik time-limit bootstrapping tuzagi.
            agent.push(obs, a, r, next_obs, done and not info["truncated"], next_mask)
            loss = agent.learn()
            if loss is not None:
                loss_w.append(loss)
            obs, ep_ret = next_obs, ep_ret + r

        ret_w.append(ep_ret)
        len_w.append(info["len1"])
        gap_w.append(info["gap1"] if info["reached"] else np.nan)

        row = None
        if ep % eval_every == 0:
            ev = evaluate_dqn(agent, env)
            row = {"eval_optimal_frac": round(ev["optimal_frac"], 4),
                   "eval_mean_gap": round(ev["mean_gap"], 4)}
            print(f"  [ep {ep:6d}] EVAL 600 cift: optimal %{100*ev['optimal_frac']:.1f}"
                  f"  ort.gap {ev['mean_gap']:+.3f}"
                  f"  basarisiz {len(ev['fails'])}", flush=True)

        if ep % log_every == 0 or row:
            mean_gap = float(np.nanmean(gap_w)) if len(gap_w) else float("nan")
            entry = {
                "episode": ep, "steps": agent.steps, "eps": round(agent.eps, 4),
                "return": round(float(np.mean(ret_w)), 3),
                "len": round(float(np.mean(len_w)), 3),
                "gap": round(mean_gap, 4),
                "loss": round(float(np.mean(loss_w)), 5) if loss_w else "",
                "eval_optimal_frac": "", "eval_mean_gap": "",
            }
            entry.update(row or {})
            logger.writerow(entry)
            log_file.flush()
            if row is None:
                print(f"  ep {ep:6d} | eps {agent.eps:.3f} | odul {entry['return']:+.2f}"
                      f" | uzunluk {entry['len']:.2f} | gap {mean_gap:+.3f}"
                      f" | loss {entry['loss']}", flush=True)

    log_file.close()
    ckpt = f"{RUNS_DIR}/ckpt/dqn.pt"
    agent.save(ckpt)

    print(f"\nEgitim bitti: {time.time()-t0:.0f}s, {agent.steps} adim -> {ckpt}")
    ev = evaluate_dqn(agent, env)
    print("\n=== ASAMA 3 KABUL KRITERI ===")
    print(f"  600 (start, goal) ciftinin tamami, deterministik greedy (eps=0)")
    print(f"  hedefe ulasan        : {ev['reached']}/{ev['n']}")
    print(f"  optimal (gap == 0)   : {100*ev['optimal_frac']:.2f}%")
    print(f"  ortalama gap         : {ev['mean_gap']:+.4f}   (hedef: 0.0000)")
    if ev["fails"]:
        print(f"  BASARISIZ {len(ev['fails'])} cift, ilk 10: {ev['fails'][:10]}")
        print("  -> KABUL KRITERI GECMEDI, Asama 4'e gecme.")
    else:
        print("  TUM 600 CIFT OPTIMAL  BFS ile birebir ayni. Asama 4'e gecilebilir.")
    return agent


# ==================================================================== IQL

def load_all_configs(n_samples: int = 1_000) -> tuple[list, list]:
    """Kucuk N'de (5x5 gibi) Asama 2'nin TAM tarama dosyasindan (difficulty.csv)
    KESIN kovalar okunur — IQL'in Asama 4 kabul kriteri bu sekilde Asama 2'nin
    dogruluk zeminiyle AYNI konfig uzayina karsi olculur.

    BUYUK N'de (100x100) tam tarama IMKANSIZ (14.400 konfig -> 10^12'ye cikar,
    baselines/scan.py calisamaz) — bunun yerine CurriculumSampler'in orneklem-
    tabanli havuzlarindan (grid_env.py'nin is_hard esigiyle AYNI kural)
    n_samples kadar konfig + etiket uretilir. Kesin degil ama enumerate
    gerektirmez.
    """
    if GRID_N <= 20 and os.path.exists(DIFFICULTY_CSV):
        configs, difficulty = [], []
        with open(DIFFICULTY_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                s1 = (int(row["s1_r"]), int(row["s1_c"]))
                s2 = (int(row["s2_r"]), int(row["s2_c"]))
                g = (int(row["g_r"]), int(row["g_c"]))
                configs.append((s1, s2, g))
                difficulty.append(row["difficulty"])
        return configs, difficulty

    sampler = CurriculumSampler(seed=12345, total_episodes=1,
                                grid_n=GRID_N, n_samples=n_samples)
    configs = sampler.hard + sampler.easy_nontrivial + sampler.easy_trivial
    difficulty = (["hard"] * len(sampler.hard)
                 + ["easy"] * (len(sampler.easy_nontrivial) + len(sampler.easy_trivial)))
    return configs, difficulty


def evaluate_iql(agents: dict, env: MARLGridEnv, configs: list,
                 difficulty: list | None = None) -> dict:
    """Verilen konfig listesinde deterministik greedy (eps=0) degerlendirme.

    PLAN §0.3: ana metrik "kilitleme" degil "zarar" (kilit VEYA uzama) —
    kilitleme konfig-agirlikli sadece %0.82'de, 2000 eval'de ~1 tane gorulur.
    PLAN §Asama 6: metrikler kolay/zor/genel ayri raporlanir — genel ortalama
    %70.8'lik "bedava" kolay kesimle sulanip curriculum'un etkisini gizler.
    """
    n = len(configs)
    reached1 = reached2 = blocked = detoured = harmed = gap1_bad = success = 0
    gap2_sum = 0.0
    hard_harmed = hard_n = 0
    easy_harmed = easy_n = 0

    for i, cfg in enumerate(configs):
        info, _ = play_episode(env, agents, train=False, config=cfg)
        if info["gap1"] is not None:
            reached1 += 1
            gap1_bad += info["gap1"] != 0
        if info["gap2"] is not None:
            reached2 += 1
            gap2_sum += info["gap2"]
        blocked += info["blocked"]
        detoured += bool(info.get("detoured"))
        harmed += bool(info.get("harmed"))
        success += bool(info.get("success"))
        if difficulty is not None:
            if difficulty[i] == "hard":
                hard_n += 1
                hard_harmed += bool(info.get("harmed"))
            else:
                easy_n += 1
                easy_harmed += bool(info.get("harmed"))

    return {
        "n": n,
        "success_rate": success / n,
        "reached1_frac": reached1 / n,
        "reached2_frac": reached2 / n,
        "gap1_bad": gap1_bad,
        "mean_gap2": gap2_sum / max(reached2, 1),
        "block_rate": blocked / n,
        "detour_rate": detoured / n,
        "harm_rate": harmed / n,
        "hard_harm_rate": (hard_harmed / hard_n) if hard_n else float("nan"),
        "hard_n": hard_n,
        "easy_harm_rate": (easy_harmed / easy_n) if easy_n else float("nan"),
        "easy_n": easy_n,
    }


def _make_iql_agent(seed: int) -> DQNAgent:
    return DQNAgent(seed=seed, buffer_size=IQL_BUFFER, batch_size=IQL_BATCH,
                    lr=IQL_LR, eps_decay_steps=IQL_EPS_DECAY_STEPS,
                    learn_start=IQL_LEARN_START, target_update=IQL_TARGET_UPDATE)


def train_iql(episodes: int = IQL_EPISODES, seed: int = SEED,
              eval_every: int = IQL_EVAL_EVERY,
              quick_eval_n: int = 200, tag: str = "iql",
              obstacle_difficulty: str | None = None) -> dict:
    """Iki bagimsiz DQN, ortak odul YOK. PLAN §Asama 4.

    grid_env.py'deki info["r_ind"] zaten sadece step-cost + kendi hedef
    bonusunu iceriyor; kilitleme/takim/optimallik cezalari (r_team'e ait)
    hic karismiyor — bu IQL'in "ortak odul yok" tanimini otomatik saglar.

    tag: cikti dosyalarinin onekini belirler (runs/ckpt/{tag}_agent*.pt,
    runs/{tag}_train_log.csv, runs/{tag}_train_harm.csv). Varsayilan "iql"
    Asama 4'un kanonik 40.000-episode kosusuyla ayni dosyalari kullanir —
    farkli episode sayisiyla ayri bir kosu yapacaksan (orn. gosterim/rapor
    icin) FARKLI bir tag ver, yoksa belgelenmis sonuclarin uzerine yazilir.

    Konfigler UNIFORM ornekleniyor (env.sample_config()) — zorluk-agirlikli
    curriculum KALDIRILDI, bkz. train_vdn'deki ayni not.

    obstacle_difficulty: None (varsayilan, engelsiz) / "easy"/"medium"/"hard"
    -- MARLGridEnv'e statik duvar zorlugu olarak gecirilir (bkz. config.py
    WALL_WIDTHS). "difficulty" (tekil) adini KASITLI KULLANMADIK: bu
    fonksiyonda zaten mesafe-tabanli hard/easy ETIKETLERI (all_difficulty,
    quick_difficulty) var, isim karismasin diye.
    """
    env = MARLGridEnv(seed=seed, difficulty=obstacle_difficulty)
    agents = {AGENT_1: _make_iql_agent(seed), AGENT_2: _make_iql_agent(seed + 1)}

    all_configs, all_difficulty = load_all_configs()
    rng = np.random.default_rng(seed + 999)
    quick_idx = rng.choice(len(all_configs), size=quick_eval_n, replace=False)
    quick_configs = [all_configs[i] for i in quick_idx]
    quick_difficulty = [all_difficulty[i] for i in quick_idx]

    os.makedirs(f"{RUNS_DIR}/ckpt", exist_ok=True)
    log_path = f"{RUNS_DIR}/{tag}_train_log.csv"
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    logger = csv.DictWriter(log_file, fieldnames=[
        "episode", "steps1", "steps2", "eps1", "eps2", "eval_success_rate",
        "eval_reached1", "eval_reached2",
        "eval_gap1_bad", "eval_mean_gap2", "eval_block_rate",
        "eval_detour_rate", "eval_harm_rate", "eval_hard_harm_rate"])
    logger.writeheader()

    # Egitim SIRASINDAKI (epsilon-greedy) zarar orani — yogun hareketli
    # ortalama. DIKKAT: bu politikanin GERCEK kalitesini degil, epsilon +
    # politika karisimini olcer; epsilon dustukce "iyilesiyormus gibi"
    # gorunebilir sadece rastgele hamle azaldigi icin. Politikanin GERCEK
    # zarar orani logger'daki eval_harm_rate'tir (eps=0, tam/orneklem tarama).
    harm_path = f"{RUNS_DIR}/{tag}_train_harm.csv"
    harm_file = open(harm_path, "w", newline="", encoding="utf-8")
    harm_logger = csv.DictWriter(harm_file, fieldnames=[
        "episode", "train_harm_rate", "train_harm_rate_hard", "n_hard_in_window"])
    harm_logger.writeheader()
    harm_w = deque(maxlen=TRAIN_HARM_WINDOW)
    hard_harm_w = deque(maxlen=TRAIN_HARM_WINDOW)

    t0 = time.time()
    for ep in range(1, episodes + 1):
        cfg = None          # uniform ornekleme (curriculum kaldirildi)
        info, _ = play_episode(env, agents, train=True, config=cfg)
        harm_w.append(bool(info.get("harmed")))
        if info.get("is_hard"):
            hard_harm_w.append(bool(info.get("harmed")))

        if ep % TRAIN_HARM_LOG_EVERY == 0:
            harm_logger.writerow({
                "episode": ep,
                "train_harm_rate": round(float(np.mean(harm_w)), 4),
                "train_harm_rate_hard": (round(float(np.mean(hard_harm_w)), 4)
                                        if hard_harm_w else ""),
                "n_hard_in_window": len(hard_harm_w),
            })
            harm_file.flush()

        if ep % eval_every == 0 or ep == episodes:
            ev = evaluate_iql(agents, env, quick_configs, quick_difficulty)
            row = {
                "episode": ep,
                "steps1": agents[AGENT_1].steps, "steps2": agents[AGENT_2].steps,
                "eps1": round(agents[AGENT_1].eps, 4),
                "eps2": round(agents[AGENT_2].eps, 4),
                "eval_success_rate": round(ev["success_rate"], 4),
                "eval_reached1": round(ev["reached1_frac"], 4),
                "eval_reached2": round(ev["reached2_frac"], 4),
                "eval_gap1_bad": ev["gap1_bad"],
                "eval_mean_gap2": round(ev["mean_gap2"], 4),
                "eval_block_rate": round(ev["block_rate"], 4),
                "eval_detour_rate": round(ev["detour_rate"], 4),
                "eval_harm_rate": round(ev["harm_rate"], 4),
                "eval_hard_harm_rate": round(ev["hard_harm_rate"], 4),
            }
            logger.writerow(row)
            log_file.flush()
            print(f"  [ep {ep:6d}] eps1 {agents[AGENT_1].eps:.3f}"
                  f"  eps2 {agents[AGENT_2].eps:.3f}"
                  f"  A1-vardi %{100*ev['reached1_frac']:.1f}"
                  f"  basari %{100*ev['success_rate']:.1f}"
                  f"  A1-kotu {ev['gap1_bad']}/{quick_eval_n}"
                  f"  A2-gap {ev['mean_gap2']:+.3f}"
                  f"  kilit %{100*ev['block_rate']:.2f}"
                  f"  zarar %{100*ev['harm_rate']:.2f}"
                  f"  zarar(zor) %{100*ev['hard_harm_rate']:.2f}", flush=True)

        if ep % 1000 == 0:
            # Ara-checkpoint: run yarida kesilirse (ornegin elle durdurulursa)
            # SON tam episode'daki agirliklar diskte kalsin, bellekte kaybolmasin.
            # AYNI dosyaya yazar (final ile CAKISIR) — normal bitiste zaten
            # asagidaki son save() bunun UZERINE yazacak, ekstra dosya yok.
            agents[AGENT_1].save(f"{RUNS_DIR}/ckpt/{tag}_agent1.pt")
            agents[AGENT_2].save(f"{RUNS_DIR}/ckpt/{tag}_agent2.pt")

    log_file.close()
    harm_file.close()
    agents[AGENT_1].save(f"{RUNS_DIR}/ckpt/{tag}_agent1.pt")
    agents[AGENT_2].save(f"{RUNS_DIR}/ckpt/{tag}_agent2.pt")
    print(f"\nEgitim bitti: {time.time()-t0:.0f}s -> runs/ckpt/{tag}_agent{{1,2}}.pt")

    print("\n=== ASAMA 4 KABUL KRITERI (TAM 14.400 konfig) ===")
    ev = evaluate_iql(agents, env, all_configs, all_difficulty)
    print(f"  A1 optimal degil (gap1!=0)   : {ev['gap1_bad']}/{ev['n']}  (hedef: 0)")
    print(f"  A2 ortalama gap (ORACLE'a gore): {ev['mean_gap2']:+.4f}  (hedef: ~0.0)")
    print(f"  kilitleme orani               : %{100*ev['block_rate']:.2f}"
          f"   (random-shortest baseline: %0.82)")
    print(f"  ZARAR orani (genel)           : %{100*ev['harm_rate']:.2f}"
          f"   (random-shortest baseline: %13.28)")
    print(f"  ZARAR orani (kolay, n={ev['easy_n']})   : %{100*ev['easy_harm_rate']:.2f}")
    print(f"  ZARAR orani (zor alt-kume, n={ev['hard_n']}): %{100*ev['hard_harm_rate']:.2f}"
          f"   <-- VDN ile karsilastirilacak asil sayi")
    return {"agents": agents, "eval": ev}


def _record_demo_episodes(episode_fn, n: int, tag: str) -> list[dict]:
    """episode_fn() -> terminal info dict'i uretir (her cagrida env sifirlanmis
    olmali, "forbidden" alani DAHIL). Ortak govde: IQL ve VDN'in demo-kayit
    fonksiyonlari sadece episode_fn'i farkli kurar (play_episode vs
    play_episode_vdn, iki-ajan-dict vs tek-paylasilan-ajan)."""
    episodes = []
    for i in range(n):
        info = episode_fn()
        s1, s2, goal = info["config"]
        episodes.append({
            "idx": i, "s1": s1, "s2": s2, "goal": goal,
            "path1": info["path1"], "path2": info["path2"],
            "forbidden": info["forbidden"], "walls": info.get("walls", []),
            "success": info["success"], "blocked": info["blocked"],
            "detoured": info["detoured"], "harmed": info["harmed"],
            "len1": info["len1"], "len2": info["len2"],
            "oracle_len1": info["oracle_len1"], "oracle_len2": info["oracle_len2"],
            "free_len2": info["free_len2"], "is_hard": info["is_hard"],
        })

    out_path = f"{RUNS_DIR}/{tag}_demo_episodes.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(episodes, f, ensure_ascii=False, indent=2)
    n_harmed = sum(e["harmed"] for e in episodes)
    print(f"\n{n} deterministik gosterim episode'u -> {out_path}"
          f"  ({n_harmed}/{n} zarar gordu)")
    return episodes


def record_demo_episodes(agents: dict, n: int = DEMO_EPISODES, seed: int = DEMO_SEED,
                         tag: str = "iql", obstacle_difficulty: str | None = None) -> list[dict]:
    """Egitim SONRASI, epsilon=0 (tam deterministik, rastgele hamle YOK) N episode
    oynatir; her birinin tam yol/harita verisini JSON'a yazar (viz/plot_iql_report.py
    bunu okuyup harita cizer).

    Ayri bir MARLGridEnv kullanir: egitimin kendi rng akisina dokunmaz, ve
    DEMO_SEED sabit oldugu icin gosterim konfigleri egitim uzunlugundan
    bagimsiz olarak HER ZAMAN aynidir — kosular arasi karsilastirilabilir.

    obstacle_difficulty: egitimde kullanilanla AYNI verilmeli, yoksa gosterim
    haritasi duvarsiz (ya da farkli genislikte) cizilir — egitimle tutarsiz olur.
    """
    demo_env = MARLGridEnv(seed=seed, difficulty=obstacle_difficulty)

    def one():
        info, _ = play_episode(demo_env, agents, train=False)
        return {**info, "forbidden": sorted(demo_env.forbidden), "walls": sorted(demo_env.walls)}

    return _record_demo_episodes(one, n=n, tag=tag)


def record_demo_episodes_vdn(agent: VDNAgent, n: int = DEMO_EPISODES,
                             seed: int = DEMO_SEED, tag: str = "vdn",
                             obstacle_difficulty: str | None = None) -> list[dict]:
    """record_demo_episodes ile ayni sozlesme, tek paylasilan VDNAgent icin."""
    demo_env = MARLGridEnv(seed=seed, difficulty=obstacle_difficulty)

    def one():
        info, _ = play_episode_vdn(demo_env, agent, train=False)
        return {**info, "forbidden": sorted(demo_env.forbidden), "walls": sorted(demo_env.walls)}

    return _record_demo_episodes(one, n=n, tag=tag)


def record_demo_episodes_qmix(agent: QMixAgent, n: int = DEMO_EPISODES,
                              seed: int = DEMO_SEED, tag: str = "qmix",
                              obstacle_difficulty: str | None = None) -> list[dict]:
    """record_demo_episodes ile ayni sozlesme, QMixAgent icin."""
    demo_env = MARLGridEnv(seed=seed, difficulty=obstacle_difficulty)

    def one():
        info, _ = play_episode_qmix(demo_env, agent, train=False)
        return {**info, "forbidden": sorted(demo_env.forbidden), "walls": sorted(demo_env.walls)}

    return _record_demo_episodes(one, n=n, tag=tag)


# ==================================================================== VDN

def evaluate_vdn(agent: VDNAgent, env: MARLGridEnv, configs: list,
                 difficulty: list | None = None) -> dict:
    """evaluate_iql ile ayni sozlesme/metrikler (kolay/zor/genel dahil), tek
    paylasilan ajan icin."""
    n = len(configs)
    reached1 = reached2 = blocked = detoured = harmed = gap1_bad = success = 0
    gap2_sum = 0.0
    hard_harmed = hard_n = 0
    easy_harmed = easy_n = 0

    for i, cfg in enumerate(configs):
        info, _ = play_episode_vdn(env, agent, train=False, config=cfg)
        if info["gap1"] is not None:
            reached1 += 1
            gap1_bad += info["gap1"] != 0
        if info["gap2"] is not None:
            reached2 += 1
            gap2_sum += info["gap2"]
        blocked += info["blocked"]
        detoured += bool(info.get("detoured"))
        harmed += bool(info.get("harmed"))
        success += bool(info.get("success"))
        if difficulty is not None:
            if difficulty[i] == "hard":
                hard_n += 1
                hard_harmed += bool(info.get("harmed"))
            else:
                easy_n += 1
                easy_harmed += bool(info.get("harmed"))

    return {
        "n": n,
        "success_rate": success / n,
        "reached1_frac": reached1 / n,
        "reached2_frac": reached2 / n,
        "gap1_bad": gap1_bad,
        "mean_gap2": gap2_sum / max(reached2, 1),
        "block_rate": blocked / n,
        "detour_rate": detoured / n,
        "harm_rate": harmed / n,
        "hard_harm_rate": (hard_harmed / hard_n) if hard_n else float("nan"),
        "hard_n": hard_n,
        "easy_harm_rate": (easy_harmed / easy_n) if easy_n else float("nan"),
        "easy_n": easy_n,
    }


def train_vdn(episodes: int = VDN_EPISODES, seed: int = SEED,
             eval_every: int = VDN_EVAL_EVERY,
             quick_eval_n: int = 200, tag: str = "vdn",
             obstacle_difficulty: str | None = None) -> dict:
    """Paylasilan TEK Q-agi, golge NOOP ile baglanmis TEK TD hatasi. PLAN §Asama 5.

    IQL'den (train_iql) TEK mimari fark play_episode_vdn'de: HER t'de HER IKI
    ajanin (obs,aksiyon) cifti ayni joint transition'a yaziliyor, boylece
    A1'in "kilitleme" hatasi artik A1'in gradyanina ULASIYOR — IQL'de
    ulasmiyordu (agents/dqn.py'nin push()'u pasif ajan icin hic cagrilmiyordu).

    KONFIG ORNEKLEME: uniform (env.sample_config()) — her episode ayni zorluk
    dagilimindan cekilir. Zorluk-agirlikli curriculum (zor konfig payini egitim
    boyunca %20->%80 yukselten PLAN §Asama 6 mekanizmasi) KALDIRILDI: ajanin
    hedefe varamadigi bir donemde gorevi giderek zorlastirmak, ayni anda
    dusen epsilon'la birleserek cifte sikistirma yaratiyordu.

    obstacle_difficulty: bkz. train_iql'deki ayni parametrenin notu (statik
    duvar zorlugu, mesafe-tabanli difficulty listeleriyle KARISTIRILMASIN).
    """
    env = MARLGridEnv(seed=seed, difficulty=obstacle_difficulty)
    agent = VDNAgent(seed=seed)

    all_configs, all_difficulty = load_all_configs()
    rng = np.random.default_rng(seed + 999)
    quick_idx = rng.choice(len(all_configs), size=quick_eval_n, replace=False)
    quick_configs = [all_configs[i] for i in quick_idx]
    quick_difficulty = [all_difficulty[i] for i in quick_idx]

    os.makedirs(f"{RUNS_DIR}/ckpt", exist_ok=True)
    log_path = f"{RUNS_DIR}/{tag}_train_log.csv"
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    logger = csv.DictWriter(log_file, fieldnames=[
        "episode", "steps", "eps", "eval_success_rate",
        "eval_reached1", "eval_reached2", "eval_gap1_bad",
        "eval_mean_gap2", "eval_block_rate", "eval_detour_rate",
        "eval_harm_rate", "eval_hard_harm_rate"])
    logger.writeheader()

    harm_path = f"{RUNS_DIR}/{tag}_train_harm.csv"
    harm_file = open(harm_path, "w", newline="", encoding="utf-8")
    harm_logger = csv.DictWriter(harm_file, fieldnames=[
        "episode", "train_harm_rate", "train_harm_rate_hard", "n_hard_in_window"])
    harm_logger.writeheader()
    harm_w = deque(maxlen=TRAIN_HARM_WINDOW)
    hard_harm_w = deque(maxlen=TRAIN_HARM_WINDOW)

    health_checked = False  # PLAN §Asama 5 saglik kontrolu, bkz. asagida

    t0 = time.time()
    for ep in range(1, episodes + 1):
        # d(s1,g)=1 olan konfigler (%13.3) FAZ A'yi TEK adimda bitirir — o
        # episode'u yakalarsak 1 olcumle "SABIT" diye yanlis alarm veririz.
        # ep>=200'den itibaren en az 3 olcum toplayan ILK episode'u kullan.
        probe = [] if (not health_checked and ep >= 200) else None
        cfg = None          # uniform ornekleme (curriculum kaldirildi)
        info, _ = play_episode_vdn(env, agent, train=True, config=cfg, health_probe=probe)

        if probe is not None and len(probe) >= 3:
            health_checked = True
            spread = max(probe) - min(probe)
            verdict = "OK — degisiyor" if spread > 1e-4 else "UYARI: SABIT — golge NOOP kopuk olabilir!"
            print(f"  [saglik kontrolu, ep{ep}] Q(obs_2,NOOP) FAZ A boyunca "
                  f"{len(probe)} olcum: min={min(probe):+.4f} max={max(probe):+.4f} "
                  f"fark={spread:.4f}  ({verdict})", flush=True)

        harm_w.append(bool(info.get("harmed")))
        if info.get("is_hard"):
            hard_harm_w.append(bool(info.get("harmed")))
        if ep % TRAIN_HARM_LOG_EVERY == 0:
            harm_logger.writerow({
                "episode": ep,
                "train_harm_rate": round(float(np.mean(harm_w)), 4),
                "train_harm_rate_hard": (round(float(np.mean(hard_harm_w)), 4)
                                        if hard_harm_w else ""),
                "n_hard_in_window": len(hard_harm_w),
            })
            harm_file.flush()

        if ep % eval_every == 0 or ep == episodes:
            ev = evaluate_vdn(agent, env, quick_configs, quick_difficulty)
            row = {
                "episode": ep, "steps": agent.steps, "eps": round(agent.eps, 4),
                "eval_success_rate": round(ev["success_rate"], 4),
                "eval_reached1": round(ev["reached1_frac"], 4),
                "eval_reached2": round(ev["reached2_frac"], 4),
                "eval_gap1_bad": ev["gap1_bad"],
                "eval_mean_gap2": round(ev["mean_gap2"], 4),
                "eval_block_rate": round(ev["block_rate"], 4),
                "eval_detour_rate": round(ev["detour_rate"], 4),
                "eval_harm_rate": round(ev["harm_rate"], 4),
                "eval_hard_harm_rate": round(ev["hard_harm_rate"], 4),
            }
            logger.writerow(row)
            log_file.flush()
            print(f"  [ep {ep:6d}] eps {agent.eps:.3f}"
                  f"  A1-vardi %{100*ev['reached1_frac']:.1f}"
                  f"  basari %{100*ev['success_rate']:.1f}"
                  f"  A1-kotu {ev['gap1_bad']}/{quick_eval_n}"
                  f"  A2-gap {ev['mean_gap2']:+.3f}"
                  f"  kilit %{100*ev['block_rate']:.2f}"
                  f"  zarar %{100*ev['harm_rate']:.2f}"
                  f"  zarar(zor) %{100*ev['hard_harm_rate']:.2f}", flush=True)

        if ep % 1000 == 0:
            # Ara-checkpoint: run yarida kesilirse SON tam episode'daki
            # agirliklar diskte kalsin. Ayni dosya, normal bitiste UZERINE yazilir.
            agent.save(f"{RUNS_DIR}/ckpt/{tag}.pt")

    log_file.close()
    harm_file.close()
    agent.save(f"{RUNS_DIR}/ckpt/{tag}.pt")
    print(f"\nEgitim bitti: {time.time()-t0:.0f}s -> runs/ckpt/{tag}.pt")

    print("\n=== ASAMA 5 KABUL KRITERI (TAM 14.400 konfig) ===")
    ev = evaluate_vdn(agent, env, all_configs, all_difficulty)
    print(f"  A1 optimal degil (gap1!=0)   : {ev['gap1_bad']}/{ev['n']}  (hedef: 0)")
    print(f"  A2 ortalama gap (ORACLE'a gore): {ev['mean_gap2']:+.4f}  (hedef: ~0.0)")
    print(f"  kilitleme orani               : %{100*ev['block_rate']:.2f}"
          f"   (IQL: %0.84, random-shortest: %0.82)")
    print(f"  ZARAR orani (genel)           : %{100*ev['harm_rate']:.2f}"
          f"   (IQL: %12.91, hedef: <%2)")
    print(f"  ZARAR orani (kolay, n={ev['easy_n']})   : %{100*ev['easy_harm_rate']:.2f}")
    print(f"  ZARAR orani (zor alt-kume, n={ev['hard_n']}): %{100*ev['hard_harm_rate']:.2f}"
          f"   (IQL: %42.71, hedef: <%13.3 <-- VDN gercekten calisiyor mu kaniti)")
    return {"agent": agent, "eval": ev}


# ==================================================================== QMIX

def evaluate_qmix(agent: QMixAgent, env: MARLGridEnv, configs: list,
                  difficulty: list | None = None) -> dict:
    """evaluate_vdn ile ayni sozlesme/metrikler, QMixAgent icin."""
    n = len(configs)
    reached1 = reached2 = blocked = harmed = gap1_bad = success = 0
    gap2_sum = 0.0
    hard_harmed = hard_n = easy_harmed = easy_n = 0

    for i, cfg in enumerate(configs):
        info, _ = play_episode_qmix(env, agent, train=False, config=cfg)
        if info["gap1"] is not None:
            reached1 += 1
            gap1_bad += info["gap1"] != 0
        if info["gap2"] is not None:
            reached2 += 1
            gap2_sum += info["gap2"]
        blocked += info["blocked"]
        harmed += bool(info.get("harmed"))
        success += bool(info.get("success"))
        if difficulty is not None:
            if difficulty[i] == "hard":
                hard_n += 1
                hard_harmed += bool(info.get("harmed"))
            else:
                easy_n += 1
                easy_harmed += bool(info.get("harmed"))

    return {
        "n": n,
        "success_rate": success / n,
        "reached1_frac": reached1 / n,
        "reached2_frac": reached2 / n,
        "gap1_bad": gap1_bad,
        "mean_gap2": gap2_sum / max(reached2, 1),
        "block_rate": blocked / n,
        "harm_rate": harmed / n,
        "hard_harm_rate": (hard_harmed / hard_n) if hard_n else float("nan"),
        "hard_n": hard_n,
        "easy_harm_rate": (easy_harmed / easy_n) if easy_n else float("nan"),
        "easy_n": easy_n,
    }


def train_qmix(episodes: int = QMIX_EPISODES, seed: int = SEED,
              eval_every: int = QMIX_EVAL_EVERY,
              quick_eval_n: int = 200, tag: str = "qmix",
              obstacle_difficulty: str | None = None) -> dict:
    """VDN'in ustune monotonik mixer. PLAN §Asama 7.

    train_vdn ile TEK fark play_episode_qmix'te: her joint transition'a
    global state (env.state()) eklenir, mixer agirliklarini bundan uretir.
    Konfigler uniform ornekleniyor (curriculum kaldirildi, bkz. train_vdn).

    obstacle_difficulty: bkz. train_iql'deki ayni parametrenin notu.
    """
    env = MARLGridEnv(seed=seed, difficulty=obstacle_difficulty)
    agent = QMixAgent(seed=seed)

    all_configs, all_difficulty = load_all_configs()
    rng = np.random.default_rng(seed + 999)
    quick_idx = rng.choice(len(all_configs), size=quick_eval_n, replace=False)
    quick_configs = [all_configs[i] for i in quick_idx]
    quick_difficulty = [all_difficulty[i] for i in quick_idx]

    os.makedirs(f"{RUNS_DIR}/ckpt", exist_ok=True)
    log_path = f"{RUNS_DIR}/{tag}_train_log.csv"
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    logger = csv.DictWriter(log_file, fieldnames=[
        "episode", "steps", "eps", "eval_success_rate",
        "eval_reached1", "eval_reached2", "eval_gap1_bad",
        "eval_mean_gap2", "eval_block_rate", "eval_harm_rate",
        "eval_hard_harm_rate"])
    logger.writeheader()

    harm_path = f"{RUNS_DIR}/{tag}_train_harm.csv"
    harm_file = open(harm_path, "w", newline="", encoding="utf-8")
    harm_logger = csv.DictWriter(harm_file, fieldnames=[
        "episode", "train_harm_rate", "train_harm_rate_hard", "n_hard_in_window"])
    harm_logger.writeheader()
    harm_w = deque(maxlen=TRAIN_HARM_WINDOW)
    hard_harm_w = deque(maxlen=TRAIN_HARM_WINDOW)

    health_checked = False

    t0 = time.time()
    for ep in range(1, episodes + 1):
        probe = [] if (not health_checked and ep >= 200) else None
        cfg = None          # uniform ornekleme (curriculum kaldirildi)
        info, _ = play_episode_qmix(env, agent, train=True, config=cfg, health_probe=probe)

        if probe is not None and len(probe) >= 3:
            health_checked = True
            spread = max(probe) - min(probe)
            verdict = "OK — degisiyor" if spread > 1e-4 else "UYARI: SABIT — golge NOOP kopuk olabilir!"
            print(f"  [saglik kontrolu, ep{ep}] Q(obs_2,NOOP) FAZ A boyunca "
                  f"{len(probe)} olcum: min={min(probe):+.4f} max={max(probe):+.4f} "
                  f"fark={spread:.4f}  ({verdict})", flush=True)

        harm_w.append(bool(info.get("harmed")))
        if info.get("is_hard"):
            hard_harm_w.append(bool(info.get("harmed")))
        if ep % TRAIN_HARM_LOG_EVERY == 0:
            harm_logger.writerow({
                "episode": ep,
                "train_harm_rate": round(float(np.mean(harm_w)), 4),
                "train_harm_rate_hard": (round(float(np.mean(hard_harm_w)), 4)
                                        if hard_harm_w else ""),
                "n_hard_in_window": len(hard_harm_w),
            })
            harm_file.flush()

        if ep % eval_every == 0 or ep == episodes:
            ev = evaluate_qmix(agent, env, quick_configs, quick_difficulty)
            row = {
                "episode": ep, "steps": agent.steps, "eps": round(agent.eps, 4),
                "eval_success_rate": round(ev["success_rate"], 4),
                "eval_reached1": round(ev["reached1_frac"], 4),
                "eval_reached2": round(ev["reached2_frac"], 4),
                "eval_gap1_bad": ev["gap1_bad"],
                "eval_mean_gap2": round(ev["mean_gap2"], 4),
                "eval_block_rate": round(ev["block_rate"], 4),
                "eval_harm_rate": round(ev["harm_rate"], 4),
                "eval_hard_harm_rate": round(ev["hard_harm_rate"], 4),
            }
            logger.writerow(row)
            log_file.flush()
            print(f"  [ep {ep:6d}] eps {agent.eps:.3f}"
                  f"  A1-vardi %{100*ev['reached1_frac']:.1f}"
                  f"  basari %{100*ev['success_rate']:.1f}"
                  f"  A1-kotu {ev['gap1_bad']}/{quick_eval_n}"
                  f"  A2-gap {ev['mean_gap2']:+.3f}"
                  f"  kilit %{100*ev['block_rate']:.2f}"
                  f"  zarar %{100*ev['harm_rate']:.2f}"
                  f"  zarar(zor) %{100*ev['hard_harm_rate']:.2f}", flush=True)

        if ep % 1000 == 0:
            # Ara-checkpoint: run yarida kesilirse SON tam episode'daki
            # agirliklar diskte kalsin. Ayni dosya, normal bitiste UZERINE yazilir.
            agent.save(f"{RUNS_DIR}/ckpt/{tag}.pt")

    log_file.close()
    harm_file.close()
    agent.save(f"{RUNS_DIR}/ckpt/{tag}.pt")
    print(f"\nEgitim bitti: {time.time()-t0:.0f}s -> runs/ckpt/{tag}.pt")

    print("\n=== ASAMA 7 KABUL KRITERI (TAM 14.400 konfig) ===")
    ev = evaluate_qmix(agent, env, all_configs, all_difficulty)
    print(f"  A1 optimal degil (gap1!=0)   : {ev['gap1_bad']}/{ev['n']}  (hedef: 0)")
    print(f"  A2 ortalama gap (ORACLE'a gore): {ev['mean_gap2']:+.4f}  (hedef: ~0.0)")
    print(f"  kilitleme orani               : %{100*ev['block_rate']:.2f}")
    print(f"  ZARAR orani (genel)           : %{100*ev['harm_rate']:.2f}")
    print(f"  ZARAR orani (kolay, n={ev['easy_n']})   : %{100*ev['easy_harm_rate']:.2f}")
    print(f"  ZARAR orani (zor alt-kume, n={ev['hard_n']}): %{100*ev['hard_harm_rate']:.2f}"
          f"   <-- §2.1 hipotezi: QMIX, VDN'i geciyor mu?")
    return {"agent": agent, "eval": ev}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="dqn", choices=["dqn", "iql", "vdn", "qmix"])
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--tag", default=None,
                    help="cikti dosyalarinin oneki (varsayilan: --algo degeri)")
    p.add_argument("--no-demo", action="store_true",
                    help="iql/vdn icin: egitim sonrasi 10-episode gosterim+grafik adimini atla")
    p.add_argument("--difficulty", default=None, choices=["easy", "medium", "hard"],
                    help="statik duvar zorlugu (bkz. config.py WALL_WIDTHS); "
                         "verilmezse duvarsiz (eski davranis)")
    args = p.parse_args()

    if args.algo == "dqn":
        episodes = args.episodes or DQN_EPISODES
        print(f"=== DQN egitimi | {episodes} episode | seed {args.seed} ===")
        train_dqn(episodes=episodes, seed=args.seed)
    elif args.algo == "iql":
        episodes = args.episodes or IQL_EPISODES
        tag = args.tag or "iql"
        print(f"=== IQL egitimi | {episodes} episode | seed {args.seed} | tag={tag}"
              f" | duvar={args.difficulty} ===")
        result = train_iql(episodes=episodes, seed=args.seed, tag=tag,
                           obstacle_difficulty=args.difficulty)
        if not args.no_demo:
            record_demo_episodes(result["agents"], tag=tag, obstacle_difficulty=args.difficulty)
            from viz.plot_iql_report import plot_demo_grids, plot_harm_curve
            os.makedirs(f"{RUNS_DIR}/viz", exist_ok=True)
            plot_harm_curve(tag, f"{RUNS_DIR}/viz/{tag}_harm_curve.png")
            plot_demo_grids(tag, f"{RUNS_DIR}/viz/{tag}_demo_grids.png")
    elif args.algo == "vdn":
        episodes = args.episodes or VDN_EPISODES
        tag = args.tag or "vdn"
        print(f"=== VDN egitimi | {episodes} episode | seed {args.seed} | tag={tag}"
              f" | duvar={args.difficulty} ===")
        result = train_vdn(episodes=episodes, seed=args.seed, tag=tag,
                           obstacle_difficulty=args.difficulty)
        if not args.no_demo:
            record_demo_episodes_vdn(result["agent"], tag=tag, obstacle_difficulty=args.difficulty)
            from viz.plot_iql_report import plot_demo_grids, plot_harm_curve
            os.makedirs(f"{RUNS_DIR}/viz", exist_ok=True)
            plot_harm_curve(tag, f"{RUNS_DIR}/viz/{tag}_harm_curve.png")
            plot_demo_grids(tag, f"{RUNS_DIR}/viz/{tag}_demo_grids.png")
    else:  # qmix
        episodes = args.episodes or QMIX_EPISODES
        tag = args.tag or "qmix"
        print(f"=== QMIX egitimi | {episodes} episode | seed {args.seed} | tag={tag}"
              f" | duvar={args.difficulty} ===")
        result = train_qmix(episodes=episodes, seed=args.seed, tag=tag,
                            obstacle_difficulty=args.difficulty)
        if not args.no_demo:
            record_demo_episodes_qmix(result["agent"], tag=tag, obstacle_difficulty=args.difficulty)
            from viz.plot_iql_report import plot_demo_grids, plot_harm_curve
            os.makedirs(f"{RUNS_DIR}/viz", exist_ok=True)
            plot_harm_curve(tag, f"{RUNS_DIR}/viz/{tag}_harm_curve.png")
            plot_demo_grids(tag, f"{RUNS_DIR}/viz/{tag}_demo_grids.png")
