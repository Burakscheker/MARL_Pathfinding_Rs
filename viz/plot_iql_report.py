"""IQL egitim raporu: zarar orani egrisi + N deterministik episode haritasi.

Calistir:  .venv\\Scripts\\python.exe -m viz.plot_iql_report --tag iql_demo

train.py --algo iql zaten egitim sonunda bu iki fonksiyonu otomatik cagirir;
bu script sadece VAR OLAN CSV/JSON'lardan yeniden cizmek icin (orn. stil
degisikligi sonrasi replot) ayrica calistirilabilir.
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")          # headless: ekrana degil dosyaya ciz
import matplotlib.pyplot as plt
import numpy as np

from config import GRID_N, RUNS_DIR, TRAIN_HARM_WINDOW

C_A1, C_A2, C_FORBIDDEN, C_GOAL = "#4C72B0", "#DD8452", "#d9d9d9", "#2ca02c"
BASELINE_HARM_PCT = 13.28      # random-shortest baseline, PLAN §0.3


def plot_harm_curve(tag: str, out_path: str):
    """Egitim boyunca A2'nin zarar gorme orani: yogun egitim-ici egri +
    seyrek deterministik (eps=0) dogrulama noktalari, ayni eksende.
    """
    episodes, rates, rates_hard = [], [], []
    with open(f"{RUNS_DIR}/{tag}_train_harm.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            episodes.append(int(row["episode"]))
            rates.append(float(row["train_harm_rate"]) * 100)
            rates_hard.append(float(row["train_harm_rate_hard"]) * 100
                              if row["train_harm_rate_hard"] else np.nan)

    eval_eps, eval_rates = [], []
    try:
        with open(f"{RUNS_DIR}/{tag}_train_log.csv", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                eval_eps.append(int(row["episode"]))
                eval_rates.append(float(row["eval_harm_rate"]) * 100)
    except FileNotFoundError:
        pass

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(episodes, rates, color=C_A1, lw=1.4,
            label=f"egitim sirasi (epsilon-greedy, {TRAIN_HARM_WINDOW}-episode hareketli ort.)")
    ax.plot(episodes, rates_hard, color=C_A2, lw=1.1, ls="--",
            label="egitim sirasi, SADECE zor konfigler")
    if eval_rates:
        ax.plot(eval_eps, eval_rates, "o-", color="#55A868", lw=2, ms=6,
                label="deterministik degerlendirme (eps=0 — gercek politika)")
    ax.axhline(BASELINE_HARM_PCT, color="gray", ls=":", lw=1.2,
               label=f"random-shortest taban çizgisi (%{BASELINE_HARM_PCT})")
    ax.set_xlabel("egitim episode'u")
    ax.set_ylabel("A2'nin zarar görme oranı (%)  — kilitlendi VEYA uzadı")
    ax.set_title("IQL: A2'nin zarar görme oranı egitim boyunca")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"yazildi: {out_path}")


def _cell_center(cell):
    r, c = cell
    return c + 0.5, r + 0.5


def _draw_board(ax, ep: dict):
    n = GRID_N
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks(range(n + 1))
    ax.set_yticks(range(n + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, color="#cccccc", lw=0.6)
    for spine in ax.spines.values():
        spine.set_color("#999999")

    for cell in ep["forbidden"]:
        r, c = cell
        ax.add_patch(plt.Rectangle((c, r), 1, 1, color=C_FORBIDDEN, zorder=0))

    p1 = [_cell_center(c) for c in ep["path1"]]
    xs, ys = zip(*p1)
    ax.plot(xs, ys, "-o", color=C_A1, lw=2, ms=4, zorder=2)

    p2 = [_cell_center(c) for c in ep["path2"]]
    if len(p2) > 1:
        xs2, ys2 = zip(*p2)
        ax.plot(xs2, ys2, "-o", color=C_A2, lw=2, ms=4, zorder=3)
    else:
        ax.plot(*p2[0], "x", color=C_A2, ms=12, mew=3, zorder=3)

    ax.plot(*_cell_center(ep["goal"]), "*", color=C_GOAL, ms=18, zorder=4)
    ax.plot(*_cell_center(ep["s1"]), "s", color=C_A1, ms=10, mfc="none", mew=2, zorder=4)
    ax.plot(*_cell_center(ep["s2"]), "s", color=C_A2, ms=10, mfc="none", mew=2, zorder=4)

    if ep["blocked"]:
        status = "KİLİTLENDİ"
    elif ep["harmed"]:
        status = f"UZADI (+{ep['len2'] - ep['free_len2']})"
    else:
        status = "temiz"
    zor = " [ZOR]" if ep["is_hard"] else ""
    ax.set_title(f"#{ep['idx']+1} {status}{zor} | len1={ep['len1']} "
                f"len2={ep['len2']}", fontsize=9)


def plot_demo_grids(tag: str, out_path: str):
    """N deterministik (eps=0) episode'un tamamini kucuk panellerde ciz."""
    with open(f"{RUNS_DIR}/{tag}_demo_episodes.json", encoding="utf-8") as f:
        episodes = json.load(f)

    cols = 5
    rows = (len(episodes) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.5 * rows))
    axes = np.atleast_1d(axes).reshape(-1)
    for ax, ep in zip(axes, episodes):
        _draw_board(ax, ep)
    for ax in axes[len(episodes):]:
        ax.axis("off")

    handles = [
        plt.Line2D([0], [0], color=C_A1, marker="o", label="A1 yolu"),
        plt.Line2D([0], [0], color=C_A2, marker="o", label="A2 yolu"),
        plt.Rectangle((0, 0), 1, 1, color=C_FORBIDDEN, label="yasak bölge (A1 izi)"),
        plt.Line2D([0], [0], color=C_GOAL, marker="*", lw=0, ms=12, label="hedef"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
              bbox_to_anchor=(0.5, -0.01))
    n_harmed = sum(e["harmed"] for e in episodes)
    fig.suptitle(f"Egitim sonrasi {len(episodes)} deterministik episode "
                f"(epsilon=0, rastgele hamle yok) — {n_harmed}/{len(episodes)} zarar gordu",
                fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"yazildi: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="iql_demo")
    args = p.parse_args()
    os.makedirs(f"{RUNS_DIR}/viz", exist_ok=True)
    plot_harm_curve(args.tag, f"{RUNS_DIR}/viz/{args.tag}_harm_curve.png")
    plot_demo_grids(args.tag, f"{RUNS_DIR}/viz/{args.tag}_demo_grids.png")
