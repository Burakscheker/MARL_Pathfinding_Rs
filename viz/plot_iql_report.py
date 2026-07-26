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
    if n <= 20:
        # Kucuk gridde hucre-hucre cizgiler faydali; buyuk N'de (100x100)
        # 101 cizgi hem yavas hem gorsel olarak anlamsiz (her hucre ~piksel
        # boyutunda) — sadece dis cerceve birakiliyor.
        ax.set_xticks(range(n + 1))
        ax.set_yticks(range(n + 1))
        ax.grid(True, color="#cccccc", lw=0.6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
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


def plot_curriculum_comparison(tag_uniform: str, tag_curriculum: str, out_path: str):
    """PLAN §Asama 6 kabul kriteri: zor alt-kumede iki egitim egrisini YAN YANA
    ciz — uniform sampling'de gorunmeyen iyilesme curriculum'da gorunuyor mu?
    """
    def load_hard(tag):
        eps, rates = [], []
        with open(f"{RUNS_DIR}/{tag}_train_harm.csv", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["train_harm_rate_hard"]:
                    eps.append(int(row["episode"]))
                    rates.append(float(row["train_harm_rate_hard"]) * 100)
        return eps, rates

    eps_u, rates_u = load_hard(tag_uniform)
    eps_c, rates_c = load_hard(tag_curriculum)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(eps_u, rates_u, color="#4C72B0", lw=1.6,
            label=f"uniform sampling ({tag_uniform})")
    ax.plot(eps_c, rates_c, color="#55A868", lw=1.6,
            label=f"curriculum sampling ({tag_curriculum})")
    ax.axhline(13.28, color="gray", ls=":", lw=1.0,
               label="random-shortest genel taban çizgisi (%13.28)")
    ax.set_xlabel("egitim episode'u")
    ax.set_ylabel("ZOR alt-kumede zarar orani (%)  — egitim sirasi, hareketli ort.")
    ax.set_title("PLAN §Asama 6: curriculum, zor alt-kumede sinyali yogunlastiriyor mu?")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"yazildi: {out_path}")


def plot_algorithm_comparison(tags_labels: list[tuple[str, str]], out_path: str):
    """PLAN'daki nihai IQL vs VDN vs QMIX karsilastirma grafigi: HER algoritmanin
    DETERMINISTIK (eps=0) zor-alt-kume zarar oranini AYNI eksende ciz.

    tags_labels: [(tag, etiket), ...] — orn.
        [("iql", "IQL"), ("vdn_final", "VDN+curriculum"), ("qmix_final", "QMIX+curriculum")]
    Egitim-ici gurultulu egri DEGIL — sadece {tag}_train_log.csv'deki
    eval_hard_harm_rate (seyrek ama temiz) noktalari kullanilir.
    """
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    any_data = False
    for (tag, label), color in zip(tags_labels, colors):
        path = f"{RUNS_DIR}/{tag}_train_log.csv"
        if not os.path.exists(path):
            print(f"atlandi: {path} yok")
            continue
        eps, rates = [], []
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                eps.append(int(row["episode"]))
                rates.append(float(row["eval_hard_harm_rate"]) * 100)
        if eps:
            ax.plot(eps, rates, "o-", color=color, lw=2, ms=6, label=label)
            any_data = True
    if not any_data:
        print("hicbir tag icin veri bulunamadi, grafik yazilmadi")
        return
    ax.axhline(13.28, color="gray", ls=":", lw=1.2, label="random-shortest genel taban çizgisi (%13.28)")
    ax.set_xlabel("egitim episode'u")
    ax.set_ylabel("ZOR alt-kumede zarar orani (%)  — deterministik (eps=0)")
    ax.set_title("PLAN §2.1: IQL vs VDN vs QMIX — zor alt-kumede zarar orani")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"yazildi: {out_path}")


def plot_success_rate_stacked(tags_labels: list[tuple[str, str]], out_path: str):
    """VDN/IQL/QMIX icin USTLU ALTLI 3 ayri panel, TEK PNG: run (episode)
    sayisi arttikca BASARI oraninin (her iki ajan da hedefe vardi) degisimi.
    Hangi algoritma daha basarili, panellerin OZ olcegi ayni tutularak
    (paylasilan y ekseni) dogrudan karsilastirilabiliyor.

    tags_labels: [(tag, etiket), ...] — 3 algoritma bekleniyor ama sayi
    esnek (kac tag verilirse o kadar panel ciziliyor).
    """
    n_panels = len(tags_labels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(9, 3.2 * n_panels), sharex=False)
    axes = np.atleast_1d(axes)
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

    for ax, (tag, label), color in zip(axes, tags_labels, colors):
        path = f"{RUNS_DIR}/{tag}_train_log.csv"
        if not os.path.exists(path):
            ax.set_title(f"{label} — veri yok ({path})", fontsize=10)
            ax.axis("off")
            continue
        eps, rates = [], []
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("eval_success_rate"):
                    eps.append(int(row["episode"]))
                    rates.append(float(row["eval_success_rate"]) * 100)
        ax.plot(eps, rates, "o-", color=color, lw=2, ms=5)
        ax.set_ylim(0, 100)
        ax.set_ylabel("başarı (%)")
        ax.grid(alpha=0.3)
        final = f", son değer %{rates[-1]:.1f}" if rates else ""
        ax.set_title(f"{label}{final}", fontsize=11, loc="left")

    axes[-1].set_xlabel("eğitim episode'u")
    fig.suptitle("VDN / IQL / QMIX — eğitim boyunca başarı oranı (deterministik, ε=0)",
                fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"yazildi: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="iql_demo")
    p.add_argument("--compare-with", default=None,
                    help="verilirse --tag'i BU tag'e karsi curriculum karsilastirma grafigi ciz")
    p.add_argument("--final", action="store_true",
                    help="IQL vs VDN vs QMIX nihai karsilastirma grafigini ciz")
    args = p.parse_args()
    os.makedirs(f"{RUNS_DIR}/viz", exist_ok=True)
    if args.final:
        algos = [("iql_final", "IQL"),
                 ("vdn_final", "VDN+curriculum"),
                 ("qmix_final", "QMIX+curriculum")]
        plot_algorithm_comparison(algos, f"{RUNS_DIR}/viz/final_algorithm_comparison.png")
        plot_success_rate_stacked(algos, f"{RUNS_DIR}/viz/final_success_rate_stacked.png")
    elif args.compare_with:
        plot_curriculum_comparison(args.compare_with, args.tag,
                                   f"{RUNS_DIR}/viz/{args.tag}_vs_{args.compare_with}.png")
    else:
        plot_harm_curve(args.tag, f"{RUNS_DIR}/viz/{args.tag}_harm_curve.png")
        plot_demo_grids(args.tag, f"{RUNS_DIR}/viz/{args.tag}_demo_grids.png")
