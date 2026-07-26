# MARL Pathfinding

5x5 gridde iki ajanlı sıralı yol bulma, Multi-Agent RL (IQL / VDN / QMIX).

Her episode'da `start1`, `start2` ve **ortak** `goal` rastgele seçilir.
Önce Ajan 1 hedefe gider; geçtiği hücreler **yasak bölge** olur; sonra Ajan 2
o bölgeye girmeden aynı hedefe gider. Ajan 1 kendi optimalliğinden ödün
vermeden Ajan 2'ye yer bırakmayı öğrenmelidir.

Tam plan, ölçülmüş istatistikler ve aşama aşama yol haritası: **[PLAN.md](PLAN.md)**

## Kurulum

```bash
python3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Çalıştırma

Tüm komutlar proje kökünden çalıştırılır.

Tam konfigürasyon taraması (14.400 konfig, ~1 sn) — `runs/*.csv` üretir:

```bash
.venv\Scripts\python.exe -m baselines.scan
```

Testler:

```bash
.venv\Scripts\python.exe -m tests.test_oracle
```

```bash
.venv\Scripts\python.exe -m tests.test_env
```

## Durum

| Aşama | Durum |
|---|---|
| 0 Kurulum | ✅ |
| 1 Ortam `env/grid_env.py` | ✅ |
| 2 BFS oracle `baselines/` | ✅ |
| 3 Tek ajan DQN | ✅ (600/600 optimal, gap 0.0) |
| 4 IQL baseline | ✅ (zarar %12.9, baseline-seviyesinde çakılı — bkz. altta) |
| 5 VDN `agents/vdn.py` | 🟡 kod tamam (ayrı-ağ mimarisi, bkz. altta), final koşu bekleniyor |
| 6 Curriculum `env/sampler.py` | 🟡 kod tamam, ölçüm net kazanç göstermedi (bkz. PLAN §Aşama 6) |
| 7 QMIX `agents/qmix.py` | 🟡 kod tamam, final koşu bekleniyor |
| 8 Değerlendirme `eval/evaluate.py` | 🟡 kod tamam, tüm algoritmalar için final koşu bekleniyor |
| 9 Görselleştirme `viz/plot_iql_report.py` | 🟡 kod tamam |

## Tek ajan DQN (Aşama 3)

```bash
.venv\Scripts\python.exe train.py --algo dqn --episodes 30000
```

600 (start, goal) çiftinin **tamamında** (örneklem değil) deterministik
greedy: **600/600 hedefe ulaşıyor, ortalama gap 0.0000** — BFS ile birebir
aynı. ~200 saniye, 122k adım. Checkpoint: `runs/ckpt/dqn.pt`.

Yol boyunca bir Q-value divergence bulunup düzeltildi (target update 500→2000
adım, ayrı bir DQN LR'i 1e-4'e düşürüldü) — detay PLAN.md §Aşama 3.

## IQL baseline (Aşama 4)

```bash
.venv\Scripts\python.exe train.py --algo iql --episodes 40000
```

İki bağımsız DQN, ortak ödül yok. TAM 14.400 konfigde deterministik greedy:

| Metrik | IQL | Random-shortest baseline |
|---|---:|---:|
| A1 optimal değil | 0/14.400 | — |
| A2 own_gap2 (gerçek yasak bölgeye göre) | ort. +0.027, %98.8 optimal | — |
| Kilitleme | %0.84 | %0.82 |
| **Zarar oranı (genel)** | **%12.91** | %13.28 |
| **Zarar oranı (zor alt-küme)** | **%42.71** | ~%45.5 |

İki DQN de ayrı ayrı kusursuz ama **takım performansı hiç iyileşmiyor** —
A1'e A2'nin akıbetine dair hiçbir sinyal ödül fonksiyonunda yok, bu yüzden
zarar oranı random-shortest baseline'ıyla aynı seviyede kalıyor. "VDN neden
gerekli" sorusunun ölçülmüş cevabı — detay PLAN.md §Aşama 4.

## VDN (Aşama 5) ve Curriculum (Aşama 6)

```bash
.venv\Scripts\python.exe train.py --algo vdn --episodes 15000 --tag vdn_final --curriculum
```

`agents/vdn.py`: her ajan için **ayrı** Q ağı (paylaşımlı tek ağ ep~1750'de
öğrenip çöküyordu — detay dosya docstring'inde ve PLAN.md §Aşama 5'te),
tek TD hedefi (`Q_tot = Q_1+Q_2`) golge NOOP ile ikisine birden geri yayılıyor.
`--curriculum` bayrağı `env/sampler.py`'deki zorluk-ağırlıklı örneklemeyi açar.

Otomatik olarak üretir: `runs/{tag}_train_log.csv` (seyrek, deterministik eval),
`runs/{tag}_train_harm.csv` (yoğun, eğitim-içi), 10 deterministik gösterim
episode'u + iki PNG (`runs/viz/{tag}_harm_curve.png`, `..._demo_grids.png`).

## QMIX (Aşama 7)

```bash
.venv\Scripts\python.exe train.py --algo qmix --episodes 15000 --tag qmix_final --curriculum
```

`agents/qmix.py`: VDN'in toplamsal `Q_1+Q_2`'sini, `env.state()`'e koşullu
monotonik bir hypernetwork mixer'a (`abs(W)` ile non-negatif ağırlıklar)
çevirir. Per-ajan ağlar VDN'deki gibi yine ayrı.

## Final karşılaştırma (Aşama 8-9)

```bash
.venv\Scripts\python.exe -m eval.evaluate --vdn-tag vdn_final --qmix-tag qmix_final
.venv\Scripts\python.exe -m viz.plot_iql_report --final
```

İlki `runs/eval_report.md`'ye Random-shortest / Bencil BFS / Oracle / IQL /
VDN / QMIX'i TAM 14.400 konfigde tek tabloda yazar; ikincisi zor alt-kümede
üç öğrenen algoritmayı tek grafikte karşılaştırır.

## Ölçülmüş temel sayılar

Tam tarama (`baselines/scan.py`) ve 20.000 episode simülasyonu ile doğrulandı:

| | Değer |
|---|---:|
| Değerlendirilen konfigürasyon | 14.400 |
| Çözümsüz konfigürasyon | **0** |
| A2'nin optimal gidebildiği | %98.1 |
| A2'nin +2 uzamak zorunda kaldığı | %1.9 (hepsi A1'in tek yollu olduğu durumlar) |
| "Zor" konfig (A1'in seçimi önemli) | %29.2 |
| **Random-shortest baseline: A2 zarar oranı** | **%13.3** ← ana metrik |
| Random-shortest baseline: kilitleme | %0.82 |
| Random-shortest baseline: başarı | %99.16 |

## Ortam API'si

```python
from env.grid_env import MARLGridEnv

env = MARLGridEnv(seed=0)
obs = env.reset()                      # {0: (129,), 1: (129,)}
obs, r_team, done, info = env.step(action)   # r_team TEK skaler (VDN icin)
print(env.render())
```

- `env.action_mask(agent)` — aktif ajanda `NOOP` kapalı, pasif ajanda sadece `NOOP`
- `env.state()` — QMIX mixer için merkezi global state (102,)
- `info["r_ind"]` — IQL baseline'ı için ajan başına ödül
- `info` (terminal) — `success`, `blocked`, `harmed`, `detoured`, `gap1`, `gap2`,
  `oracle_len2`, `is_hard`, `path1`, `path2`
