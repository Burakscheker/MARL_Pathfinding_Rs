# MARL Pathfinding

50x50 gridde iki ajanlı sıralı yol bulma, Multi-Agent RL (IQL / VDN / QMIX).
İsteğe bağlı statik engel (duvar) zorluk modları: `easy` / `medium` / `hard`.

Her episode'da `start1`, `start2` ve **ortak** `goal` rastgele seçilir.
Önce Ajan 1 hedefe gider; geçtiği hücreler **yasak bölge** olur; sonra Ajan 2
o bölgeye girmeden aynı hedefe gider. Ajan 1 kendi optimalliğinden ödün
vermeden Ajan 2'ye yer bırakmayı öğrenmelidir.

Tam plan, ölçülmüş istatistikler ve aşama aşama yol haritası: **[PLAN.md](PLAN.md)**

## Algoritma nasıl çalışıyor (pseudo kod)

### 1. Episode akışı

```
episode_oynat(s1, s2, hedef, zorluk):
    duvarlar = duvar_kur(s1, hedef) + duvar_kur(s2, hedef)   # 5 / 7 / 11 hücre
    if BFS ile iki ajan da hedefe varamıyorsa:
        duvarlar = {}                     # çözümsüz config üretme
    yasak_bölge = {}

    # ---- FAZ A: sadece A1 hareket eder ----
    while A1 hedefe varmadı ve faz_adımı < 140:
        a = A1.aksiyon_seç(gözlem(A1))
        A1'i hareket ettir
        A2 "gölge NOOP" basar             # hareket etmez ama Q değeri hesaplanır

    yasak_bölge = A1'in geçtiği hücreler - {s1, s2, hedef}

    # ---- FAZ B: sadece A2 hareket eder ----
    while A2 hedefe varmadı ve faz_adımı < 140:
        a = A2.aksiyon_seç(gözlem(A2))    # duvarlara VE yasak_bölgeye giremez
        A2'yi hareket ettir
```

A1 zaman aşımına uğrasa bile FAZ B başlar (yasak bölge A1'in kısmi izinden
sabitlenir) — aksi halde A1 başarısız olduğu her episode'da A2 hiç eğitim
verisi göremezdi.

### 2. Ajanın gözlemi (`OBS_DIM = 898`)

```
gözlem(ajan) = [
    21x21 pencere : etraftaki duvarlar + yasak bölge,
    21x21 pencere : kendi geçtiği hücreler,
    11 skaler     : ajan_id, faz, zaman, kendi konumu,
                    hedefe düz-çizgi farkı, diğer ajana fark, ...,
    BFS_kendi_mesafe,                     # hedefe GERÇEK (engel-farkında) uzaklık
    BFS_fark[yukarı, sağ, aşağı, sol]     # +1 = bir adım yakınlaştırır
                                          # -1 = uzaklaştırır / engel
]
```

Son 5 skaler kritik: onlarsız ajan yalnızca **düz-çizgi** yön bilgisi görüyor
ve engelin etrafından hangi taraftan dolaşacağını bilemiyordu (ölçüldü:
adımların %17.8'inde düz-çizgi yönü gerçek optimal yönle çelişiyor).

### 3. Ödül

```
her adım                    : -0.05          # acele et
duvara/yasak hücreye hamle  : -0.10
daha önce geçtiği hücre     : -0.05          # git-gel önleme
kendi hedefine varış        : +10
İKİSİ de vardı              : +30            # TAKIM
süre doldu                  : -10
A2 kilitlendi (BFS)         : -3             # TAKIM
A2 optimalden uzun gitti    : -0.5 x fazla_adım   # TAKIM

+ her adımda potansiyel-tabanlı shaping:
      20 x (γ·Φ(s') - Φ(s)),  Φ = 1 - BFS_mesafe/max_mesafe
  (optimal politikayı değiştirmez — Ng, Harada, Russell 1999)
```

### 4. Öğrenme çekirdeği (üçünde de aynı: Double DQN)

```
her adımda:
    tekrar_belleğine_yaz(gözlem, aksiyon, ödül, sonraki_gözlem, maske)

her 8 adımda bir:
    32 rastgele geçmiş deneyim çek
    en_iyi_a' = argmax Q_online(sonraki, ·)           # aksiyonu ONLINE seçer
    hedef     = ödül + γ · Q_target(sonraki, en_iyi_a')   # değeri TARGET biçer
    kayıp     = (Q_online(gözlem, aksiyon) - hedef)²
    gradyan adımı + gradyan kırpma
```

### 5. Üç algoritmanın TEK farkı: TD hatası kime yayılıyor

```
IQL   : Q1 ve Q2 tamamen AYRI eğitilir, her biri kendi r_ind'iyle.
        r_ind = adım maliyeti + kendi hedef bonusu (TAKIM cezaları YOK).
        -> A1, "A2'yi engelledim" sinyalini hiç görmez.

VDN   : Q_toplam = Q1(o1,a1) + Q2(o2,a2)
        kayıp = (Q_toplam - TAKIM_ödülü)²
        -> TEK hata ikisine BİRDEN yayılır, A1 kilitleme cezasını hisseder.

QMIX  : Q_toplam = Mixer(Q1, Q2 | global_durum)
        Mixer ağırlıkları hypernetwork ile global durumdan üretilir,
        abs() ile pozitif tutulur (monotonluk garantisi).
        -> VDN'in "sadece toplama" kısıtını gevşetir.
```

**Gölge NOOP neden önemli:** FAZ A boyunca A2 hareket etmez ama her adımda
gözlem üretip NOOP basar, Q değeri toplama girer. Böylece "A1 şu hücreye
girince A2'nin durumu kötüleşti" bilgisi gradyanla A1'e geri akabilir.
IQL'de bu kanal yapısal olarak yoktur — projenin test ettiği asıl hipotez budur.

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
