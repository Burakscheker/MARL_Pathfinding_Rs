# MARL Pathfinding — Proje Planı

> **5x5 grid, 2 ajan, Multi-Agent RL (VDN / QMIX / IQL).**
> Her episode'da `start1`, `start2` ve **ortak** `goal` rastgele seçilir.
> Başlangıçlar aynı da olabilir, farklı da.
> **Akış sıralıdır (turn-based):** önce Ajan 1 hedefe gider → geçtiği hücreler
> yasak bölge olarak sabitlenir → sonra Ajan 2 o bölgeye girmeden, en kısa
> yoldan aynı hedefe gider.
> **Ajan 1, Ajan 2'nin yolunu kilitlerse ceza alır.**

Durum: 📋 Planlama · Güncelleme: 2026-07-23

---

## 0. Ölçülmüş gerçekler (tahmin değil, taradım)

Plan yazmadan önce 5x5'teki **tüm 15.625 `(s1, s2, goal)` konfigürasyonunu** BFS
ile taradım. Aşağıdaki sayılar projenin tüm tasarım kararlarını belirliyor.

### 0.1 Konfigürasyon uzayı

| | Adet | Oran |
|---|---:|---:|
| Toplam `(s1, s2, g)` | 15.625 | |
| Trivial (`s1==g` veya `s2==g`) | 1.225 | 7.8% |
| **Değerlendirilen** | **14.400** | |
| Aynı başlangıç (`s1 == s2`) | 600 | 4.2% |

### 0.2 Çözülebilirlik — iyi haber

| Sonuç (A1 en iyi yolu seçerse) | Adet | Oran |
|---|---:|---:|
| A2 de optimal gidiyor | 14.120 | **98.1%** |
| A2 gidiyor ama +2 adım fazla | 280 | 1.9% |
| **Hiçbir A1 yolu kurtarmıyor** | **0** | **0.0%** |

👉 **Hiçbir konfigürasyon çözümsüz değil.** Sabit `(0,0)→(4,4)` senaryosundaki
"Hex kilitlenmesi" problemi, random goal gelince tamamen kayboldu. Fizibilite
filtresine gerek yok — her episode oynanabilir.

O 280 vakayı ayrıca inceledim: **%100'ü, A1'in tek bir en kısa yolu olduğu
(yani `s1` ile `g`'nin aynı satır/sütunda olduğu) durumlar.** A1'in seçme şansı
yok, dolayısıyla **bu vakalarda A1 cezalandırılmamalı.** Bu, §5'teki ödül
tasarımının temel kuralı.

### 0.3 Koordinasyon ne kadar gerekli — ve asıl sinyal nerede

⚠️ Burada **ağırlıklandırma** kritik. İki farklı "kilitleme oranı" var ve
karıştırırsan yanlış hedef kovalarsın:

| Ölçüm | Değer | Ne anlama geliyor |
|---|---:|---|
| **Yol**-ağırlıklı kilitleme | 5.6% | (konfig, yol) çiftleri üzerinden. Çok yollu (uzun `d1`) konfigler baskın. |
| **Konfig**-ağırlıklı kilitleme | **0.82%** | 🔑 **Eğitimde göreceğin gerçek oran** — episode'lar konfigi uniform çekiyor. |

Yani "%5.6" senin göreceğin sayı **değil**. Doğru sayı **%0.82**.
(Ortam bunu simülasyonla doğruladı: 20.000 episode → %0.84.)

**Bunun sonucu: sert kilitleme çok nadir. Asıl öğrenme sinyali "uzatma".**

| Random-shortest A1 politikası altında | Oran |
|---|---:|
| A2 tamamen **kilitlenir** | **0.82%** |
| A2 **uzamak zorunda kalır** (kilitlenmeden) | ~12.5% |
| **A2 zarar görür (kilit VEYA uzama)** | **13.3%** 🔑 |
| Başarı oranı (ölçüldü, 20k episode) | 99.16% |

Konfigürasyon bazında:

| | Adet | Oran |
|---|---:|---:|
| Kilitleme **mümkün** olan konfig | 668 | 4.6% |
| Uzatma **mümkün** olan konfig | 4.108 | 28.5% |
| İkisinden biri = "**zor**" | 4.200 | 29.2% |
| A1'in seçiminin hiç önemi yok = "kolay" | 10.200 | 70.8% |

👉 **İki tasarım sonucu:**

1. **Ana metrik "kilitleme oranı" değil, "A2'nin uzama oranı" olmalı.** Kilitleme
   %0.82'de; 100 eval episode'unda ortalama **1 tane** görürsün, hiçbir şey
   ölçemezsin. Uzama %13.3'te ve gerçek koordinasyon içeriğini taşıyor.
   Kilitleme yine de raporlanır — nadir ama yüksek bedelli olay.
2. **Ödülde ağır işi `R_OPT_GAP` yapıyor, `R_BLOCKED` değil.** Bu yüzden
   `R_OPT_GAP = −0.5` (tipik 2 adımlık uzama → −1.0, +3.0'lık takım bonusuna
   karşı anlamlı ağırlık). `config.py`'de bu şekilde ayarlandı.

**Çözüm (Aşama 6): zor konfigleri fazladan örnekle ve metrikleri kolay/zor alt
kümelerde ayrı raporla.** Ölçtüm: **en zor %10'luk dilimde zarar oranı %69.2** —
uniform'daki %13.3'e göre **5 kat sinyal yoğunlaşması**. Bu plandaki en önemli
tek karar.

### 0.4 Zorluk, A1'in hedefe uzaklığıyla artıyor

| `d(s1, g)` | konfig | kilitleme | **zarar (kilit+uzama)** |
|---:|---:|---:|---:|
| 1 | 1920 | 0.00% | **0.00%** |
| 2 | 2976 | 0.00% | 7.26% |
| 3 | 3264 | 0.00% | 11.89% |
| 4 | 2880 | 0.09% | 16.39% |
| 5 | 1920 | 0.42% | 21.63% |
| 6 | 960 | 3.42% | 27.17% |
| 7 | 384 | 11.25% | 32.44% |
| 8 | 96 | **33.45%** | **37.50%** |

Bu tablo hazır bir **curriculum** veriyor: kolay (d≤3) → orta (d 4-5) → zor
(d≥6). İki net çıkarım:

- **`d(s1,g)=1` olan 1920 konfigde (%13.3) öğrenilecek sıfır koordinasyon var**
  — A1 tek adımda hedefte, izi boş. Eğitimde bunları neredeyse tamamen kıs.
- **Kilitleme sadece `d1≥6`'da anlamlı** (960+384+96 = 1440 konfig). Kilitleme
  davranışını özellikle ölçmek istiyorsan eval setine bu dilimi ayrı koy.

### 0.5 Neden artık tabular Q-table olmaz

14.400 konfig × ajan pozisyonu × yasak bölge maskesi → tabular imkânsız.
Fonksiyon yaklaşımı (neural net) **zorunlu**. Yani VDN/QMIX tercihi burada
"havalı olsun diye" değil, **gerçekten gerekli**. Projenin gerekçesi sağlam.

---

## 1. Formal tanım

### Ortam
- Grid 5x5, hücre `(satır, sütun)` ∈ `0..4`.
- Aksiyonlar: `0=YUKARI, 1=SAĞ, 2=AŞAĞI, 3=SOL, 4=NOOP` (5 aksiyon).
  `NOOP` sırası **gelmemiş** ajanın gölge aksiyonudur (§2.2) — sırası gelen
  ajan için maskelenir, çünkü beklemek asla optimal değil.
- Duvara / yasak hücreye hamle: ajan yerinde kalır + küçük ceza (episode bitmez).
- `reset()` her episode `s1`, `s2`, `g` rastgele çeker (`s1==s2` serbest,
  `s ≠ g` şartı ile).

### Yürütme modu: **sıralı (turn-based)** — tek mod

Episode tek bir zaman ekseni üzerinde iki fazdan oluşur:

```
t = 0 ................ T1        T1+1 ................ T
    |<--- FAZ A: A1 ---->|        |<---- FAZ B: A2 ---->|
     A1 hareket eder                A2 hareket eder
     A2 NOOP (gölge)                A1 NOOP (gölge)
                        ^
                        └─ yasak bölge B burada SABİTLENİR
```

**Bu seçimin iki büyük avantajı var:**

1. **BFS oracle tam kesin.** Eşzamanlı modda oracle sadece bir üst sınır
   verebilirdi; burada `ORACLE(s1,s2,g)` matematiksel olarak kesin optimum.
   Yani "ajan optimal mi" sorusunu **kanıtlayarak** cevaplıyorsun. RL
   projelerinde nadir bir lüks.
2. **IQL'in çöküşü çok net görünür.** Sıralı modda Ajan 1, Faz A boyunca
   Ajan 2'nin akıbetine dair **hiçbir** kendi-ödülü almaz. IQL'de bu sinyal
   fiziksel olarak yoktur → A1 sonsuza kadar %13.3 oranında zarar verir. VDN'de
   ise ortak TD hatası A1'in ağına geri akar. **Aradaki fark eşzamanlı
   moddakinden daha keskin çıkar** — karşılaştırma tablon daha güçlü olur.

Ödediğin bedel: VDN'i sıralı akışa doğru bağlamak biraz dikkat istiyor. Nasıl
yapılacağı §2.2'de.

Çarpışma kuralı, `BEKLE` koordinasyonu, eşzamanlı hareket gibi konular bu modda
**tamamen ortadan kalkıyor** — ortam belirgin şekilde basitleşiyor.

### Yasak bölge kuralı
```python
B(t) = set(A1'in t anına kadar ziyaret ettigi hucreler) - {s1, s2, g}
```
> ⚠️ **`s1`, `s2` ve `g` mutlaka muaf.** Aksi halde `s1==s2` olan 600 konfigde
> A2 daha ilk adımda kendi başlangıcında yasak hücrede olur ve problem tanım
> gereği çözümsüz olur. Ortamın 1 numaralı birim testi bu.

### Amaç
```
1. A1: len(P1) = manhattan(s1, g)                      → A1 optimal kalsın
2. A2: len(P2) = ORACLE(s1, s2, g)                     → A2 mümkün olan en kısa
3. Kısıt: P1 ∩ P2 ⊆ {s1, s2, g}
```
`ORACLE(s1,s2,g)` = A1'in tüm optimal yolları üzerinden A2'nin ulaşabileceği
**minimum** uzunluk. §0.2'deki 280 vakada bu, serbest Manhattan'dan 2 fazladır —
ve A1'in suçu değildir.

---

## 2. MARL algoritma yığını

Üçünü de yaz, karşılaştır. Rapor/sunumun omurgası bu tablo olacak.

| Algoritma | Fikir | Bu problemde beklentim |
|---|---|---|
| **IQL** (Independent Q-Learning) | Her ajan kendi DQN'i, diğerini ortamın parçası sanar. Ayrıştırma yok. | Baseline. Nonstationarity yüzünden zor konfiglerde tıkanmalı — "neden VDN lazım"ı kanıtlar. |
| **VDN** | `Q_tot = Q_1 + Q_2`. Takım ödülünü toplamsal ayrıştırır, CTDE. | Ana model. Credit assignment'ı çözer: "A2'yi kilitleyen A1'di" sinyalini A1'in Q'suna taşır. |
| **QMIX** | `Q_tot = f(Q_1, Q_2 \| s)`, monotonik mixing network + hypernetwork. | VDN'den iyi olmalı — çünkü bu problemde ödül **toplamsal değil** (§2.1). |

### 2.1 Hipotez: VDN burada yetersiz kalmalı, QMIX toparlamalı

VDN `Q_tot = Q_1 + Q_2` varsayar, yani toplam değer ajanların katkılarının
**toplamı**dır. Ama bizim ödülümüz böyle değil: A1 iyi bir yol seçtiyse A2'nin
katkısı yüksek, kötü seçtiyse A2 **ne yaparsa yapsın** sıfır. Yani A2'nin değeri
A1'in seçimine **çarpımsal/koşullu** bağlı — toplamsal değil.

QMIX'in monotonik mixing ağı bu etkileşimi temsil edebilir, VDN edemez.

👉 **Bu, raporunun tezi olabilir:** "5x5 blokaj probleminde VDN'in additivity
varsayımı ihlal ediliyor; QMIX zor konfig alt kümesinde VDN'i X puan geçiyor."
Ölçülebilir, güzel bir bulgu. **Ölçmen gereken yer §0.3'teki %29.2'lik zor alt
küme** — kolay konfiglerde üçü de aynı çıkar, fark orada görünmez.

### 2.2 VDN sıralı modda nasıl çalışır — "gölge NOOP" tasarımı

Sıralı akışın tek zorluğu şu: VDN `Q_tot = Q_1 + Q_2` der, ama ajanlar aynı
timestep'te oynamıyor. Çözüm standart ve basit:

> **Her timestep'te iki ajanın da ağı sorgulanır. Sırası gelmeyen ajan zorunlu
> `NOOP` basar, ama Q değeri toplama yine dahil edilir.**

```python
# her t icin
q1 = net(obs1)[a1]        # Faz A'da gercek aksiyon, Faz B'de NOOP
q2 = net(obs2)[a2]        # Faz A'da NOOP,          Faz B'de gercek aksiyon
q_tot = q1 + q2
loss  = (r_team + γ · max q_tot_target − q_tot) ** 2
```

**Neden bu doğru VDN'dir:** VDN'in tek teorik şartı IGM (Individual-Global-Max),
yani merkezi argmax'ın ajan-başına argmax'lara ayrışması. Toplamsal ayrıştırmada
bu her zaman sağlanır — ajanların **aynı anda oynaması şart değil.** Sıralı
Dec-POMDP tamamen geçerli bir VDN ortamıdır.

**Neden bu tam istediğimiz şeyi yapıyor:** Faz A boyunca `Q_2(obs_2, NOOP)`,
"A1'in şu ana kadar bıraktığı yasak bölgeyle ben ne kadar iyi durumdayım"ı
tahmin eden bir değer fonksiyonuna dönüşür. TD zinciri faz sınırını geçerek
geriye aktığı için, bu değer **Faz A'daki A1 aksiyonlarının gradyanına ulaşır.**
Yani A1, henüz A2 hiç kımıldamadan, "bu hamle A2'yi köşeye sıkıştırıyor"
sinyalini alır. IQL'de böyle bir kanal **yok** — §2.1'in kanıtı bu.

**Kritik implementasyon detayı — erken sonlandırma:**
Faz sınırında yasak bölge belli olur olmaz BFS ile A2'nin çözülebilirliğini
kontrol et. Kilitliyse episode'u **oracak yerde bitir**, `−3.0` cezasını hemen
ver. Böylece cezanın A1'in ilk hamlesine ulaşması için gereken discount zinciri
`γ^(T1+T2)` yerine `γ^T1` olur — kredi ataması belirgin şekilde hızlanır ve
boşuna 15 adım simüle etmezsin.

γ hesabı: T1+T2 ≤ 16 (tipik), `γ=0.99` → `0.99^16 ≈ 0.85`. Terminal ceza A1'in
ilk hamlesine %85 gücüyle ulaşıyor. Discount sorunu yok, γ'yı düşürme.

**Yedek plan (gölge NOOP buglanırsa):** "faz-ayrık VDN" — iki ayrı TD zinciri,
sadece terminal `r_team`'i paylaşırlar. Yazması daha kolay ama kredi kanalı
zayıf (A1 sadece terminal skaleri görür, `Q_2` üzerinden bootstrap yok).
Gölge NOOP çalışmıyorsa buna düş, ama rapora farkı not et.

### 2.3 Kütüphane kullanma, elle yaz

PyMARL/EPyMARL Windows'ta kurulum kâbusu (SMAC bağımlılıkları, eski gym
sürümleri). VDN ~40 satır, QMIX ~90 satır. Elle yaz:
- Hem ne olduğunu anlarsın (asıl amaç bu)
- Hem sunumda "kütüphane çağırdım" değil "implement ettim" dersin

Ortam API'sini yine de **PettingZoo `parallel_env` imzasına uygun** yaz —
ileride hazır bir kütüphaneye bağlamak istersen bedava gelir.

---

## 3. Gözlem ve ağ tasarımı

### Ajan gözlemi (her ajan için, 5x5x5 = 125 float + 3 skaler)

| Kanal | İçerik |
|---|---|
| 0 | Kendi konumu (one-hot grid) |
| 1 | Diğer ajanın konumu |
| 2 | Ortak hedef |
| 3 | Yasak bölge maskesi `B(t)` |
| 4 | Kendi ziyaret ettiği hücreler (geri dönmeyi caydırır) |

Ek skaler: `[agent_id, faz (0=A/1=B), t/max_steps, kalan_manhattan/8]`
→ toplam **129 girdi**.

> **`faz` biti şart:** gölge NOOP tasarımında (§2.2) ağ, "sıram geldi mi"yi
> bilmeden doğru Q üretemez. Parametre paylaşımı kullandığın için `agent_id` ile
> `faz` birlikte "aktif miyim"i belirler.

> **Neden grid kanalı, `(x,y)` koordinat değil:** koordinat verirsen ağ
> "yasak bölge nerede"yi öğrenemez — maske doğası gereği uzamsal. Kanal formatı
> ayrıca ileride CNN'e ve NxN genellemeye bedava geçiş sağlar.

### Ağ
```
MLP: 128 → 128 (ReLU) → 128 (ReLU) → 5 (Q değerleri)
```
Toplam ~35k parametre. CPU'da rahat eğitilir, GPU gerekmez.

**GRU gerekli mi?** Tam gözlemlenebilir tasarladık (yasak bölge gözlemde var),
o yüzden **hayır**. Kısmi gözlemlenebilirlik denemesi yapmak istersen (ajan
sadece 3x3 komşuluğunu görsün) o zaman GRU şart — Aşama 9 stretch.

### QMIX mixer
```
global state: 5x5x4 (iki konum + hedef + yasak bölge) + t  → 101
hypernet: state → mixing agirliklari (|W| ile monotonluk garantisi)
mixer:    (Q_1, Q_2), state → Q_tot
```

---

## 4. Aşamalar

Her aşamanın kabul kriteri var. **Kriter geçmeden sonrakine geçme.**

### ✅ Aşama 0 — Kurulum — TAMAM

- [x] venv: `numpy 2.5.1`, `matplotlib 3.11.1`, **`torch 2.13.0+cpu`**
      (`cp314` wheel'i sorunsuz kuruldu, `cuda=False` — CPU yeterli)
- [x] Klasör iskeleti (Bölüm 6)
- [x] `.gitignore` (`.venv/`, `__pycache__/`, `runs/`)
- [x] `config.py` — tüm hiperparametreler tek dosyada
- [x] `git init` + ilk commit + push →
      [gulergokhan/MARL_PATHFINDING](https://github.com/gulergokhan/MARL_PATHFINDING)

---

### ✅ Aşama 1 — Ortam (`env/grid_env.py`) — TAMAM

```python
class MARLGridEnv:
    def reset(self, seed=None, config=None) -> dict[agent_id, obs]
    def step(self, actions: dict) -> (obs, rewards, dones, infos)
    def state(self) -> np.ndarray          # QMIX mixer icin global state
    def render(self) -> str                # ASCII
    def action_mask(self, agent) -> np.ndarray[5]
```

ASCII render (debug'ın %80'i buradan gelir):
```
1 . . . .      1 = Ajan1   2 = Ajan2   G = ortak hedef
# . . . .      # = yasak bolge (A1 izi)   * = ikisi ayni hucrede
# . 2 . .
# . . . .
# # # # G
```

- [x] Random `(s1, s2, g)` üretimi; `s1==s2` destekleniyor
- [x] `s1`, `s2`, `g` yasak bölgeden muaf (`test_exemption_same_start`)
- [x] **İki fazlı akış:** A1 hedefe varınca `B` sabitlenir, faz biti 1 olur
- [x] **Gölge NOOP:** pasif ajanın gözlemi Faz A boyunca güncelleniyor
      (`test_passive_obs_updates` — VDN'in kredi kanalının canlı olduğunu
      doğrulayan test)
- [x] Aksiyon maskesi: aktif ajanda `NOOP` kapalı, pasif ajanda sadece `NOOP`
- [x] **Faz sınırında BFS fizibilite kontrolü** + erken sonlandırma
- [x] `max_steps_per_phase=15`, toplam 30
- [x] **Kabul (istatistiksel) GEÇTİ:** 20.000 episode `random_shortest` →
      kilitleme **%0.84** (analitik %0.82), zarar **%13.12** (analitik %13.28),
      başarı **%99.16**. Simülasyon ile tam tarama birbirini doğruluyor.

> ⚠️ Baseline **"random walk" değil, "random shortest path"** olmalı. Rastgele
> yürüyüşün izi devasa olur ve %0.82 yerine çok daha yüksek bir kilitleme
> üretir — yanlış baseline'a karşı ölçüm yaparsan RL'in "iyileştirdiği" şey
> koordinasyon değil, sadece "kısa yol bulmak" olur.

---

### ✅ Aşama 2 — BFS Oracle + tarama (`baselines/bfs_oracle.py`) — TAMAM

Projenin doğruluk zemini. Tarama 0.7 saniyede bitiyor.

- [x] `bfs_dist` / `bfs_path` / `all_shortest_paths` / `oracle`
- [x] Tarama **§0.2 ve §0.3 ile birebir aynı**: `14120 / 280 / 0`,
      zor konfig `%29.2`, kilitlenme yol-ağırlıklı `%5.62`, konfig-ağırlıklı `%0.82`
- [x] `280` vakanın **%100'ünün** tek-yollu olduğu doğrulandı
- [x] `runs/feasibility.csv` + `runs/difficulty.csv` yazıldı
- [x] **Kapalı form testi geçti:** `s1=s2=(0,0), g=(4,4)` için
      "ilk hamle == son hamle ⟺ kilitli" kuralı BFS ile %100 uyuşuyor, 30/70
- [x] 14.400/14.400 konfigin çözülebilir olduğu doğrulandı

---

### ✅ Aşama 3 — Tek ajan DQN (`agents/dqn.py`) — TAMAM

Sadece Ajan 1: random `s1`, random `g`, yasak bölge yok. `env/single_agent.py`
sarmalayıcısı MARLGridEnv'in Faz A'sını tek başına oynatır — gözlem borusu
(129 boyut) Aşama 4-5'e **birebir** taşınacak.

- [x] Replay buffer (100k transition), target network, Double DQN, ε-greedy
- [x] **Kabul GEÇTİ (bağımsız doğrulandı):** 600 (start, goal) çiftinin
      **TAMAMINDA** (örneklem değil, `d(s1,g)` uzayının tamamı) deterministik
      greedy (ε=0) → **600/600 optimal, ortalama gap = 0.0000.**

**Yol boyunca bir Q-value divergence bulundu ve düzeltildi** — projenin ilk
gerçek "hata teşhis et, kanıtla, düzelt" vakası:

| Belirti | Kök neden | Kanıt | Düzeltme |
|---|---|---|---|
| ep16000'de %80.2 optimale ulaşıp ep24000'de **%0.2'ye çökme**, loss 0.008→9.0 (1000 episode'da ~10× büyüme) | Target ağı çok sık güncelleniyordu (500 adım) + LR yüksekti (5e-4) → moving-target pozitif geri besleme, Double DQN'i bile aşan bir Q-şişmesi | Loss'un üstel büyüme paterni klasik divergence imzası; LR/target-update'i düşürünce aynı seed'de aynı yerde tekrar etmedi | `DQN_TARGET_UPDATE` 500→2000, ayrı `DQN_LR=1e-4` (genel `LR`'den bağımsız) |
| Timeout'ta episode'un "gerçek terminal" gibi `done=True` yazılması | Time-limit bootstrapping: 15. adımda hedefe uzaksan değeri 0 sayıyordu, halbuki gerçek değer ~0.7 — hata tam ajanın kaybolduğu durumlara enjekte oluyordu | — | `info["truncated"]` ayrımı: gerçek terminalde `done=True`, timeout'ta buffer'a `done=False` yazılıp bootstrap'e devam edildi |

Ayrıca: Windows'ta stdout bir dosyaya/boruya yönlendirilince (`>`, `\|`)
konsol değil `cp1252` kullanılıyor ve Türkçe karakterlerde `UnicodeEncodeError`
veriyordu — `train.py` başında `sys.stdout.reconfigure(encoding="utf-8")` ile
kalıcı çözüldü.

Öğrenme eğrisi (`runs/dqn_train_log.csv`): ep2000 %82.2 optimal → ep8000 %99.7
→ **ep12000'den itibaren %100.0, gap 0.0, 122.575 adımda, ~200 saniyede.**

---

### ✅ Aşama 4 — IQL (baseline) — TAMAM

İki bağımsız DQN. Ortak ödül **yok** — her ajan sadece `info["r_ind"]`'i alır
(step-cost + kendi hedef bonusu; kilitleme/takım/optimallik cezaları r_team'e
ait, r_ind'e hiç karışmıyor — bu ayrım grid_env.py'de zaten vardı, IQL'in
tanımını otomatik sağladı). Gölge NOOP yok: her ajan SADECE kendi aktif
olduğu adımlarda push()/learn() görür (`env/two_agent.py:play_episode`).

- [x] A2, yasak bölge maskesini (`physical_mask`) gözleminden okuyup kaçınıyor
- [x] **Kabul GEÇTİ (TAM 14.400 konfig, 40.000 episode, 836s):**

| Metrik | IQL sonucu | Random-shortest baseline | Yorum |
|---|---:|---:|---|
| A1 optimal değil (gap1≠0) | **0/14.400** | — | A1 mükemmel |
| A2 **own_gap2** (gerçekleşen yasak bölgeye göre) | ortalama **+0.027**, %98.8'i tam optimal | — | **A2'nin DQN'i de sağlam** |
| A2 ORACLE-gap (A1 en iyi seçseydi) | +0.287 | — | A2'nin hatası değil (aşağıda) |
| Kilitleme oranı | **%0.84** | %0.82 | neredeyse özdeş |
| **Zarar oranı (genel)** | **%12.91** | %13.28 | 🔑 baseline'da çakılı |
| **Zarar oranı (zor alt-küme)** | **%42.71** | ~%45.5 (hesaplanan) | 🔑 aynı seviyede |

**İki DQN de ayrı ayrı kusursuz, ama takım performansı hiç iyileşmiyor.**
`own_gap2` ile `ORACLE-gap` arasındaki fark (0.027 vs 0.287) tam olarak
beklenen ayrımı doğruluyor: A2 kendi payına düşen (gerçekleşmiş) yasak
bölgeye göre neredeyse her zaman optimal — ORACLE-gap'in pozitif çıkması
A2'nin hatası değil, **A1'in "hangi optimal yolu seçtiği önemsiz" diye
öğrenmiş olmasının doğal sonucu.**

Eğitim eğrisi ilginç bir örüntü gösterdi (`runs/iql_train_log.csv`): ep4000'de
zarar(zor) %23 iken ep16000'e kadar %43-44'e **çıkıp** orada düzlendi. Bu bir
kötüleşme değil — rastgele politikadan (epsilon yüksek) gerçek durağan
değerine (baseline seviyesi) **yakınsama.**

> Bu bir bug değil, **beklenen ve istenen sonuç, deneyle doğrulandı.** Sıralı
> modda A1, Faz A boyunca A2'nin akıbetine dair hiçbir ödül sinyali almaz —
> bu bilgi IQL'in ödül yapısında fiziksel olarak **yoktur**. A1 kendi yolunu
> mükemmel öğrenir (gap1=0) ve A2 de kendi payına düşeni mükemmel öğrenir
> (own_gap2≈0) — ama A1'in optimal 70 yol arasından **hangisini** seçtiği
> hiç önemli değilmiş gibi davranması yüzünden takım zararı random-shortest
> ile aynı seviyede kalıyor. Bu, "VDN neden gerekli" sorusunun ölçülmüş
> deneysel cevabı.

---

### 🟡 Aşama 5 — VDN (`agents/vdn.py`) — kod tamam, final sayı bekleniyor

```python
Q_tot = Q_1(o_1, a_1) + Q_2(o_2, a_2)
loss  = (r_team + γ·max Q_tot_target − Q_tot)²
```
Tek TD hatası, iki ajana geri yayılır. A1'in "kilitleme" hatası artık **A1'in**
gradyanına ulaşır — IQL'de ulaşmıyordu. Mekanizma **§2.2 (gölge NOOP)**.

- [x] Gölge NOOP bağlandı: her t'de iki ajanın da Q'su toplama giriyor
      (`play_episode_vdn`, `env/two_agent.py`)
- [x] Tek `r_team`, ajan-başına ödül **kaldırıldı**
- [x] Joint replay buffer: her satır BİR global timestep, HER İKİ ajanın
      (obs,aksiyon,next_obs,next_mask) bilgisini birlikte tutuyor (faz sınırı
      dahil, tam trajektori işleniyor — ayrıca "own_done" mantığına gerek
      kalmıyor, tek `env.done` yeterli)
- [x] **Sağlık kontrolü** eklendi ve geçti: `Q(obs_2,NOOP)` Faz A boyunca
      değişiyor (ör. `fark=0.84`), gölge NOOP'un gözlemi güncelleniyor.
- [ ] ~~Parametre paylaşımı~~ **TERSİ ÇIKTI — paylaşım YOK.** İlk üç tam-ölçekli
      (40k episode) denemede paylaşımlı tek ağ, A1'in SALT navigasyonunu
      (Aşama 3'ün 600 çiftlik testi, A2 yokken) ep~1750'de 237/600'e kadar
      öğretip sonra **çöktü** (150-170/600'e düştü) — hem varsayılan LR/target-
      update'te hem 3× düşük LR + 2× yavaş target update'te tekrarlandı. İzole
      teşhis: AYNI her şey, AYRI iki ağla ep5000'den itibaren 595-597/600'de
      **istikrarlı** kaldı. Kök neden: tek ağ, iki niteliksel farklı rolü
      (A1 serbest gezinme / A2 yasak-bölge-kısıtlı) tek `agent_id` bitiyle
      ayırt etmeye çalışırken yıkıcı parazit üretiyor. `agents/vdn.py` artık
      **ayrı iki ağ** kullanıyor (VDN'in özü — toplamsal ayrıştırma + tek TD
      hedefi — paylaşım gerektirmiyor, bunlar bağımsız tasarım kararları).
- [ ] **Kabul:** zor konfig alt kümesinde **zarar oranı** IQL'den anlamlı
      şekilde düşük (hedef: %13.3 → <%2, IQL: %42.71). Düzeltilmiş mimariyle
      küçük ölçekli koşularda (8-15k episode) zaten <%3'e iniyor — **final
      sayı, deadline nedeniyle küçültülmüş bir eğitim koşusu tamamlanınca**
      `eval/evaluate.py` ile TAM 14.400 konfigde doğrulanacak.

> **Deadline notu:** Orijinal planın 40.000 episode'luk "kanonik" koşu fikri
> zaman bütçesine sığmadı; VDN/QMIX için 8.000-15.000 episode'a düşürüldü.
> Küçük ölçekli doğrulamalar (§Aşama 5 sağlık kontrolü, izole A1 testi)
> bunun yeterli olduğunu gösteriyor — mimari düzeltmeden sonra yakınsama
> ep5.000-8.000 civarında oluyor, IQL'in 40k'sına ihtiyaç yok.

---

### 🟡 Aşama 6 — Curriculum & örnekleme (~2 saat) — **kritik**

§0.3'ün sonucu: uniform sampling'de öğrenilecek şeyin %70'i bedava. Düzelt.

**Ölçülmüş kazanç:** en zor %10'luk dilimde (1440 konfig) zarar oranı **%69.2**,
uniform'da **%13.3**. Yani curriculum sinyali **5 katına** çıkarıyor. Kilitlemeye
özel bakacaksan `d1≥6` dilimi (1440 konfig) kilitlemenin neredeyse tamamını
içeriyor.

**Örnekleme stratejisi (önerilen):**
```python
p_hard = min(0.8, 0.2 + 0.6 * (episode / total_episodes))   # 20% -> 80%
```
Yani başta kolay konfiglerle temel navigasyonu öğren, giderek zor konfiglere
kay. Zor konfig listesi `runs/difficulty.csv`'den geliyor (Aşama 2 üretti).

- [x] `env/sampler.py` — `CurriculumSampler`, `difficulty.csv`'den 3 kova
      (`hard` 4200, `easy_nontrivial` 8280, `easy_trivial` 1920), birim
      testte `p_hard(ep)` teorik değerle ölçülen oran ±0.02 içinde tutuyor
- [x] `d(s1,g)=1` konfigleri (1920 adet, **sıfır** koordinasyon içeriği)
      `easy` havuzu içinde `TRIVIAL_WEIGHT=0.1` ile kısıldı (atılmadı — ağın
      "d1=1 diye bir şey var" bilgisini tamamen kaybetmemesi için)
- [x] Metrikler **kolay / zor / genel** olarak ayrı raporlanıyor
      (`evaluate_iql`/`evaluate_vdn`'e `easy_harm_rate`/`easy_n` eklendi)
- [x] `train.py --curriculum` bayrağı (`iql` ve `vdn` için), duman testiyle
      doğrulandı (300 episode, kolay/zor ayrı satırlarda basıyor)
- [x] **Kabul — ÖLÇÜLDÜ, beklenen kadar net değil:** 8.000 episode'da uniform
      vs curriculum VDN (deterministik, ε=0) karşılaştırıldı:

      | | Uniform (ep8000) | Curriculum (ep8000) |
      |---|---:|---:|
      | A1 kötü (gap1≠0) | 9/1500 | 1/1500 |
      | Kilitleme | %0.40 | %0.33 |
      | Zarar (genel) | %1.93 | %1.80 |
      | Zarar (zor) | %1.74 | %2.61 |

      Bazı metriklerde curriculum hafif iyi, zor alt-kümede hafif kötü — bu
      örneklem boyutunda (1500 konfig, ~435 zor) fark gürültü seviyesinde,
      **net bir kazanç yok**. Muhtemel neden: bu bulgunun dayandığı orijinal
      gerekçe IQL'in *hiç iyileşmeme*sine dayanıyordu (§0.3); VDN'in ayrı-ağ
      mimari düzeltmesi zaten kendi başına hızlı yakınsıyor, curriculum'un
      ekstra katkısı beklenenden küçük. **Dürüstçe negatif/nötr sonuç olarak
      kaydedildi** — abartılmadı. Curriculum yine de final VDN/QMIX
      modellerinde açık bırakıldı (zararı yok, potansiyel faydası var).

---

### 🟡 Aşama 7 — QMIX (`agents/qmix.py`) — kod tamam, final sayı bekleniyor

VDN'in `Q_tot = Q_1 + Q_2` toplamsallığını, monotonik bir mixing ağıyla değiştir.
Ortam **hiç değişmiyor** — sadece toplama işleminin yerine mixer geliyor.
Per-ajan Q ağları VDN'deki gibi **ayrı** (aynı paylaşım riski burada da geçerli).

- [x] Hypernetwork + monotonik mixer (`abs(W)` ile non-negatif ağırlıklar,
      `QMixer` sınıfı: `hyper_w1/b1/w2/b2`, `embed_dim=32`)
- [x] Global state: `env.state()` (iki konum + hedef + yasak bölge + faz),
      `play_episode_qmix` her joint transition'a `state`/`next_state` ekliyor
- [x] Sağlık kontrolü geçti (duman testinde `Q(obs_2,NOOP)` FAZ A boyunca
      değişiyor, `fark=0.20`)
- [ ] **Kabul:** §2.1 hipotezi test edilecek — QMIX, zor alt kümede VDN'i
      geçiyor mu? **Geçmese bile rapora yazılacak**, negatif sonuç da sonuçtur.
      Final eğitim koşusu tamamlanınca `eval/evaluate.py` ile ölçülecek.

---

### 🟡 Aşama 8 — Değerlendirme (`eval/evaluate.py`) — kod tamam, final koşu bekleniyor

| Metrik | Tanım | Hedef |
|---|---|---|
| **Zarar oranı** | A2 kilitlendi **veya** uzadı | 🔑 **ANA METRİK** — baseline %13.3, hedef <%2 |
| **Zor-alt-küme zarar** | Sadece %29.2'lik zor konfiglerde | 🔑 asıl ayrım burada görünür |
| Success rate | İki ajan da hedefe vardı | >%99.9 (baseline %99.16) |
| Kilitleme oranı | A1, A2'yi tamamen bloke etti | ~%0 (baseline %0.82) |
| Optimality gap A1 | `len(P1) − manhattan(s1,g)` | 0.0 |
| Optimality gap A2 | `len(P2) − ORACLE(s1,s2,g)` | 0.0 |
| İhlal sayısı | A2'nin yasak bölgeye giriş denemesi | 0 |
| Sample efficiency | %95 başarıya kaç episode'da ulaşıldı | algoritma karşılaştırması |

> Kilitlemeyi tek başına ana metrik yapma: %0.82'de, 100 eval episode'unda
> ortalama **1 tane** görürsün. Anlamlı ölçüm için ya 2000+ episode ya da
> `d1≥6` dilimine odaklı ayrı bir eval seti gerekir.

> ⚠️ A2'nin gap'ini **`ORACLE`'a göre** ölç, serbest Manhattan'a göre değil.
> Yoksa §0.2'deki 280 vakada ajanı imkânsız bir şey için cezalandırırsın ve
> metriklerin yalan söyler.

Karşılaştırma tablosu (raporun ana çıktısı):

| | Random-shortest | Bencil BFS | IQL | VDN | QMIX | Oracle |
|---|---|---|---|---|---|---|
| **Zarar oranı (genel)** | **13.3%** | | | | | **0%** |
| **Zarar oranı (zor)** | | | | | | 0% |
| Success | 99.16% | | | | | 100% |
| Kilitleme | 0.82% | | | | | 0% |

> "Random-shortest" = A1 optimal yollarından uniform rastgele biri.
> **"Random walk" ile karıştırma** — o çok daha kötü bir baseline'dır ve
> RL'in kazancını yapay olarak şişirir (§Aşama 1 uyarısı).

- [x] `eval/evaluate.py`: Random-shortest, Bencil BFS, Oracle, IQL, VDN, QMIX
      — HEPSİ **TAM 14.400 konfigde** (2000 örneklem değil), TEK ortak
      `run_episode()` döngüsünden geçiyor (scripted politikalar ile öğrenen
      ajanlar arasında kod-yolu sapması riski yok)
- [x] Scripted baselines + IQL ile duman testi yapıldı, tablo doğru üretiyor
- [x] `runs/eval_report.md` üretiliyor
- [ ] 3 farklı random seed ile eğitim → ortalama ± std — **deadline nedeniyle
      atlandı**, tek seed (0) ile raporlanıyor. Zaman kalırsa eklenir.

---

### 🟡 Aşama 9 — Görselleştirme (`viz/plot_iql_report.py`) — kod tamam

- [x] Öğrenme eğrileri: her algoritma için `{tag}_train_harm.csv`'den (yoğun,
      egitim-ici, genel + zor alt-küme ayrı çizgi) + `{tag}_train_log.csv`'den
      (seyrek, DETERMİNİSTİK eval noktaları) — `plot_harm_curve()`
- [x] `plot_algorithm_comparison()`: IQL vs VDN vs QMIX, zor alt-kümede
      deterministik zarar oranı, TEK grafikte (`train.py --final` sonrası)
- [x] Grid çizimi: A1 mavi, A2 turuncu, yasak bölge gri, hedef yeşil, start
      kareleri — `plot_demo_grids()`, 10 deterministik episode 2×5 panelde
- [x] Curriculum vs uniform karşılaştırma grafiği (`plot_curriculum_comparison`)
- [ ] GIF animasyonu (`FuncAnimation`) — **deadline nedeniyle atlandı**,
      statik PNG paneller yeterli kabul edildi
- [ ] "Önce/sonra" tekil örnek — demo grid panelinde zaten [ZOR]/UZADI/
      KİLİTLENDİ etiketleriyle dolaylı olarak var, ayrı görsel yapılmadı

**Stretch (değer sırasıyla):**
1. **NxN genelleme** — 5x5'te eğit, 7x7'de test et (kanal formatı bunu destekliyor; CNN'e geç)
2. **3+ ajan** — zincirleme kısıt (A3, A1∪A2'den kaçınır); parametre paylaşımı
   sayesinde ağ hiç değişmez, sadece faz sayısı artar
3. **Eşzamanlı mod** — yasak bölge adım adım büyür, çarpışma kuralı + `BEKLE`
   aksiyonu devreye girer. Sıralı sonuçlarla karşılaştırınca güzel bir ablation
   olur (ama oracle orada sadece üst sınır verir)
4. **Kısmi gözlemlenebilirlik** — 3x3 görüş + GRU (gerçek Dec-POMDP)
5. **Statik engeller** — random duvarlı gridler
6. **QTRAN / QPLEX** — additivity kısıtını tamamen kaldıran yöntemler

---

## 5. Ödül tasarımı

VDN/QMIX **tek bir takım ödülü** ister — ayrıştırmayı algoritma yapar. Ajan
başına ödül yazarsan VDN'in varlık sebebini yok edersin.

| Olay | `r_team` | Ne zaman | Neden |
|---|---:|---|---|
| Her timestep | −0.05 | her t (her iki fazda) | En kısa yolu teşvik |
| Duvara / yasak hücreye hamle | −0.10 | anında | Boşa adım caydırıcı |
| A1 hedefe vardı | +1.0 | faz sınırı | Ara terminal, sinyali yoğunlaştırır |
| **A2 kilitlendi** | **−3.0** | **faz sınırı (BFS ile)** | 🔑 **İstediğin kilitleme cezası** |
| A2 hedefe vardı | +1.0 | episode sonu | Ara terminal |
| **İkisi de vardı** | **+3.0** | episode sonu | Takım hedefi |
| Timeout (herhangi faz) | −3.0 | max_steps | Sonsuz dolaşmayı kes |
| **Optimallik cezası** | **`−0.5 × (len2 − ORACLE)`** | episode sonu | 🔑 **asıl sinyal bu** (§0.3) |

> **Kilitleme cezası faz sınırında verilir**, episode sonunda değil — çünkü
> orada BFS zaten cevabı biliyor (§2.2 erken sonlandırma). Bu, cezayı A1'in
> hamlelerine `γ^T1` uzaklığa getirir; episode sonuna bırakırsan `γ^(T1+T2)`
> olur ve kredi ataması gereksiz yere zayıflar.

### Kalibrasyon mantığı
A1'in kendi step-cost'u tüm optimal yollarında sabit (`−0.05 × d(s1,g)`, en
fazla −0.4). Kilitleme farkı **6.0 puan** — yani A1'in kendi maliyetinin ~15
katı. Bu kadar büyük olmalı ki A1 "kısa yoldan gittim ya" deyip A2'yi ezmesin.

### Üç kural
1. **`ORACLE`'a göre ölç.** §0.2'deki 280 vakada A1'in alternatifi yok; serbest
   Manhattan'a göre ceza yazarsan ajanı imkânsız bir şey için cezalandırırsın ve
   gradyanı zehirlersin.
2. **Ödül ikili olmasın.** Sadece "+3 / −3" verirsen sinyal çok seyrek olur
   (zaten konfiglerin %70'i bedava kazanç). Ara terminal (+1.0) ve step cost
   sinyali yoğunlaştırır.
3. **Reward shaping yapacaksan potential-based yap.**
   `Φ(s)` = A2'nin hedefe erişilebilir kalan mesafesi; `r' = r + γΦ(s') − Φ(s)`.
   Potential-based shaping optimal politikayı **değiştirmez** (Ng et al. 1999) —
   naif shaping değiştirir, ajan ödül farmına başlar.

---

## 6. Klasör yapısı

```
MARL-Pathfinding/
├─ PLAN.md
├─ README.md
├─ requirements.txt
├─ config.py                # TÜM hiperparametreler ve mod bayrakları
├─ env/
│  ├─ grid_env.py           # MARLGridEnv (sirali akis -> PettingZoo AEC benzeri)
│  └─ sampler.py            # curriculum / zorluk agirlikli konfig sampler
├─ agents/
│  ├─ networks.py           # MLP, hypernetwork
│  ├─ dqn.py                # tek ajan (Asama 3)
│  ├─ iql.py                # bagimsiz Q-learning
│  ├─ vdn.py                # Q_tot = sum Q_i
│  ├─ qmix.py               # monotonik mixer
│  └─ buffer.py             # episode-bazli replay
├─ baselines/
│  ├─ bfs_oracle.py         # BFS + tum optimal yollar + oracle()
│  └─ scan.py               # 14.400 konfig taramasi -> feasibility/difficulty.csv
├─ train.py                 # python train.py --algo vdn --seed 0
├─ eval/
│  └─ evaluate.py           # metrikler + kolay/zor ayrimi + baseline tablosu
├─ viz/
│  ├─ plot_curves.py
│  └─ animate.py
├─ tests/
│  ├─ test_env.py           # s1/s2/g muafiyeti, maske, max_steps
│  └─ test_oracle.py        # BFS, 70 yol, muafiyet, kapali form kurali (30/70)
└─ runs/                    # git'e girmez
   ├─ feasibility.csv
   ├─ difficulty.csv
   ├─ ckpt/
   └─ eval_report.md
```

---

## 7. Hiperparametreler (başlangıç seti)

| | Değer | Not |
|---|---|---|
| Optimizer | Adam, `lr=5e-4` | |
| γ | 0.99 | episode kısa, yüksek γ sorun değil |
| ε | 1.0 → 0.05, 50k adımda lineer | |
| Replay buffer | 5.000 episode | |
| Batch | 32 episode | VDN/QMIX episode bazlı örnekler |
| Target network | her 200 episode'da hard update | |
| Grad clip | 10.0 | mixer'da patlamayı önler |
| Toplam eğitim | 100.000 episode | CPU'da ~1-2 saat |
| Ağ | 128-128 MLP, ReLU | ~35k parametre |
| Double DQN | açık | overestimation'ı kırar |
| Parametre paylaşımı | açık (`agent_id` gözlemde) | 2 ajan için daha stabil |

---

## 8. Tuzaklar tablosu (belirti → kök neden → çözüm)

| Belirti | Kök neden | Çözüm |
|---|---|---|
| A2 daha ilk adımda çözümsüz | `s1`/`s2`/`g` yasak bölgeden muaf değil | §1'deki kritik kural (600 `s1==s2` konfigi) |
| Öğrenme eğrisi düz, "hiçbir şey öğrenmiyor" | Uniform sampling — %70 konfig bedava kazanç | Curriculum (Aşama 6), metrikleri **zor alt kümede** ölç |
| VDN, IQL'den farksız | Ödül ajan-başına yazılmış, takım ödülü yok | Tek `r_team`, ayrıştırmayı VDN yapsın (§5) |
| A1 hep kısa yol buluyor ama kilitliyor | Kilitleme cezası step-cost'a göre küçük | Fark ≥ 10× step-cost (§5 kalibrasyon) |
| Metrikler iyi ama ajan aptal görünüyor | Gap serbest Manhattan'a göre ölçülmüş | `ORACLE()`'a göre ölç (280 vaka) |
| Q değerleri patlıyor / NaN | Mixer ağırlıkları negatife kayıyor | `abs(W)` ile monotonluk + grad clip 10.0 |
| Eğitim ilerledikçe kötüleşiyor | Nonstationarity, target network çok sık güncelleniyor | Target update aralığını 200→500 çıkar |
| Eval sonucu eğitimden kötü | Eval'de ε > 0 kalmış | `ε=0`, deterministik greedy |
| Sonuçlar koşudan koşuya çok değişiyor | Tek seed ile rapor | 3+ seed, ortalama ± std |
| Aktif ajan `NOOP` basıp duruyor | `NOOP` aktif ajan için maskelenmemiş | Maske: aktif ajanda `NOOP` kapalı (§Aşama 1) |
| Torch kurulumu takılıyor | GPU wheel'i indirmeye çalışıyor (2.5 GB) | CPU wheel yeter, `pip install torch` varsayılanı doğru |
| **VDN, IQL ile birebir aynı çıkıyor** | Gölge NOOP'ta pasif ajanın **gözlemi güncellenmiyor** → `Q_2` sabit → gradyan A1'e akmıyor | Faz A'da `Q_2(obs_2, NOOP)`'un t ile değiştiğini logla (§Aşama 5 sağlık kontrolü). **En sinsi bug bu.** |
| Kilitleme cezası A1'e ulaşmıyor | Ceza episode sonunda veriliyor, discount zinciri uzun | Faz sınırında ver (§5) |
| Faz B'de A1'in Q'su saçmalıyor | Pasif ajana `NOOP` dışında aksiyon açık kalmış | Pasif ajanda **sadece** `NOOP` açık |

---

## 9. Zaman tahmini

| Aşama | Süre | Kümülatif |
|---|---:|---:|
| 0 Kurulum | 0.75 s | 0.75 |
| 1 Ortam | 3 s | 3.75 |
| 2 BFS oracle + tarama | 3 s | 6.75 |
| 3 Tek ajan DQN | 4 s | 10.75 |
| 4 IQL baseline | 3 s | 13.75 |
| 5 **VDN** | 4 s | 17.75 |
| 6 **Curriculum** | 2 s | 19.75 |
| 7 QMIX | 4 s | 23.75 |
| 8 Değerlendirme | 3 s | 26.75 |
| 9 Görselleştirme | 3 s | **~30 saat** |

Odaklanmış **5-6 günlük** iş. En riskli iki yer: **Aşama 5 (gölge NOOP'un doğru
bağlanması — VDN'in gerçekten IQL'i geçmesi)** ve **Aşama 6 (curriculum olmadan
fark görünmez)**. Tamponu oraya bırak.

Zaman daralırsa kesme sırası: 9 → 7(QMIX) → 4(IQL). **Aşama 2 ve 6 asla kesilmez** —
biri doğruluk zemini, diğeri projenin ölçülebilir olmasını sağlıyor.

---

## 10. Tek paragraf özet

İki fazlı sıralı ortamı yaz (start/goal muafiyeti + faz sınırında BFS kontrolü
kritik) → BFS oracle'ı yaz ve §0'daki `14120/280/0`, `%0.82`, `%13.3`, `%29.2`
sayılarını kendi kodunla yeniden üret → tek ajan DQN'i optimal yaptır → IQL
baseline'ını koş ve **zarar oranının %13.3'te çakılı kalmasını göster** →
VDN'i **gölge NOOP** ile bağla,
takım ödülünü ayrıştır → **curriculum ile zor konfigleri öne çıkar** (yoksa
hiçbir fark görünmez) → QMIX ile additivity hipotezini test et → üçünü 3 seed
üzerinde kolay/zor ayrımıyla karşılaştır → çiz, yaz.

**Altın kural:** Her sonucu **zor alt kümede** (%29.2) rapor et. Genel ortalamada
üç algoritma da ~%95 çıkar ve hiçbir şey öğrenmemiş olursun.

---

## 11. Aşama 10 — Büyük ölçek genelleme (5x5 → 50x50)

Tüm önceki bölümler 5x5'in **tam tarama / exhaustive** metodolojisini
belgeliyor (14.400 konfig, BFS oracle'ın tüm optimal yolları enumerate etmesi
vb.). Bu bölüm, aynı problemi **büyük N**'de (önce 100x100 denendi, sonra
eğitim süresi nedeniyle **50x50**'ye çekildi) çalıştırmak için yapılan
mimari değişiklikleri özetler — büyük N, 5x5'in doğruluk-zemini varsayımlarını
(exhaustive scan, tüm-yolları-enumerate) kırdığı için bunlar zorunluydu.

### 11.1 Neyin kırıldığını, neden

- **`all_shortest_paths()`/`oracle()`**: 5x5'te 70 yol (C(8,4)) enumerate
  edilebiliyordu. Büyük N'de kombinatorik patlıyor (100x100 köşe-köşe
  C(198,99) — trilyonlarca) VE bu fonksiyon her episode'da **ödül
  hesaplarken** çağrılıyordu — yani egitim bile başlayamazdı. **Çözüm:**
  `grid_env.py`'nin `_finish()`/`_terminal_info()`'u artık `oracle()`
  kullanmıyor, SERBEST Manhattan mesafesiyle kıyaslıyor (5x5'te bu iki ölçüt
  zaten %98.1 aynıydı — §0.2). Kilitleme kontrolü (`bfs_dist`, O(n²)) hiç
  değişmedi.
- **`baselines/scan.py`'nin exhaustive taraması**: 14.400 konfig büyük N'de
  ~10^12'ye çıkıyor, taranamaz. **Çözüm:** `env/sampler.py`'nin
  `CurriculumSampler`'ı artık `grid_n` verilirse `difficulty.csv` yerine
  n_samples (varsayılan 1000-5000) rastgele konfig üretip AYNI eşik kuralıyla
  (d1/max_man ≥ 0.6 → "hard") sınıflandırıyor. `train.py`'nin
  `load_all_configs()`'u ve `eval/evaluate.py` bunu kullanıyor.
- **Tam-grid one-hot gözlem**: 5 kanal × N² (100x100'de 50.000 float,
  %99.99'u sıfır) hem replay buffer'ı patlatıyordu (300k transition için 56
  GB) hem öğrenmeyi zorlaştırıyordu. **Çözüm:** ajan artık SADECE kendi
  **yerel penceresini** (`PATCH_RADIUS=10`, 21x21) + hedefe/diğer ajana
  GÖRELİ yön vektörlerini görüyor (`OBS_DIM=893`, `STATE_DIM=890` — GRID_N'den
  BAĞIMSIZ, sadece patch boyutuna bağlı). Bu sayede GRID_N değiştiğinde
  (100→50) ağ mimarisi hiç değişmedi.
- **MLP → CNN**: Patch-tabanlı girdiyi işlemek için `agents/networks.py`'ye
  `CNNQNet` eklendi (stride-2 conv + `AdaptiveAvgPool2d` — parametre sayısı
  patch boyutundan bağımsız). `QMixer`'in hypernetwork'ü de aynı nedenle
  küçük bir CNN state-kodlayıcıdan geçiyor artık.
- **Seyrek ödül**: Büyük N'de rastgele gezinmeyle hedefe ulaşma süresi
  karesel büyür — early training'de terminal ödül neredeyse hiç
  yakalanmıyordu. **Çözüm:** potential-based reward shaping eklendi
  (`SHAPING_COEF`, Ng ve ark. 1999 — optimal politikayı degistirmedigi
  kanıtlı), her adımda hedefe yaklaşma/uzaklaşmaya orantılı yoğun sinyal.
- **`learn()` maliyeti**: CNN'in CPU'daki `learn()` çağrısı ölçüldü (~10ms,
  batch=128) — 15k episode 13 saate çıkıyordu. **Çözüm:** `VDN_BATCH` vb.
  128→32, `LEARN_EVERY=4` (her adımda değil her 4 adımda bir gradyan adımı).
  ~6.5x hızlanma.

### 11.2 Ölçek kararı: 100x100 → 50x50

100x100'de ölçülen süreler (curriculum açık, 3000 episode): IQL ~21dk, VDN
~24dk (tahmini), QMIX ~41.5dk (tahmini) → 30.000 episode'da IQL ~3.5 saat,
VDN ~4 saat, **QMIX ~6.9 saat**. Deadline baskısıyla **50x50**'ye çekildi:
aynı ölçümler ~2.3x hızlandı (VDN 30k ≈ **2.1 saat**, QMIX 30k ≈ **3.0 saat**).
Gözlem/ağ mimarisi GRID_N'den bağımsız olduğu için bu geçiş **tek satır**
(`config.GRID_N`) + `MAX_STEPS_PER_PHASE`'in orantılı küçültülmesiyle yapıldı.

### 11.3 İlk 100x100 3000-episode denemesinin bulgusu (dürüst kayıt)

İlk deneme (100x100, 3000 episode, curriculum) IQL/VDN/QMIX'in hepsinde
`eval_success_rate` %0-4.5 civarında çıktı — yani ajan çoğu episode'da hedefe
hiç ulaşamadı. Bu bir hata DEĞİL: 100x100'ün konfig uzayı 5x5'teki gibi
ezberlenebilir/sonlu değil, gerçek genelleme gerektiriyor, ve 3000 episode
bunun için görünüşe göre yetersizdi (IQL'in eğrisi yükseliyordu, sadece
yavaş). `mean_gap2`/`zarar oranı` gibi metrikler `success_rate` KONTROL
EDİLMEDEN yorumlanırsa yanıltıcı olur (az sayıda başarılı episode'un ortalama
gap'i, "genel" başarısızlığı gizleyebilir) — bu, VDN'nin Aşama 5'teki "sahte
başarı" bulgusunun BÜYÜK N'DE TEKRARIDIR, aynı kontrolü (`success_rate`'i her
zaman önce bak) burada da uygula.

### 11.4 Otomasyon: `run_all.bat`

Proje kökünde `run_all.bat <episodes>` — IQL/VDN/QMIX'i PARALEL başlatır,
üçü de bitene kadar bekler (marker dosyası deseni: eğitim hata ile bitse bile
marker yazılır, sonsuz beklemeyi önler), sonra `eval.evaluate` ve
`viz.plot_iql_report --final`'i otomatik çalıştırır. **NOT:** `.bat`
dosyaları CRLF satır sonu ister — LF ile yazılırsa cmd.exe her satırın ilk
karakterini yutarak bozuk şekilde çalıştırır (bu projede bir kez yaşandı,
`[System.IO.File]::WriteAllText(...,[System.Text.Encoding]::ASCII)` ile
CRLF'ye normalize edilerek düzeltildi).
