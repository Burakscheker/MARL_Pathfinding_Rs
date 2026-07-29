"""Tum hiperparametreler ve sabitler burada. Koda deger gomme.

PLAN.md ile senkron tut: ozellikle odul degerleri (§5) ve max_steps (§Asama 1).
"""

# ---------------------------------------------------------------- grid / ortam
GRID_N = 50
N_AGENTS = 2

AGENT_1 = 0          # once giden, izi yasak bolge olan ajan
AGENT_2 = 1          # yasak bolgeden kacinarak giden ajan

# Aksiyonlar
UP, RIGHT, DOWN, LEFT, NOOP = range(5)
N_ACTIONS = 5
ACTION_NAMES = ("UP", "RIGHT", "DOWN", "LEFT", "NOOP")

# (d_satir, d_sutun) — NOOP dahil degil, o ayri ele aliniyor
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))

# ---------------------------------------------------------------- statik engeller (zorluk modu)
# Zorluk secilirse, HER ajanin baslangici ile hedef arasina, o dogrultunun
# BASKIN eksenine DIK, sabit genislikte bir duvar konur (agent_env.py
# _build_wall()). Genislik GRID_N'den kucuk oldugu icin duvar en fazla TEK
# kenara deger — geometrik olarak her zaman dolanilabilir bir bosluk kalir.
# Iki ajanin duvarlari (nadiren) birlesip hedefi sarabilecegi icin reset()
# ayrica BFS ile dogrular; cozulemezse o episode icin duvarlar KAPATILIR
# (config'in kendisi ASLA degistirilmez).
WALL_WIDTHS = {"easy": 5, "medium": 7, "hard": 10}

# Faz basina adim limiti. 50x50'de en uzun optimal yol 98 (49+49);
# egitimin ilk asamalarinda politika henuz yon bulmayi ogrenmemisken biraz
# dolanmaya izin vermek icin ~1.4x tampon (140, 100x100'un 270'iyle ayni oran).
# Cok comert tutulmadi: her fazlalik adim, ozellikle rastgele-a-yakin erken
# episode'larda, dogrudan egitim suresine ekleniyor.
# NOT: OBS_DIM/STATE_DIM (asagida) PATCH_SIZE'a bagli, GRID_N'e DEGIL — grid
# boyutu degistiginde ag mimarisi hic etkilenmiyor (yerel-pencere tasariminin
# faydasi tam burada).
MAX_STEPS_PER_PHASE = 140
MAX_STEPS_TOTAL = 2 * MAX_STEPS_PER_PHASE

# Ajanlar ayni hucreden baslayabilir mi (PLAN §0.1: 600 konfig)
ALLOW_SAME_START = True

# ---------------------------------------------------------------- odul (PLAN §5)
R_STEP = -0.05        # her timestep
R_INVALID = -0.10     # duvara / yasak hucreye hamle (ajan yerinde kalir)
R_AGENT_GOAL = +10.0  # bir ajan hedefe vardi (ara terminal)
R_BOTH_GOAL = +30.0   # ikisi de vardi (takim hedefi)
R_BLOCKED = -3.0      # A2 kilitlendi — faz sinirinda BFS ile tespit
R_TIMEOUT = -10.0     # herhangi bir fazda max_steps asildi
# OLCEK NOTU: bir ara denemede bunlar 50/300'e cikarilmisti (hedefe varmayi
# daha cok "motive etsin" diye). QMIX o kosuda basari oranini ep2500'de %6.5'e
# cikarip epsilon tabana indigi ANDA %0'a cokturdu — config.py'de VDN icin
# zaten belgelenmis olan olcek-kaynakli cokme imzasi (LR=3e-5 ile ~350
# buyuklugunde Q degeri temsil etmek kirilgan). 10/30 ayni sirlamayi ve
# "hedefe varmak cok degerli" niyetini korur, kanitlanmis stabil olcege yakin
# kalir. Asil "hedefe git" sinyalini zaten SHAPING_COEF tasiyor.
R_REVISIT = -0.05     # kendi fazinda DAHA ONCE ugradigi bir hucreye geri donus.
                      # -0.20 degil: A2 yasak bolgeyi DOLASMAK icin cogu zaman
                      # MECBUREN geri adim atar (elle yazilmis greedy politika
                      # tam bu yuzden A2'de %76'da kaldi) — sert bir revisit
                      # cezasi A2'yi dogrudan sakatliyor. Gitgel'i zaten
                      # guclendirilmis shaping cezalandiriyor.
# Katsayi: R_OPT_GAP * (len2 - ORACLE(s1,s2,g)).
# Tarama sonucu (baselines/scan.py): random-shortest A1 altinda A2'nin
# KILITLENME olasiligi sadece %0.82, ama UZAMA olasiligi %13.3. Yani asil
# ogrenme sinyalini bu terim tasiyor, R_BLOCKED degil. Tipik uzama 2 adim ->
# -1.0 ceza, +3.0'lik takim bonusuna karsi anlamli bir agirlik.
R_OPT_GAP = -0.50

# Potential-based reward shaping katsayisi (Ng ve ark. 1999) — grid_env.py
# step()'te uygulanir: r' = r + COEF*(gamma*Phi(s')-Phi(s)),
# Phi(s) = 1 - manhattan(s,goal)/max_man  (0 = en uzak, 1 = hedefte).
# 50x50'de SEYREK terminal odulun rastgele/az-egitilmis politikayla neredeyse
# hic yakalanamamasi sorununu cozer (duzlemde rastgele yurusun hedefe varis
# suresi N ile karesel buyur) — HER ADIMDA hedefe yaklasma/uzaklasmaya
# orantili yogun sinyal ekler. Optimal politikayi degistirmedigi KANITLANMIS
# (herhangi bir Phi ve herhangi bir KATSAYI icin policy-invariance garantisi).
#
# 1.0 -> 20.0 GEREKCESI (olculdu): COEF=1'de hedefe dogru bir adimin yon
# sinyali sadece ~+0.015'ti; R_STEP(-0.05) 3x, R_INVALID(-0.10) 7x,
# R_REVISIT(-0.20) 13x daha buyuktu. Yani ajanin duydugu EN ZAYIF sinyal
# "hedefe git" olani oluyordu ve ajan hedefi bulamadan 140 adimi tuketiyordu
# (demo gorsellerinde len1=140 timeout'lari). Elle yazilmis greedy politika
# AYNI ortamda 300/300 optimal cikti — yani ortam saglamdi, sorun sinyal
# gucundeydi. COEF=20'de yon sinyali ~±0.3'e cikar, diger terimlerin ustune
# gecer.
SHAPING_COEF = 20.0

# ---------------------------------------------------------------- egitim (ortak)
SEED = 0
GAMMA = 0.99
LR = 5e-4
GRAD_CLIP = 10.0
HIDDEN = 128

# CNN'in learn() cagrisi CPU'da ~2-4ms olculdu (batch=32, P1/scalar_enc dahil),
# tek bir env.step()'ten (~0.18ms) ~13x pahali. HER env adiminda degil,
# LEARN_EVERY adimda bir gercek gradyan adimi atarak (araya giren push()'lar
# hala buffer'a giriyor, sadece learn() hesaplamasi atlaniyor) toplam egitim
# suresini kisaltir. 4->8: olculen ortalama adim maliyeti ~0.76ms->~0.47ms
# (~%38 daha hizli). Bedeli: ayni episode sayisinda toplam gradyan adimi
# yariya iner, yakinsama biraz yavaslayabilir.
LEARN_EVERY = 8

# ---------------------------------------------------------------- MARL (Asama 4+)
EPS_START, EPS_END, EPS_DECAY_STEPS = 1.0, 0.05, 50_000
BUFFER_EPISODES = 5_000
BATCH_EPISODES = 32
TARGET_UPDATE_EVERY = 200      # episode
TOTAL_EPISODES = 100_000

# ---------------------------------------------------------------- DQN (Asama 3)
# Tek ajan, yasak bolge yok. Amac: MARL'a gecmeden once DQN makinesinin
# dogrulugunu kanitlamak. Episode'lar cok kisa (ortalama ~3.3 adim), o yuzden
# epsilon ADIM sayisina gore soner, episode sayisina gore degil.
DQN_EPISODES = 30_000
DQN_BUFFER = 100_000           # transition
DQN_BATCH = 32                 # transition (128->32: CNN'de learn() CPU'da ~10ms/cagri
                               # olculdu, buyuk N'de 15k episode 13 saate cikiyordu)
DQN_EPS_DECAY_STEPS = 30_000   # adim
DQN_LEARN_START = 1_000        # bu kadar transition birikmeden ogrenme
DQN_LR = 1e-4                  # genel LR'den (5e-4) daha dusuk: Q-divergence onlemi
DQN_TARGET_UPDATE = 2_000       # adim (500 -> 2000: moving-target/divergence onlemi)
DQN_EVAL_EVERY = 2_000         # episode

# Curriculum (PLAN §Asama 6): zor konfig orani 0.2 -> 0.8
P_HARD_START, P_HARD_END, P_HARD_CAP = 0.20, 0.80, 0.80

# ---------------------------------------------------------------- IQL (Asama 4)
# Iki bagimsiz DQN (Asama 3'teki DQNAgent, degistirilmeden). Ortak odul YOK --
# her ajan info["r_ind"][agent]'i alir (grid_env.py'de zaten sadece step-cost +
# kendi hedef bonusu iceriyor; kilitleme/takim/optimallik cezalari r_team'e
# ait, r_ind'e hic eklenmiyor). Episode basina iki faz oldugu icin Asama 3'e
# gore hem daha az episode hem daha genis buffer/decay yeterli.
# 100x100 NOTU: olcum, IQL episode basina ~0.33s (CNN + LEARN_EVERY=4 ile).
# 40.000 episode ~3.7 saat surerdi -- deadline'a sigmaz. 3000 episode ~17dk.
IQL_EPISODES = 3_000
IQL_BUFFER = 100_000           # transition, ajan basina (bellek: ~15.6GB RAM'e gore ayarlandi)
IQL_BATCH = 32
IQL_EPS_DECAY_STEPS = 145_000  # adim, ajan basina (gercek olcum ~72.7 adim/ep; 4000 episode'lik run'da ep~2000'de tabana iner)
IQL_LEARN_START = 1_000
IQL_LR = 1e-4
IQL_TARGET_UPDATE = 2_000      # adim
IQL_EVAL_EVERY = 500           # episode (kucuk N'deki 4000 100x100'de egitimin
                               # sonuna kadar hic tetiklenmezdi -- egri icin cok seyrek)

# Egitim SIRASINDAKI (epsilon-greedy) zarar orani icin yogun/hareketli-ortalama
# loglama — kaba eval_every (4000) yerine "30000 run boyunca" turu bir egri
# icin bu cok seyrek kalir. Neredeyse bedava: ekstra ag gecisi gerektirmez,
# sadece play_episode'un zaten dondurdugu info["harmed"] bayragini biriktirir.
TRAIN_HARM_WINDOW = 500        # hareketli ortalama penceresi (episode)
TRAIN_HARM_LOG_EVERY = 100     # bu kadar episode'da bir CSV'ye yaz

# Egitim SONRASI, epsilon=0 (tam deterministik) N gosterim episode'u —
# haritalarini cizip gorsellestirmek icin.
DEMO_EPISODES = 10
DEMO_SEED = 777

# ---------------------------------------------------------------- VDN (Asama 5)
# Paylasilan TEK Q-agi (parametre paylasimi, agent_id/faz zaten gozlemde).
# Q_tot = Q(obs_1,a_1) + Q(obs_2,a_2), TEK TD hatasi ikisine BIRDEN geri yayilir.
# Golge NOOP (PLAN §2.2) burada gercek anlam kazaniyor: pasif ajanin Q(obs,NOOP)'u
# da toplama girer ve gradyan alir — IQL'de push() bile edilmiyordu.
# 100x100 NOTU: VDN episode basina ~0.48s olculdu. 3000 episode ~24dk.
VDN_EPISODES = 3_000
VDN_BUFFER = 150_000           # ortak (joint) transition — her satir BIR global timestep (bellek guvenligi)
VDN_BATCH = 32
# DIKKAT: VDN'in TEK paylasilan agi her t'de (HER IKI ajan icin de) sayaci
# ilerletir; IQL'in agent basina AYRI sayaclari sadece o ajanin AKTIF oldugu
# yarim fazda ilerliyordu. Yani ayni sayisal esik VDN'de ~2x daha HIZLI
# (episode cinsinden) tukeniyor. Ilk 40k-episode kosuda bu 40_000 degeriyle
# epsilon ep~2500'de tabana indi (IQL'de ep~6000-7000) ve A1 pek cok konfigde
# 2-hucreli bir Q-salinimina (asagi/yukari sonsuz dongu) kilitlenip kacamadi
# (reached1_frac sadece %17.1 cikti). 80_000, VDN'in episode-basina decay
# hizini IQL'inkiyle YAKLASIK esitler.
VDN_EPS_DECAY_STEPS = 370_000  # adim (paylasilan agin GLOBAL adim sayaci; gercek olcum ~186 adim/ep; 4000 episode'lik run'da ep~2000'de tabana iner)
VDN_LEARN_START = 1_000
# DIKKAT: ilk denemede DQN_LR (1e-4) + DQN_TARGET_UPDATE (2000) ile VDN
# ep~1750 civarinda (A1-yalniz 600 ciftlik izole navigasyon testinde 237/600)
# tepe yapip COKUYORDU (ep3000'de 167/600'e dusuyordu) -- Q_tot = Q1+Q2'nin
# TEK, PAYLASILAN agdan gelen ORTAK TD hatasi, ayri-agli IQL/DQN'e gore cok
# daha kirilgan/ossilasyona yatkin. LR'i 1e-4->3e-5 ve target_update'i
# 2000->4000 yapinca AYNI izole test 12000 episode'da 486/600'e MONOTON
# yukseldi, cokme gorulmedi. Bunlari degistirmeden VDN'i egitme.
VDN_LR = 3e-5
VDN_TARGET_UPDATE = 4_000
VDN_EVAL_EVERY = 500

# ---------------------------------------------------------------- QMIX (Asama 7)
# VDN'in Q_tot = Q_1+Q_2 toplamsalligini monotonik bir mixing agiyla degistirir
# (hypernetwork: agirliklari GLOBAL STATE'ten uretir, abs(W) ile monotonluk
# garanti edilir). Per-ajan Q aglari VDN'deki gibi AYRI (agents/vdn.py'deki
# ayrik-ag bulgusu burada da gecerli — ayni nedenle paylasilmiyor). Ayni
# LR/target_update VDN'de stabil oldu, QMIX'te de baslangic noktasi olarak
# kullaniliyor.
# 100x100 NOTU: QMIX episode basina ~0.83s (mixer'in CNN state-kodlayicisi
# ekstra maliyet katiyor). 3000 episode ~41dk -- VDN/IQL ile ADIL kiyaslama
# icin AYNI episode sayisi tutuldu (farkli sayida verilse karsilastirma
# egitim MIKTARI farkindan mi yoksa algoritma farkindan mi geldigi karisir).
QMIX_EPISODES = 3_000
QMIX_BUFFER = 100_000           # state/next_state ekstra yer kapladigi icin VDN'den kucuk
QMIX_BATCH = 32
QMIX_EPS_DECAY_STEPS = 315_000  # gercek olcum ~158 adim/ep; 4000 episode'lik run'da ep~2000'de tabana iner
QMIX_LEARN_START = 1_000
QMIX_LR = 3e-5
QMIX_TARGET_UPDATE = 4_000
QMIX_EVAL_EVERY = 500
QMIX_MIXER_EMBED = 32           # mixer'in gizli katman genisligi

# ---------------------------------------------------------------- gozlem boyutu
# BUYUK GRID NOTU (100x100): tam-grid one-hot kanal (5x100x100=50.000 float,
# %99.99'u sifir) HEM bellek acisindan felaket (replay buffer 300k transition
# icin 56 GB istiyordu) HEM ogrenme acisindan verimsiz (ag neredeyse tamami
# sifir olan bir girdiden anlam cikarmak zorunda). Cozum: ajan SADECE kendi
# YEREL penceresini (PATCH_RADIUS yaricapinda) gorsun + hedefe/diger ajana
# GORELI yon vektorlerini skaler olarak alsin. Bu, buyuk grid navigasyonunda
# standart yaklasimdir ve "100x100'de pathfinding nasil olacak" sorusunun
# dogrudan cevabidir: ajan butun gridi degil, cevresini + yonu gorur.
PATCH_RADIUS = 10              # pencere boyutu = (2*10+1)^2 = 21x21
PATCH_SIZE = 2 * PATCH_RADIUS + 1

OBS_CHANNELS = 2                # yerel: [yasak_bolge, kendi_izi]
N_SCALARS = 11                  # agent_id, faz, t/max, own_row, own_col,
                                # dx_hedef, dy_hedef, dist_hedef, dx_diger,
                                # dy_diger, dist_diger  (hepsi normalize)
OBS_DIM = OBS_CHANNELS * PATCH_SIZE * PATCH_SIZE + N_SCALARS   # 893

STATE_CHANNELS = 2              # yerel: [A1 cevresi yasak-bolge, A2 cevresi yasak-bolge]
STATE_SCALARS = 8               # A1_row,A1_col,A2_row,A2_col,goal_row,goal_col,faz,t/max
STATE_DIM = STATE_CHANNELS * PATCH_SIZE * PATCH_SIZE + STATE_SCALARS   # 890

# ---------------------------------------------------------------- ag mimarisi
# Patch kucuk oldugu icin (21x21) agir bir downsample gerekmiyor — 2 conv
# katmani + kucuk bir AdaptiveAvgPool yeterli. Kanal/havuz boyutu PATCH_SIZE'dan
# bagimsiz (NxN / patch-boyutu genellemesi icin onemli).
CNN_CHANNELS = (16, 32)
CNN_POOL_SIZE = 4

# SKALAR KODLAYICI GENISLIGI — sinyal seyreltmesi duzeltmesi.
# SORUN: CNN dali 32*4*4 = 512 boyut uretiyordu, skalarlar ise 11. Head'e
# giren 523 girdinin sadece 2'si (%0.4) ajana "hedef su yonde" diyen
# dx_hedef/dy_hedef idi. Ustelik FAZ A'da A1 icin ortamda HIC engel yok
# (step()'te A1'in forb kumesi bos, kendi izinin ustunden gecebiliyor) —
# yani A1'in gordugu 512 boyutluk CNN ciktisi neredeyse tamamen bilgisiz
# ve 2 boyutluk gercek sinyali boguyordu.
# COZUM: skalarlar once kucuk bir MLP ile SCALAR_EMBED boyuta acilir, sonra
# CNN ciktisiyla birlestirilir. HAM skalarlar da ayrica ekleniyor (skip
# baglantisi) ki dx/dy'den Q'ya DOGRUDAN lineer bir yol kalsin.
# Yeni oran: 128/(512+128+11) = %20 (onceki %2.1).
SCALAR_EMBED = 128

# ---------------------------------------------------------------- yollar
RUNS_DIR = "runs"
FEASIBILITY_CSV = "runs/feasibility.csv"
DIFFICULTY_CSV = "runs/difficulty.csv"
