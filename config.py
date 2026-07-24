"""Tum hiperparametreler ve sabitler burada. Koda deger gomme.

PLAN.md ile senkron tut: ozellikle odul degerleri (§5) ve max_steps (§Asama 1).
"""

# ---------------------------------------------------------------- grid / ortam
GRID_N = 5
N_AGENTS = 2

AGENT_1 = 0          # once giden, izi yasak bolge olan ajan
AGENT_2 = 1          # yasak bolgeden kacinarak giden ajan

# Aksiyonlar
UP, RIGHT, DOWN, LEFT, NOOP = range(5)
N_ACTIONS = 5
ACTION_NAMES = ("UP", "RIGHT", "DOWN", "LEFT", "NOOP")

# (d_satir, d_sutun) — NOOP dahil degil, o ayri ele aliniyor
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))

# Faz basina adim limiti. 5x5'te en uzun optimal yol 8; 15 fazlasiyla yeter.
MAX_STEPS_PER_PHASE = 15
MAX_STEPS_TOTAL = 2 * MAX_STEPS_PER_PHASE

# Ajanlar ayni hucreden baslayabilir mi (PLAN §0.1: 600 konfig)
ALLOW_SAME_START = True

# ---------------------------------------------------------------- odul (PLAN §5)
R_STEP = -0.05        # her timestep
R_INVALID = -0.10     # duvara / yasak hucreye hamle (ajan yerinde kalir)
R_AGENT_GOAL = +1.0   # bir ajan hedefe vardi (ara terminal)
R_BOTH_GOAL = +3.0    # ikisi de vardi (takim hedefi)
R_BLOCKED = -3.0      # A2 kilitlendi — faz sinirinda BFS ile tespit
R_TIMEOUT = -3.0      # herhangi bir fazda max_steps asildi
# Katsayi: R_OPT_GAP * (len2 - ORACLE(s1,s2,g)).
# Tarama sonucu (baselines/scan.py): random-shortest A1 altinda A2'nin
# KILITLENME olasiligi sadece %0.82, ama UZAMA olasiligi %13.3. Yani asil
# ogrenme sinyalini bu terim tasiyor, R_BLOCKED degil. Tipik uzama 2 adim ->
# -1.0 ceza, +3.0'lik takim bonusuna karsi anlamli bir agirlik.
R_OPT_GAP = -0.50

# ---------------------------------------------------------------- egitim (ortak)
SEED = 0
GAMMA = 0.99
LR = 5e-4
GRAD_CLIP = 10.0
HIDDEN = 128

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
DQN_BATCH = 128                # transition
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
IQL_EPISODES = 40_000
IQL_BUFFER = 150_000           # transition, ajan basina
IQL_BATCH = 128
IQL_EPS_DECAY_STEPS = 40_000   # adim, ajan basina
IQL_LEARN_START = 1_000
IQL_LR = 1e-4
IQL_TARGET_UPDATE = 2_000      # adim
IQL_EVAL_EVERY = 4_000         # episode

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
VDN_EPISODES = 40_000
VDN_BUFFER = 300_000           # ortak (joint) transition — her satir BIR global timestep
VDN_BATCH = 128
# DIKKAT: VDN'in TEK paylasilan agi her t'de (HER IKI ajan icin de) sayaci
# ilerletir; IQL'in agent basina AYRI sayaclari sadece o ajanin AKTIF oldugu
# yarim fazda ilerliyordu. Yani ayni sayisal esik VDN'de ~2x daha HIZLI
# (episode cinsinden) tukeniyor. Ilk 40k-episode kosuda bu 40_000 degeriyle
# epsilon ep~2500'de tabana indi (IQL'de ep~6000-7000) ve A1 pek cok konfigde
# 2-hucreli bir Q-salinimina (asagi/yukari sonsuz dongu) kilitlenip kacamadi
# (reached1_frac sadece %17.1 cikti). 80_000, VDN'in episode-basina decay
# hizini IQL'inkiyle YAKLASIK esitler.
VDN_EPS_DECAY_STEPS = 80_000   # adim (paylasilan agin GLOBAL adim sayaci)
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
VDN_EVAL_EVERY = 4_000

# ---------------------------------------------------------------- gozlem boyutu
OBS_CHANNELS = 5               # own, other, goal, forbidden, own_visited
N_SCALARS = 4                  # agent_id, phase, t/max, kalan_manhattan/max
OBS_DIM = OBS_CHANNELS * GRID_N * GRID_N + N_SCALARS      # 129

STATE_CHANNELS = 4             # pos1, pos2, goal, forbidden
STATE_SCALARS = 2              # phase, t/max
STATE_DIM = STATE_CHANNELS * GRID_N * GRID_N + STATE_SCALARS   # 102

# ---------------------------------------------------------------- yollar
RUNS_DIR = "runs"
FEASIBILITY_CSV = "runs/feasibility.csv"
DIFFICULTY_CSV = "runs/difficulty.csv"
