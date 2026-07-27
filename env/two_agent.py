"""IQL (bagimsiz Q-learning) icin ortak episode calistiricisi — PLAN §Asama 4.

Egitim (train=True) ve degerlendirme (train=False, eps=0) AYNI fonksiyonu
kullanir; boylece iki kod yolu birbirinden sapmaz (Asama 3'te egitim/eval
tutarsizligi bir hata kaynagiydi, burada bastan onleniyor).
"""
from typing import Optional

from config import AGENT_1, AGENT_2, NOOP
from env.grid_env import MARLGridEnv

Cell = tuple[int, int]


def play_episode(env: MARLGridEnv, agents: dict, train: bool,
                 config: Optional[tuple[Cell, Cell, Cell]] = None) -> tuple[dict, dict]:
    """Bir episode'u sonuna kadar oynat.

    agents = {AGENT_1: DQNAgent, AGENT_2: DQNAgent} — HER AJAN KENDI push()/
    learn()'unu SADECE kendi aktif oldugu adimlarda gorur (golge NOOP yok,
    IQL'in tanimi geregi PLAN §Asama 4).

    Donen: (terminal_info, losses). losses = {AGENT_1: [...], AGENT_2: [...]}
    (train=False ise ikisi de bos).
    """
    obs = env.reset(config=config)
    done = False
    losses = {AGENT_1: [], AGENT_2: []}
    info: dict = {}

    while not done:
        agent = env.active
        mask = env.action_mask(agent)
        eps = None if train else 0.0
        a = agents[agent].act(obs[agent], mask, eps=eps)

        prev_active = env.active
        next_obs, _r_team, done, info = env.step(a)

        if train:
            r = info["r_ind"][agent]
            # Bu ajanin KENDI gorevi bitti mi? Ya butun episode bitti (done)
            # ya da faz degisip aktif ajan degisti (A1 basariyla Faz B'ye
            # devretti — kendi episode'u burada biter, bir daha hic oynamaz).
            own_done = done or (env.active != prev_active)
            # done VE timeout ayni anda ise bu bir KESILME (truncation):
            # gercek terminal degil, deger 0 degildir, bootstrap gerekir.
            # (info["timeout"] sadece done=True'da set edilir; agent zaten
            # o an aktif oldugu icin baska ajanin timeout'uyla karismaz.)
            # phase_timeout: A1 sureyi doldurdu ama episode DEVAM ediyor (A2
            # oynasin diye) — done=False oldugundan info["timeout"] burada
            # yok, ama A1 acisindan bu da bir kesilmedir, terminal degil.
            is_truncated = (done and info.get("timeout", False)) \
                or info.get("phase_timeout", False)
            push_done = own_done and not is_truncated
            next_mask = (env.action_mask(agent) if push_done
                        else env.physical_mask(agent))
            agents[agent].push(obs[agent], a, r, next_obs[agent], push_done, next_mask)
            loss = agents[agent].learn()
            if loss is not None:
                losses[agent].append(loss)

        obs = next_obs

    return info, losses


def play_episode_vdn(env: MARLGridEnv, agent, train: bool,
                     config: Optional[tuple[Cell, Cell, Cell]] = None,
                     health_probe: Optional[list] = None) -> tuple[dict, list]:
    """VDN icin ORTAK episode calistirici — PLAN §Asama 5.

    play_episode (IQL, yukarida) ile TEK mimari fark: HER t'de HER IKI ajanin
    (obs, aksiyon) cifti AYNI joint transition'a yazilir (golge NOOP DAHIL —
    pasif ajanin push()'u IQL'de hic yoktu). Boylece agent.learn()'un TEK TD
    hatasi ikisine BIRDEN geri yayilir; bu VDN'i IQL'den ayiran tek sey.

    Ayrica bu tasarim IQL'in own_done/is_truncated (faz-degisimi-farkinda)
    mantigina gerek BIRAKMAZ: joint episode TEK (env.done), butun timestep'ler
    (iki faz + faz siniri dahil) tek dongude ayni sekilde islenir.

    health_probe verilirse (bos liste), FAZ A boyunca Q(obs_2, NOOP) degerlerini
    biriktirir — PLAN'daki saglik kontrolu: bu deger SABIT KALIRSA golge NOOP'un
    gozlemi guncellenmiyordur (en sinsi bug, bkz. PLAN §Asama 5).
    """
    obs = env.reset(config=config)
    done = False
    losses: list = []
    info: dict = {}

    while not done:
        active = env.active
        passive = 1 - active
        mask_active = env.action_mask(active)
        eps = None if train else 0.0
        a_active = agent.act(active, obs[active], mask_active, eps=eps)

        if health_probe is not None and env.phase == 0:
            health_probe.append(agent.q_value(passive, obs[passive], NOOP))

        next_obs, r_team, done, info = env.step(a_active)
        a1 = a_active if active == AGENT_1 else NOOP
        a2 = a_active if active == AGENT_2 else NOOP

        if train:
            is_truncated = done and info.get("timeout", False)
            push_done = done and not is_truncated
            # Bootstrap icin HER ZAMAN fiziksel maske: done=True ise buffer
            # zaten done bayragiyla (1-done) carpip mask'i devre disi birakiyor
            # (JointReplayBuffer.push), done=False ise (faz siniri dahil) bu
            # gercek legal-aksiyon maskesidir — action_mask()'in "pasif ajanda
            # sadece NOOP" collapse'ine BURADA ihtiyac yok.
            nm1 = env.physical_mask(AGENT_1)
            nm2 = env.physical_mask(AGENT_2)
            agent.push(obs[AGENT_1], a1, obs[AGENT_2], a2, r_team,
                      next_obs[AGENT_1], next_obs[AGENT_2], push_done, nm1, nm2)
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)

        obs = next_obs

    return info, losses


def play_episode_qmix(env: MARLGridEnv, agent, train: bool,
                      config: Optional[tuple[Cell, Cell, Cell]] = None,
                      health_probe: Optional[list] = None) -> tuple[dict, list]:
    """QMIX icin ortak episode calistirici — PLAN §Asama 7.

    play_episode_vdn ile TEK fark: her joint transition'a GLOBAL STATE
    (env.state(), t ve t+1) DAHIL edilir — mixer'in hypernetwork'u agirliklarini
    bu state'ten uretir. Geri kalan (golge NOOP, faz-siniri-farkinda olmayan
    tek dongu, fiziksel maske) play_episode_vdn ile birebir ayni.
    """
    obs = env.reset(config=config)
    done = False
    losses: list = []
    info: dict = {}

    while not done:
        active = env.active
        passive = 1 - active
        mask_active = env.action_mask(active)
        eps = None if train else 0.0
        a_active = agent.act(active, obs[active], mask_active, eps=eps)

        if health_probe is not None and env.phase == 0:
            health_probe.append(agent.q_value(passive, obs[passive], NOOP))

        state = env.state()
        next_obs, r_team, done, info = env.step(a_active)
        next_state = env.state()
        a1 = a_active if active == AGENT_1 else NOOP
        a2 = a_active if active == AGENT_2 else NOOP

        if train:
            is_truncated = done and info.get("timeout", False)
            push_done = done and not is_truncated
            nm1 = env.physical_mask(AGENT_1)
            nm2 = env.physical_mask(AGENT_2)
            agent.push(obs[AGENT_1], a1, obs[AGENT_2], a2, r_team,
                      next_obs[AGENT_1], next_obs[AGENT_2], push_done, nm1, nm2,
                      state, next_state)
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)

        obs = next_obs

    return info, losses
