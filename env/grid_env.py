"""Sirali (turn-based) iki ajanli grid ortami — PLAN §1 ve §Asama 1.

Akis:

    t = 0 ................ T1        T1+1 ................ T
        |<--- FAZ A: A1 ---->|        |<---- FAZ B: A2 ---->|
         A1 hareket eder                A2 hareket eder
         A2 NOOP (golge)                A1 NOOP (golge)
                            ^
                            +-- yasak bolge B burada SABITLENIR

Golge NOOP (PLAN §2.2): sirasi gelmeyen ajan da her adimda gozlem uretir ve
NOOP basar; Q degeri VDN toplamina dahil olur. A2'nin gozlemi FAZ A boyunca
guncellenmeye devam eder (A1 ilerledikce yasak bolge buyur) — VDN'in kredi
atamasi tam olarak bu kanaldan calisir.
"""
from typing import Optional

import numpy as np

from baselines.bfs_oracle import bfs_dist, forbidden_from, manhattan
from config import (AGENT_1, AGENT_2, ALLOW_SAME_START, DIRS, GAMMA, GRID_N,
                    MAX_STEPS_PER_PHASE, MAX_STEPS_TOTAL, NOOP, N_ACTIONS,
                    OBS_DIM, PATCH_RADIUS, R_AGENT_GOAL, R_BLOCKED,
                    R_BOTH_GOAL, R_INVALID, R_OPT_GAP, R_STEP, R_TIMEOUT,
                    SHAPING_COEF, STATE_DIM)

Cell = tuple[int, int]


class MARLGridEnv:
    """Sirali iki fazli grid ortami.

    step() TAKIM odulunu (tek skaler) dondurur — VDN/QMIX boyle ister.
    Ajan basina odul IQL baseline'i icin info["r_ind"] icinde ayrica verilir.
    """

    def __init__(self, n: int = GRID_N,
                 max_steps_per_phase: int = MAX_STEPS_PER_PHASE,
                 allow_same_start: bool = ALLOW_SAME_START,
                 seed: Optional[int] = None):
        self.n = n
        self.max_steps_per_phase = max_steps_per_phase
        self.max_steps_total = 2 * max_steps_per_phase
        self.allow_same_start = allow_same_start
        self.rng = np.random.default_rng(seed)
        self._cells = [(r, c) for r in range(n) for c in range(n)]
        self.reset()

    # ------------------------------------------------------------- konfig

    def sample_config(self) -> tuple[Cell, Cell, Cell]:
        """Rastgele (s1, s2, goal). s1 == s2 serbest; start == goal degil."""
        while True:
            i, j, k = self.rng.integers(0, len(self._cells), size=3)
            s1, s2, g = self._cells[i], self._cells[j], self._cells[k]
            if s1 == g or s2 == g:
                continue
            if not self.allow_same_start and s1 == s2:
                continue
            return s1, s2, g

    # -------------------------------------------------------------- reset

    def reset(self, config: Optional[tuple[Cell, Cell, Cell]] = None,
              seed: Optional[int] = None) -> dict[int, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.s1, self.s2, self.goal = config if config is not None else self.sample_config()

        self.pos = {AGENT_1: self.s1, AGENT_2: self.s2}
        self.path = {AGENT_1: [self.s1], AGENT_2: [self.s2]}
        self.visited = {AGENT_1: {self.s1}, AGENT_2: {self.s2}}

        # Muaf hucreler — PLAN §1'in kritik kurali
        self.exempt = frozenset({self.s1, self.s2, self.goal})

        self.phase = 0            # 0 = FAZ A (A1), 1 = FAZ B (A2)
        self.t = 0                # global adim
        self.phase_t = 0          # icinde bulunulan fazdaki adim
        self.forbidden = frozenset()   # faz sinirinda sabitlenir
        self.done = False
        self.invalid_count = {AGENT_1: 0, AGENT_2: 0}
        self._blocked = False
        self._timeout = False
        return self.observations()

    # ------------------------------------------------------- yardimcilar

    @property
    def active(self) -> int:
        """Sirasi gelen ajan."""
        return AGENT_1 if self.phase == 0 else AGENT_2

    @property
    def passive(self) -> int:
        return AGENT_2 if self.phase == 0 else AGENT_1

    def forbidden_view(self) -> frozenset:
        """Yasak bolgenin O ANKI hali.

        FAZ A'da A1'in su ana kadarki izi (buyumeye devam ediyor), FAZ B'de
        sabitlenmis kume. Tek formul: iz eksi muaf hucreler.
        """
        return frozenset(self.visited[AGENT_1]) - self.exempt

    def _in_bounds(self, c: Cell) -> bool:
        return 0 <= c[0] < self.n and 0 <= c[1] < self.n

    def physical_mask(self, agent: int) -> np.ndarray:
        """Sadece FIZIKSEL gecerlilik: duvar + (A2 icin) yasak bolge.

        done/active durumunu HIC dikkate almaz. Egitim dongusunde bir ajanin
        kendi episode'u zaman asimiyla (timeout) bitince bootstrap icin
        next_state'in GERCEK maskesi lazim — action_mask() o anda sadece
        NOOP dondurur (asagida), bu da bootstrap'i bozar (bkz. Asama 3'teki
        SingleAgentEnv ile ayni tuzak). Bu metod o acigi kapatir.
        """
        mask = np.zeros(N_ACTIONS, dtype=np.float32)
        forb = self.forbidden if agent == AGENT_2 else frozenset()
        cur = self.pos[agent]
        for a, (dr, dc) in enumerate(DIRS):
            nxt = (cur[0] + dr, cur[1] + dc)
            if self._in_bounds(nxt) and nxt not in forb:
                mask[a] = 1.0
        if mask.sum() == 0:          # tamamen kapali kalirsa (olmamali) NOOP ac
            mask[NOOP] = 1.0
        return mask

    def action_mask(self, agent: int) -> np.ndarray:
        """Politika secimi icin: pasif/bitmis ajanda SADECE NOOP."""
        if self.done or agent != self.active:
            mask = np.zeros(N_ACTIONS, dtype=np.float32)
            mask[NOOP] = 1.0
            return mask
        return self.physical_mask(agent)

    # ------------------------------------------------------------ gozlem

    def _local_patch(self, center: Cell, cell_set, radius: int,
                     oob_value: float = 0.0) -> np.ndarray:
        """center etrafinda (2*radius+1)^2 boyutunda yerel pencere.

        BUYUK GRID TASARIMI: ajan butun gridi degil, SADECE cevresini gorur
        (bkz. config.py'nin OBS_CHANNELS notu). Sinir disi hucreler icin
        oob_value kullanilir — yasak-bolge kanalinda 1.0 (duvar gibi davran,
        agan zaten oraya gidemez), izin kanalinda 0.0 (anlamsiz, ziyaret
        edilmemis sayilir).
        """
        r0, c0 = center
        size = 2 * radius + 1
        patch = np.full((size, size), oob_value, dtype=np.float32)
        for dr in range(-radius, radius + 1):
            rr = r0 + dr
            if not (0 <= rr < self.n):
                continue
            for dc in range(-radius, radius + 1):
                cc = c0 + dc
                if 0 <= cc < self.n and (rr, cc) in cell_set:
                    patch[dr + radius, dc + radius] = 1.0
        return patch

    def observe(self, agent: int) -> np.ndarray:
        n = self.n
        own, other = self.pos[agent], self.pos[1 - agent]
        forb_patch = self._local_patch(own, self.forbidden_view(), PATCH_RADIUS, 1.0)
        visited_patch = self._local_patch(own, self.visited[agent], PATCH_RADIUS, 0.0)
        ch = np.stack([forb_patch, visited_patch])

        max_man = 2 * (n - 1)
        scalars = np.array([
            float(agent),
            float(self.phase),
            self.t / self.max_steps_total,
            own[0] / n, own[1] / n,
            (self.goal[0] - own[0]) / n, (self.goal[1] - own[1]) / n,
            manhattan(own, self.goal) / max_man,
            (other[0] - own[0]) / n, (other[1] - own[1]) / n,
            manhattan(own, other) / max_man,
        ], dtype=np.float32)
        return np.concatenate([ch.ravel(), scalars])

    def observations(self) -> dict[int, np.ndarray]:
        return {AGENT_1: self.observe(AGENT_1), AGENT_2: self.observe(AGENT_2)}

    def state(self) -> np.ndarray:
        """QMIX mixer icin: A1 ve A2'nin KENDI cevrelerindeki yasak-bolge
        penceresi + pozisyon skalarlari (tam-grid one-hot DEGIL, ayni buyuk-N
        gerekcesi — bkz. observe())."""
        n = self.n
        forb = self.forbidden_view()
        patch1 = self._local_patch(self.pos[AGENT_1], forb, PATCH_RADIUS, 1.0)
        patch2 = self._local_patch(self.pos[AGENT_2], forb, PATCH_RADIUS, 1.0)
        ch = np.stack([patch1, patch2])
        scalars = np.array([
            self.pos[AGENT_1][0] / n, self.pos[AGENT_1][1] / n,
            self.pos[AGENT_2][0] / n, self.pos[AGENT_2][1] / n,
            self.goal[0] / n, self.goal[1] / n,
            float(self.phase), self.t / self.max_steps_total,
        ], dtype=np.float32)
        return np.concatenate([ch.ravel(), scalars])

    # -------------------------------------------------------------- step

    def step(self, actions) -> tuple[dict[int, np.ndarray], float, bool, dict]:
        """actions: {AGENT_1: a1, AGENT_2: a2} veya tek int (aktif ajanin aksiyonu).

        Pasif ajanin aksiyonu ne verilirse verilsin NOOP'a zorlanir.
        Donen odul TAKIM odulu (tek skaler).
        """
        if self.done:
            raise RuntimeError("Episode bitti — reset() cagir.")

        agent = self.active
        a = actions if isinstance(actions, (int, np.integer)) else actions[agent]
        a = int(a)
        pos_before = self.pos[agent]

        r_team = R_STEP
        r_ind = {AGENT_1: 0.0, AGENT_2: 0.0}
        r_ind[agent] += R_STEP

        # --- hareket
        moved_to = None
        if a == NOOP:
            # aktif ajan icin NOOP maskeli; yine de gelirse gecersiz say
            r_team += R_INVALID
            r_ind[agent] += R_INVALID
            self.invalid_count[agent] += 1
        else:
            dr, dc = DIRS[a]
            cur = self.pos[agent]
            nxt = (cur[0] + dr, cur[1] + dc)
            forb = self.forbidden if agent == AGENT_2 else frozenset()
            if not self._in_bounds(nxt) or nxt in forb:
                r_team += R_INVALID              # duvar / yasak hucre: yerinde kal
                r_ind[agent] += R_INVALID
                self.invalid_count[agent] += 1
            else:
                moved_to = nxt

        if moved_to is not None:
            self.pos[agent] = moved_to
            self.path[agent].append(moved_to)
            self.visited[agent].add(moved_to)

        # --- potential-based reward shaping (Ng ve ark. 1999)
        # Buyuk N'de (100x100) SEYREK terminal odul (sadece hedefe varinca)
        # rastgele/az-egitilmis bir politikayla neredeyse hic yakalanmaz —
        # duzlemde rastgele yurusun hedefe beklenen varis suresi N ile
        # karesel buyur. Bu terim HER ADIMDA hedefe yaklasma/uzaklasmaya
        # orantili yogun bir sinyal verir. Kanitlanmis ozellik: optimal
        # politikayi DEGISTIRMEZ (sadece ayni optimumu daha hizli buldurur),
        # cunku r' = r + gamma*Phi(s')-Phi(s) bicimindeki HERHANGI bir Phi
        # icin butun politikalarin deger sirasi korunur.
        max_man = 2 * (self.n - 1)
        if max_man > 0:
            phi_before = -manhattan(pos_before, self.goal) / max_man
            phi_after = -manhattan(self.pos[agent], self.goal) / max_man
            shaping = SHAPING_COEF * (GAMMA * phi_after - phi_before)
            r_team += shaping
            r_ind[agent] += shaping

        self.t += 1
        self.phase_t += 1
        info: dict = {"phase": self.phase, "active": agent}

        # --- terminal kontrolleri
        if self.phase == 0 and self.pos[AGENT_1] == self.goal:
            r_team += R_AGENT_GOAL
            r_ind[AGENT_1] += R_AGENT_GOAL
            r_team += self._close_phase_a(info)
        elif self.phase == 1 and self.pos[AGENT_2] == self.goal:
            r_team += R_AGENT_GOAL + R_BOTH_GOAL
            r_ind[AGENT_2] += R_AGENT_GOAL
            r_team += self._finish(info)
        elif self.phase_t >= self.max_steps_per_phase:
            r_team += R_TIMEOUT
            self._timeout = True
            self.done = True

        if self.done:
            info.update(self._terminal_info())
        info["r_ind"] = r_ind
        return self.observations(), float(r_team), self.done, info

    def _close_phase_a(self, info: dict) -> float:
        """A1 hedefe vardi: yasak bolgeyi sabitle, A2'nin fizibilitesini BFS ile kontrol et.

        Kilitliyse episode BURADA biter (PLAN §2.2 erken sonlandirma) — ceza
        A1'in hamlelerine gamma^T1 uzaklikta kalir, gamma^(T1+T2) degil.
        """
        self.forbidden = forbidden_from(tuple(self.path[AGENT_1]),
                                        self.s1, self.s2, self.goal)
        d2 = bfs_dist(self.s2, self.goal, self.forbidden, self.n)
        info["forbidden_size"] = len(self.forbidden)
        if d2 is None:
            self._blocked = True
            self.done = True
            return R_BLOCKED
        self.phase = 1
        self.phase_t = 0
        return 0.0

    def _finish(self, info: dict) -> float:
        """A2 hedefe vardi: optimallik cezasini SERBEST Manhattan'a gore yaz.

        BUYUK GRID NOTU (100x100): 5x5'te bu ceza `oracle()`'in TUM optimal
        A1 yollarini enumerate edip bulduğu EN IYI erisilebilir A2 uzunluguna
        gore yaziliyordu (`all_shortest_paths` — C(8,4)=70 yol). 100x100'de
        kose-kose C(198,99) yol var (enumerate edilemez, pratik olarak
        sonsuz). O yuzden buyuk N'de karsilastirma serbest/engelsiz mesafeye
        (`manhattan(s2,goal)`) gore yapiliyor — 5x5'te bu iki olcut %98.1
        oranla zaten AYNIYDI (§0.2), sadece A1'in TEK yolu oldugu nadir
        durumlarda (o zaman zaten secim sansi yok) farklilasiyordu. BFS
        tabanli KILITLEME kontrolu (`_close_phase_a`) etkilenmedi — o zaten
        O(n²) ve enumerate GEREKTIRMIYORDU.
        """
        self.done = True
        len2 = len(self.path[AGENT_2]) - 1
        free_len2 = manhattan(self.s2, self.goal)
        gap2 = max(0, len2 - free_len2)
        return R_OPT_GAP * gap2

    def _terminal_info(self) -> dict:
        len1 = len(self.path[AGENT_1]) - 1
        len2 = len(self.path[AGENT_2]) - 1
        reached1 = self.pos[AGENT_1] == self.goal
        reached2 = self.pos[AGENT_2] == self.goal
        opt1 = manhattan(self.s1, self.goal)
        free_len2 = manhattan(self.s2, self.goal)
        max_man = 2 * (self.n - 1)
        # Zorluk etiketi artik ENUMERATE ETMEDEN, basit bir mesafe esigiyle
        # (PLAN §0.4'un ampirik bulgusu: zorluk d1 buyudukce artiyor). Kesin
        # degil ama enumerate gerektirmez, curriculum agirliklandirmasi icin
        # yeterli bir yaklasik deger.
        is_hard = bool(max_man > 0 and opt1 / max_man >= 0.6)
        return {
            "config": (self.s1, self.s2, self.goal),
            "success": bool(reached1 and reached2),
            "blocked": self._blocked,
            "timeout": self._timeout,
            "len1": len1, "len2": len2 if reached2 else None,
            "gap1": len1 - opt1 if reached1 else None,
            "gap2": (len2 - free_len2) if reached2 else None,
            # A2 serbest optimumundan sapti mi? Kilitlemeden cok daha SIK olan
            # ve asil ogrenme sinyalini tasiyan olcut (bkz. PLAN §0.3).
            "detoured": bool(reached2 and len2 > free_len2),
            "harmed": bool(self._blocked or (reached2 and len2 > free_len2)),
            "oracle_len1": opt1,
            "oracle_len2": free_len2,   # buyuk N'de free_len2 ile ayni (yukaridaki not)
            "free_len2": free_len2,
            "is_hard": is_hard,
            "invalid": dict(self.invalid_count),
            "path1": tuple(self.path[AGENT_1]),
            "path2": tuple(self.path[AGENT_2]),
        }

    # ------------------------------------------------------------ render

    def render(self) -> str:
        forb = self.forbidden_view()
        rows = []
        for r in range(self.n):
            row = []
            for c in range(self.n):
                cell = (r, c)
                if self.pos[AGENT_1] == cell and self.pos[AGENT_2] == cell:
                    row.append("*")
                elif self.pos[AGENT_1] == cell:
                    row.append("1")
                elif self.pos[AGENT_2] == cell:
                    row.append("2")
                elif cell == self.goal:
                    row.append("G")
                elif cell in forb:
                    row.append("#")
                else:
                    row.append(".")
            rows.append(" ".join(row))
        head = (f"faz={'A' if self.phase == 0 else 'B'} t={self.t} "
                f"aktif=A{self.active + 1} yasak={len(forb)}")
        return head + "\n" + "\n".join(rows)
