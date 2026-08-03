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

from baselines.bfs_oracle import (bfs_dist, bfs_distance_map, forbidden_from,
                                  manhattan)
from config import (AGENT_1, AGENT_2, ALLOW_SAME_START, DIRS, GAMMA, GRID_N,
                    MAX_STEPS_PER_PHASE, MAX_STEPS_TOTAL, NOOP, N_ACTIONS,
                    OBS_DIM, PATCH_RADIUS, R_AGENT_GOAL, R_BLOCKED,
                    R_BOTH_GOAL, R_INVALID, R_OPT_GAP, R_REVISIT, R_STEP,
                    R_TIMEOUT, SHAPING_COEF, STATE_DIM)
from env.obstacles import ObstacleConfig, generate_obstacle_config

Cell = tuple[int, int]

# CLI/tag'lerimiz "easy" kullaniyor, uretecin karsiligi "basic".
OBSTACLE_DIFFICULTY_ALIAS = {"easy": "basic"}


class MARLGridEnv:
    """Sirali iki fazli grid ortami.

    step() TAKIM odulunu (tek skaler) dondurur — VDN/QMIX boyle ister.
    Ajan basina odul IQL baseline'i icin info["r_ind"] icinde ayrica verilir.
    """

    def __init__(self, n: int = GRID_N,
                 max_steps_per_phase: int = MAX_STEPS_PER_PHASE,
                 allow_same_start: bool = ALLOW_SAME_START,
                 seed: Optional[int] = None,
                 difficulty: Optional[str] = None):
        """difficulty: None (varsayilan, engelsiz), "easy"/"medium"/"hard"
        (bkz. env/obstacles.py) -- verilirse reset() haritayi motif/labirent
        tabanli uretecten alir (basit "easy" -> uretecin "basic" adi)."""
        self.n = n
        self.max_steps_per_phase = max_steps_per_phase
        self.max_steps_total = 2 * max_steps_per_phase
        self.allow_same_start = allow_same_start
        self.difficulty = difficulty
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

    def reset(self, config=None, seed: Optional[int] = None,
              map_seed: Optional[int] = None) -> dict[int, np.ndarray]:
        """config: (s1, s2, goal) UCLUSU ya da bir ObstacleConfig.

        ENGEL MODU (self.difficulty verilmisse): harita env/obstacles.py'nin
        uretecinden gelir ve baslangic/hedef konumlari da HARITAYLA BIRLIKTE
        uretilir (uretec bunlari haritanin gecitlerine gore secer, sonradan
        rastgele serpistirmez). Uretec her haritayi BFS ile dogrular: A1 faz
        limitinde hedefe varabilmeli VE A1'in izi sabitlendikten sonra A2 de
        varabilmeli -- yani cozumsuz episode uretilmez.

        map_seed verilirse O haritayi deterministik uretir (degerlendirme
        icin: her algoritma AYNI haritalarda olculur). Verilmezse env'in
        kendi rng'sinden rastgele bir harita cekilir (egitim icin: her
        episode taze harita, ezberlenecek sabit havuz yok). Uretim maliyeti
        olculdu: harita basina 0.006-0.019s, 6000 episode'a ~0.5-2 dk ekler.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        obstacle_cfg = config if isinstance(config, ObstacleConfig) else None
        if obstacle_cfg is None and self.difficulty is not None and config is None:
            if map_seed is None:
                map_seed = int(self.rng.integers(0, 2**31 - 1))
            obstacle_cfg = generate_obstacle_config(
                OBSTACLE_DIFFICULTY_ALIAS.get(self.difficulty, self.difficulty),
                map_seed=map_seed, n=self.n,
                max_steps=self.max_steps_per_phase)

        if obstacle_cfg is not None:
            self.s1, self.s2, self.goal = (obstacle_cfg.start1,
                                          obstacle_cfg.start2, obstacle_cfg.goal)
        else:
            self.s1, self.s2, self.goal = (config if config is not None
                                           else self.sample_config())

        self.pos = {AGENT_1: self.s1, AGENT_2: self.s2}
        self.path = {AGENT_1: [self.s1], AGENT_2: [self.s2]}
        self.visited = {AGENT_1: {self.s1}, AGENT_2: {self.s2}}

        # Muaf hucreler — PLAN §1'in kritik kurali
        self.exempt = frozenset({self.s1, self.s2, self.goal})

        # Statik engeller. Uretec bunlari zaten dogruladi (cozulebilirlik
        # garanti), o yuzden burada ekstra BFS kontrolu YOK.
        self.walls: frozenset = (obstacle_cfg.obstacles - self.exempt
                                 if obstacle_cfg is not None else frozenset())
        self.obstacle_cfg = obstacle_cfg   # subtype/map_seed gorsellestirme icin

        # Duvar-farkinda shaping icin hedeften TEK BFS ile mesafe haritasi
        # (bkz. step()'teki shaping bloku). FAZ A boyunca A1'in tek engeli
        # duvarlar (sabit) — bu yuzden burada, episode basinda BIR KEZ
        # cikarilip fazin sonuna kadar yeniden kullanilir. A2'ninki (walls
        # + forbidden) _close_phase_a()'da, forbidden sabitlenince cikarilir.
        self._dist_map = {AGENT_1: bfs_distance_map(self.goal, self.walls, self.n)}

        self.phase = 0            # 0 = FAZ A (A1), 1 = FAZ B (A2)
        self.t = 0                # global adim
        self.phase_t = 0          # icinde bulunulan fazdaki adim
        self.forbidden = frozenset()   # faz sinirinda sabitlenir
        self.done = False
        self.invalid_count = {AGENT_1: 0, AGENT_2: 0}
        self._blocked = False
        self._timeout = False
        self._timeout1 = False    # A1 FAZ A'yi hedefe varmadan tuketti mi
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
        forb = self.walls | (self.forbidden if agent == AGENT_2 else frozenset())
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

        BUG (bulundu ve duzeltildi): eskiden `np.full(..., oob_value)` TUM
        yamayi (sinir ICI hucreler DAHIL) oob_value ile dolduruyordu, sonra
        dongu SADECE cell_set uyelerini 1.0'a CEKIYORDU — sinir ici ama
        cell_set'te OLMAYAN hucreler asla 0.0'a donmuyordu. oob_value=1.0
        olan yasak-bolge kanalinda bu, TUM yamayi (gercek yasak bolge BOS
        olsa bile) 1.0 gosteriyordu — olcduldu: bos yasak bolgeyle patch
        toplami 441/441. Ajan hic bir zaman gercek yasak-bolge sinirini
        goremiyordu, kanal sabit/bilgisizdi. Duzeltme: sinir ICI hucreler
        varsayilan 0.0'dan baslar, SADECE gercekten sinir DISI olanlar
        oob_value alir.
        """
        r0, c0 = center
        size = 2 * radius + 1
        patch = np.zeros((size, size), dtype=np.float32)
        for dr in range(-radius, radius + 1):
            rr = r0 + dr
            for dc in range(-radius, radius + 1):
                cc = c0 + dc
                if not (0 <= rr < self.n and 0 <= cc < self.n):
                    patch[dr + radius, dc + radius] = oob_value
                elif (rr, cc) in cell_set:
                    patch[dr + radius, dc + radius] = 1.0
        return patch

    def observe(self, agent: int) -> np.ndarray:
        n = self.n
        own, other = self.pos[agent], self.pos[1 - agent]
        forb_patch = self._local_patch(own, self.walls | self.forbidden_view(), PATCH_RADIUS, 1.0)
        visited_patch = self._local_patch(own, self.visited[agent], PATCH_RADIUS, 0.0)
        ch = np.stack([forb_patch, visited_patch])

        max_man = 2 * (n - 1)

        # ENGEL-FARKINDA yon bilgisi (bkz. config.py N_SCALARS notu).
        # Yukaridaki dx/dy/dist skalarlari DUZ CIZGI (Manhattan) — engelin
        # etrafindan hangi taraftan dolasilacagini SOYLEMEZ. Bu 5 deger
        # BFS mesafe haritasindan okunuyor: kendi hucre + 4 komsu. Ajan
        # boylece "hangi komsu hedefe GERCEKTEN yaklastiriyor" bilgisini
        # dogrudan goruyor.
        # FAZ A'da A2'nin kendi haritasi (walls+forbidden) HENUZ YOK —
        # yasak bolge daha sabitlenmedi (_close_phase_a'da olusuyor).
        # O ana kadar A1'in duvar-tabanli haritasina dusuluyor: A2 pasif
        # (golge NOOP) oldugu icin kararlarini etkilemez, ama gozlem boyutu
        # ve olcegi tutarli kalir.
        dmap = self._dist_map.get(agent) or self._dist_map[AGENT_1]

        # KODLAMA NOTU: komsulari MUTLAK normalize mesafe olarak vermek ise
        # yaramiyor -- 5 deger de birbirine neredeyse esit cikiyor (olculdu:
        # [0.337, 0.327, 0.347, 0.347, 0.327]), ayirt edici fark sadece
        # 1/max_man ~ %1. Ag bu %1'lik farki yakalamak zorunda kaliyor.
        # Onun yerine KOMSU FARKI veriliyor: d_own - d_komsu, yani +1 = bu
        # komsu bir adim YAKIN, -1 = bir adim UZAK. Yuksek kontrastli, dogrudan
        # aksiyon secimine karsilik gelen sinyal. Kendi mutlak mesafe ayrica
        # baglam olarak kaliyor (hedefe ne kadar kaldi).
        d_own = dmap.get(own)
        own_norm = 1.0 if d_own is None else min(1.0, d_own / max_man)

        bfs_feats = [own_norm]
        for dr, dc in DIRS:
            d_nb = dmap.get((own[0] + dr, own[1] + dc))
            if d_nb is None or d_own is None:
                bfs_feats.append(-1.0)      # ulasilamaz/engel: mumkun en kotu
            else:
                bfs_feats.append(float(d_own - d_nb))   # +1 yakin, -1 uzak

        scalars = np.array([
            float(agent),
            float(self.phase),
            self.t / self.max_steps_total,
            own[0] / n, own[1] / n,
            (self.goal[0] - own[0]) / n, (self.goal[1] - own[1]) / n,
            manhattan(own, self.goal) / max_man,
            (other[0] - own[0]) / n, (other[1] - own[1]) / n,
            manhattan(own, other) / max_man,
            *bfs_feats,
        ], dtype=np.float32)
        return np.concatenate([ch.ravel(), scalars])

    def observations(self) -> dict[int, np.ndarray]:
        return {AGENT_1: self.observe(AGENT_1), AGENT_2: self.observe(AGENT_2)}

    def state(self) -> np.ndarray:
        """QMIX mixer icin: A1 ve A2'nin KENDI cevrelerindeki yasak-bolge
        penceresi + pozisyon skalarlari (tam-grid one-hot DEGIL, ayni buyuk-N
        gerekcesi — bkz. observe())."""
        n = self.n
        forb = self.walls | self.forbidden_view()
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
            forb = self.walls | (self.forbidden if agent == AGENT_2 else frozenset())
            if not self._in_bounds(nxt) or nxt in forb:
                r_team += R_INVALID              # duvar / yasak hucre: yerinde kal
                r_ind[agent] += R_INVALID
                self.invalid_count[agent] += 1
            else:
                moved_to = nxt
                if nxt in self.visited[agent]:
                    # gitgel/dongu: bu fazda daha once ugranmis bir hucreye
                    # geri donuyor — ilerlemeden salinim yapmayi cezalar.
                    r_team += R_REVISIT
                    r_ind[agent] += R_REVISIT

        if moved_to is not None:
            self.pos[agent] = moved_to
            self.path[agent].append(moved_to)
            self.visited[agent].add(moved_to)

        self.t += 1
        self.phase_t += 1
        info: dict = {"phase": self.phase, "active": agent}

        # --- terminal kontrolleri
        if self.phase == 0 and self.pos[AGENT_1] == self.goal:
            r_team += R_AGENT_GOAL
            r_ind[AGENT_1] += R_AGENT_GOAL
            r_team += self._close_phase_a(info)
        elif self.phase == 1 and self.pos[AGENT_2] == self.goal:
            r_team += R_AGENT_GOAL
            r_ind[AGENT_2] += R_AGENT_GOAL
            # "IKISI de vardi" bonusu sadece A1 GERCEKTEN vardiysa hak edilir.
            # A1 timeout edip de faz B'ye gecildigi durumda (bkz. asagidaki
            # timeout dali) takim bu bonusu almamali — yoksa A1'in basarisiz
            # oldugu episode'lar da odullenir.
            if not self._timeout1:
                r_team += R_BOTH_GOAL
            r_team += self._finish(info)
        elif self.phase_t >= self.max_steps_per_phase:
            r_team += R_TIMEOUT
            # Timeout, ajanin KENDI navigasyon basarisizligi (koordinasyon
            # cezasi degil) — bu yuzden r_ind'e de yaziliyor, yoksa IQL'in
            # ajanlari "sure doldu" sinyalini HIC gormuyordu.
            r_ind[agent] += R_TIMEOUT
            self._timeout = True
            if self.phase == 0:
                # A1 hedefe VARAMADI. Eskiden episode burada tamamen biterdi,
                # yani A2'nin sirasi HIC gelmiyordu: A1 basarisiz oldugu surece
                # A2 tek bir egitim adimi bile goremiyordu (zincirin ilk halkasi
                # kopuk). Artik yasak bolge A1'in KISMI izinden sabitlenip FAZ B
                # yine de basliyor; A1'in basarisizligi _timeout1 ile ayrica
                # kaydedildigi icin metrikler (reached1/gap1/success) bozulmuyor.
                self._timeout1 = True
                info["phase_timeout"] = True
                r_team += self._close_phase_a(info)
            else:
                self.done = True

        # --- potential-based reward shaping (Ng, Harada, Russell 1999)
        # Buyuk N'de (100x100) SEYREK terminal odul (sadece hedefe varinca)
        # rastgele/az-egitilmis bir politikayla neredeyse hic yakalanmaz —
        # duzlemde rastgele yurusun hedefe beklenen varis suresi N ile
        # karesel buyur. Bu terim HER ADIMDA hedefe yaklasma/uzaklasmaya
        # orantili yogun bir sinyal verir.
        #
        # DUVAR-FARKINDA MESAFE (bulundu ve duzeltildi): eskiden duz Manhattan
        # kullaniliyordu — duvar/yasak-bolge YOKKEN doğruydu, ama duvar VARKEN
        # ajani DUVARA DOGRU (yanlis yone) itebiliyordu, cunku Manhattan
        # engeli gormez. Artik reset()/_close_phase_a()'da hedeften cikarilan
        # BFS mesafe haritasindan (self._dist_map) okunuyor — GERCEK en kisa
        # yol mesafesi, duvarin etrafindan dolanmayi dogru yansitir. Duvarsiz
        # halde (self.walls bos) BFS mesafesi Manhattan'la MATEMATIKSEL
        # OZDESTIR (4-yonlu acik gridde), yani eski davranis DEGISMEDEN korunur.
        #
        # POLICY-INVARIANCE KOSULU: telescoping ozdesligi G' = G +
        # gamma^T*Phi(s_T) - Phi(s_0) verir. Phi(s_0) sabittir (ayni
        # baslangic), ama Phi(s_T) POLITIKAYA GORE DEGISIRSE (bizde oyle:
        # basari->Phi=1, timeout/blocked -> degisken pozisyon) kesin sira
        # garantisi duser. Guvenli kosul: Phi(terminal)=0 TUM terminal
        # durumlar icin — bu yuzden shaping SADECE bu adim episode'u
        # BITIRMIYORSA uygulanir (self.done, TERMINAL KONTROLLERINDEN SONRA
        # okunuyor — yukarida henuz belirleniyordu, o yuzden bu blok BURADA,
        # terminal bloktan SONRA). A1'in FAZ A'yi hedefe vararak kapatmasi
        # (_close_phase_a, done=False kalir) terminal DEGILDIR, shaping
        # normal uygulanir — sadece episode'un GERCEK sonu (done=True) hariç.
        # Bu ANY Phi icin gecerli (Ng ve ark.) — metrik degisimi kaniti bozmaz.
        max_man = 2 * (self.n - 1)
        if max_man > 0 and not self.done:
            dmap = self._dist_map[agent]
            d_before = dmap.get(pos_before, max_man)
            d_after = dmap.get(self.pos[agent], max_man)
            phi_before = 1.0 - d_before / max_man
            phi_after = 1.0 - d_after / max_man
            shaping = SHAPING_COEF * (GAMMA * phi_after - phi_before)
            r_team += shaping
            r_ind[agent] += shaping

        if self.done:
            info.update(self._terminal_info())
        info["r_ind"] = r_ind
        return self.observations(), float(r_team), self.done, info

    def _close_phase_a(self, info: dict) -> float:
        """FAZ A kapanisi: yasak bolgeyi sabitle, A2'nin fizibilitesini BFS ile kontrol et.

        IKI durumda cagrilir: (a) A1 hedefe vardi, (b) A1 sureyi doldurdu
        (timeout). (b)'de yasak bolge A1'in KISMI izinden olusur — A2 yine de
        oynar ki egitim verisi toplayabilsin.

        Kilitliyse episode BURADA biter (PLAN §2.2 erken sonlandirma) — ceza
        A1'in hamlelerine gamma^T1 uzaklikta kalir, gamma^(T1+T2) degil.
        """
        self.forbidden = forbidden_from(tuple(self.path[AGENT_1]),
                                        self.s1, self.s2, self.goal)
        blocked2 = self.forbidden | self.walls
        d2 = bfs_dist(self.s2, self.goal, blocked2, self.n)
        info["forbidden_size"] = len(self.forbidden)
        if d2 is None:
            self._blocked = True
            self.done = True
            return R_BLOCKED
        # A2'nin shaping haritasi: forbidden ARTIK SABIT (faz B boyunca
        # degismez), bu yuzden burada BIR KEZ cikarilir (bkz. reset()'teki
        # A1 icin ayni gerekce).
        self._dist_map[AGENT_2] = bfs_distance_map(self.goal, blocked2, self.n)
        self.phase = 1
        self.phase_t = 0
        return 0.0

    def _wall_only_dist(self, cell: Cell) -> int:
        """cell -> hedef, SADECE duvarlardan kacinarak (A1'in izi HARIC).

        "A1 hic olmasaydi kac adim gerekirdi" referansi. self._dist_map[AGENT_1]
        reset()'te hedeften duvarlarla BFS ile cikariliyor; ayni harita hem A1
        hem A2 icin gecerli (ikisinin de statik engeli sadece duvarlar).
        Duvar yokken bu deger Manhattan'a BIREBIR esittir -- yani duvarsiz
        kosularda davranis DEGISMEZ.
        """
        d = self._dist_map[AGENT_1].get(cell)
        return manhattan(cell, self.goal) if d is None else d

    def _finish(self, info: dict) -> float:
        """A2 hedefe vardi: optimallik cezasini DUVAR-FARKINDA optimuma gore yaz.

        BUG (bulundu ve duzeltildi): referans eskiden serbest Manhattan'di,
        yani duvarlari YOK SAYIYORDU. Duvar modunda A2'nin mecburen yaptigi
        her dolasma "zarar" olarak yaziliyordu -- olculdu: A2 gercekte TAM
        optimal giderken (duvar-farkinda gap +0.00) serbest-Manhattan olcutu
        ortalama +2.85 gap ve %52.5 zarar raporluyordu. Zarar orani projenin
        ASIL arastirma metrigi ("A1, A2'yi engelledi mi") oldugu icin bu,
        duvarli tum sonuclari kullanilamaz kiliyordu.

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
        gap2 = max(0, len2 - self._wall_only_dist(self.s2))
        return R_OPT_GAP * gap2

    def _terminal_info(self) -> dict:
        len1 = len(self.path[AGENT_1]) - 1
        len2 = len(self.path[AGENT_2]) - 1
        reached1 = self.pos[AGENT_1] == self.goal
        reached2 = self.pos[AGENT_2] == self.goal
        # DUVAR-FARKINDA referanslar (bkz. _wall_only_dist / _finish notu):
        # "A1 hic olmasaydi kac adim gerekirdi". Duvarsizken Manhattan'a
        # birebir esit, yani duvarsiz kosularin metrikleri DEGISMEZ.
        opt1 = self._wall_only_dist(self.s1)
        free_len2 = self._wall_only_dist(self.s2)
        max_man = 2 * (self.n - 1)
        # Zorluk etiketi (metriklerin zor/kolay ayrimi icin).
        # ENGEL MODUNDA: olcut DOLAMBAC ORANI -- engellerin A1'i optimalden
        # ne kadar saptirdigi (env/obstacles.py'nin uretecinin kendi "hard"
        # tanimiyla AYNI esik: >= 1.5). Eski mesafe esigi (opt1/max_man>=0.6)
        # burada ISE YARAMIYOR: olculdu, uretecin medium/hard haritalarinda
        # baslangic-hedef Manhattan mesafesi kisa (ort. 31 ve 15), esigi
        # HICBIR konfig asmiyor ve "zor alt-kume" bos kaliyordu (n=0, nan).
        # ENGELSIZ MODDA: eski mesafe esigi (dolambac zaten hep 1.0 olurdu).
        if self.walls:
            man1 = manhattan(self.s1, self.goal)
            is_hard = bool(man1 > 0 and opt1 / man1 >= 1.5)
        else:
            is_hard = bool(max_man > 0 and opt1 / max_man >= 0.6)
        return {
            "config": (self.s1, self.s2, self.goal),
            "success": bool(reached1 and reached2),
            "blocked": self._blocked,
            "timeout": self._timeout,
            "timeout1": self._timeout1,   # A1 FAZ A'yi hedefe varmadan tuketti
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
                elif cell in self.walls:
                    row.append("W")
                elif cell in forb:
                    row.append("#")
                else:
                    row.append(".")
            rows.append(" ".join(row))
        head = (f"faz={'A' if self.phase == 0 else 'B'} t={self.t} "
                f"aktif=A{self.active + 1} yasak={len(forb)} duvar={len(self.walls)}")
        return head + "\n" + "\n".join(rows)
