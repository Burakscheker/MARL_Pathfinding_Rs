"""Ogrenmeyen baseline politikalari (PLAN §Asama 8 karsilastirma tablosu).

Onemli ayrim: "rastgele politika" iki farkli sey olabilir.
  - random_walk       : tamamen rastgele yuruyus. Cok kotu, iz devasa buyur.
  - random_shortest   : A1'in optimal yollarindan birini UNIFORM secer.
                        PLAN §0.3'teki kilitleme orani BU politikaya aittir —
                        ortamin dogruluk testi bununla yapilir.
"""
import numpy as np

from baselines.bfs_oracle import (bfs_path, oracle,
                                  sample_random_monotonic_path)
from config import AGENT_1, AGENT_2, DIRS, NOOP


def _path_to_actions(path) -> list[int]:
    """Hucre dizisini aksiyon dizisine cevir."""
    acts = []
    for a, b in zip(path, path[1:]):
        d = (b[0] - a[0], b[1] - a[1])
        acts.append(DIRS.index(d))
    return acts


class ScriptedPolicy:
    """Her faz basinda bir yol secip onu adim adim uygulayan politika."""

    def __init__(self, pick_path1, pick_path2=None, rng=None):
        self.pick_path1 = pick_path1
        self.pick_path2 = pick_path2 or self._bfs_path2
        self.rng = rng or np.random.default_rng()
        self._queue: list[int] = []
        self._phase_seen = None

    @staticmethod
    def _bfs_path2(env):
        return bfs_path(env.s2, env.goal, env.forbidden, env.n)

    def reset(self):
        self._queue = []
        self._phase_seen = None

    def act(self, env) -> int:
        if env.phase != self._phase_seen:
            self._phase_seen = env.phase
            path = self.pick_path1(env) if env.phase == 0 else self.pick_path2(env)
            self._queue = _path_to_actions(path) if path else []
        return self._queue.pop(0) if self._queue else NOOP


def random_shortest_policy(rng=None) -> ScriptedPolicy:
    """A1: optimal yollarindan uniform rastgele biri (enumerate ETMEDEN,
    buyuk N'de de calisir — bkz. sample_random_monotonic_path). A2: BFS."""
    rng = rng or np.random.default_rng()

    def pick(env):
        return sample_random_monotonic_path(env.s1, env.goal, rng)

    return ScriptedPolicy(pick, rng=rng)


def selfish_bfs_policy() -> ScriptedPolicy:
    """A1: BFS'in ilk buldugu yol (A2'yi hic dusunmez). A2: BFS."""
    return ScriptedPolicy(lambda env: bfs_path(env.s1, env.goal))


def oracle_policy() -> ScriptedPolicy:
    """A1: A2'ye en cok yer birakan optimal yol. Ust sinir."""
    def pick(env):
        orc = oracle(env.s1, env.s2, env.goal)
        return orc.best_path1 or bfs_path(env.s1, env.goal)
    return ScriptedPolicy(pick)


def random_walk_act(env, rng) -> int:
    """Maskeye uyan tamamen rastgele aksiyon."""
    mask = env.action_mask(env.active)
    legal = np.flatnonzero(mask)
    return int(rng.choice(legal))
