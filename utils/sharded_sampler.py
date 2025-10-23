# utils/sharded_sampler.py
"""
Deterministic, resumable sharded order for very large datasets.

- Splits [0..N) into shards of size S.
- Shuffles shard order and intra-shard order deterministically from a seed.
- Maintains a (shard_idx, offset) cursor that can be saved/loaded to resume
  *within* an epoch.
"""

import random
from typing import Dict, Iterator, List, Tuple


class ResumableShardedOrder:
    def __init__(self, n_items: int, shard_size: int, seed: int):
        self.n = int(n_items)
        self.S = max(1, int(shard_size))
        self.seed = int(seed)

        # shard boundaries: (start_index, length)
        self.shards: List[Tuple[int, int]] = [
            (i, min(self.S, self.n - i)) for i in range(0, self.n, self.S)
        ]

        self._build_orders()

        # resume cursor
        self.cur_shard_idx = 0
        self.cur_offset = 0

    def _build_orders(self):
        rng = random.Random(self.seed)
        self.shard_ids = list(range(len(self.shards)))
        rng.shuffle(self.shard_ids)

        # per-shard intra-shuffles
        self.intra = {}
        for sid in range(len(self.shards)):
            base, length = self.shards[sid]
            order = list(range(length))
            rng2 = random.Random(self.seed ^ (sid + 0x9E3779B1))
            rng2.shuffle(order)
            self.intra[sid] = order

    # ------- serialization -------
    def state_dict(self) -> Dict:
        return {
            "n": self.n,
            "S": self.S,
            "seed": self.seed,
            "shard_ids": self.shard_ids,
            "intra": self.intra,
            "cur_shard_idx": self.cur_shard_idx,
            "cur_offset": self.cur_offset,
        }

    def load_state_dict(self, s: Dict):
        assert s["n"] == self.n and s["S"] == self.S and s["seed"] == self.seed, "Sampler shape/seed mismatch"
        self.shard_ids = s["shard_ids"]
        self.intra = s["intra"]
        self.cur_shard_idx = s["cur_shard_idx"]
        self.cur_offset = s["cur_offset"]

    # ------- iteration & advancement -------
    def __iter__(self) -> Iterator[int]:
        n_shards = len(self.shard_ids)
        for si in range(self.cur_shard_idx, n_shards):
            sid = self.shard_ids[si]
            base, length = self.shards[sid]
            order = self.intra[sid]
            start = self.cur_offset if si == self.cur_shard_idx else 0
            for j in range(start, length):
                yield base + order[j]

    def advance(self, k: int):
        """Advance the cursor by k items."""
        remaining = k
        while remaining > 0 and self.cur_shard_idx < len(self.shard_ids):
            sid = self.shard_ids[self.cur_shard_idx]
            _, length = self.shards[sid]
            room = length - self.cur_offset
            step = min(room, remaining)
            self.cur_offset += step
            remaining -= step
            if self.cur_offset >= length:
                self.cur_shard_idx += 1
                self.cur_offset = 0
