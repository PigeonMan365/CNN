# utils/cache_io.py
import os, shutil, hashlib
from pathlib import Path
from collections import OrderedDict

def parse_bytes(s: str) -> int:
    s = str(s).strip().upper()
    units = {"B":1, "KB":1024, "MB":1024**2, "GB":1024**3, "TB":1024**4}
    for u in ["TB","GB","MB","KB","B"]:
        if s.endswith(u): return int(float(s[:-len(u)].strip()) * units[u])
    return int(s)

class FileLRU:
    """Simple LRU for staged files under cache_root, evict by total bytes."""
    def __init__(self, root: Path, max_bytes: int):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self._index = OrderedDict()  # key: rel, val: (size)
        self._bytes = 0
        # best-effort scan existing
        for p in self.root.glob("**/*"):
            if p.is_file():
                rel = p.relative_to(self.root).as_posix()
                sz = p.stat().st_size
                self._index[rel] = sz
                self._bytes += sz

    def _evict(self):
        while self._bytes > self.max_bytes and self._index:
            rel, sz = self._index.popitem(last=False)
            try:
                (self.root / rel).unlink(missing_ok=True)
                self._bytes -= sz
            except Exception:
                pass

    def stage(self, src: Path) -> Path:
        """Copy src into cache if missing; return cached path."""
        src = Path(src)
        h = hashlib.sha256(str(src).encode()).hexdigest()
        dst = self.root / h[:2] / h[2:4] / f"{h}{src.suffix.lower()}"
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            sz = dst.stat().st_size
            rel = dst.relative_to(self.root).as_posix()
            # update LRU
            if rel in self._index:
                self._index.move_to_end(rel, last=True)
            else:
                self._index[rel] = sz
                self._bytes += sz
            self._evict()
        else:
            rel = dst.relative_to(self.root).as_posix()
            if rel in self._index:
                self._index.move_to_end(rel, last=True)
        return dst

class TensorLRU:
    """In-RAM decoded tensor cache (byte-capped approx)."""
    def __init__(self, max_bytes: int):
        self.max = max_bytes
        self._bytes = 0
        self._cache = OrderedDict()  # key -> (tensor, nbytes)

    def get(self, key):
        v = self._cache.get(key)
        if v is None: return None
        self._cache.move_to_end(key, last=True)
        return v[0]

    def put(self, key, tensor):
        nbytes = tensor.element_size() * tensor.nelement()
        if key in self._cache:
            _, old = self._cache.pop(key)
            self._bytes -= old
        self._cache[key] = (tensor, nbytes)
        self._bytes += nbytes
        # evict
        while self._bytes > self.max and self._cache:
            _, (_, nb) = self._cache.popitem(last=False)
            self._bytes -= nb
