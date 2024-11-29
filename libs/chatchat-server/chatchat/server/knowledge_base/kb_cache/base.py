import threading
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, List, Tuple, Union, Generator

from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores.faiss import FAISS

from chatchat.utils import build_logger


logger = build_logger()


class ThreadSafeObject:
    def __init__(
        self, key: Union[str, Tuple], obj: Any = None, pool: "CachePool" = None
    ):
        self._obj = obj
        self._key = key
        self._pool = pool
        self._lock = threading.RLock()
        self._loaded = threading.Event()

    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"<{cls}: key: {self.key}, obj: {self._obj}>"

    @property
    def key(self):
        return self._key

    @contextmanager
    def acquire(self, owner: str = "", msg: str = "") -> Generator[None, None, FAISS]:
        owner = owner or f"thread {threading.get_native_id()}"
        try:
            self._lock.acquire()
            if self._pool is not None:
                self._pool._cache.move_to_end(self.key)
            logger.debug(f"{owner} 开始操作：{self.key}。{msg}")
            yield self._obj
        finally:
            logger.debug(f"{owner} 结束操作：{self.key}。{msg}")
            self._lock.release()

    def start_loading(self):
        self._loaded.clear()

    def finish_loading(self):
        self._loaded.set()

    def wait_for_loading(self):
        self._loaded.wait()

    @property
    def obj(self):
        return self._obj

    @obj.setter
    def obj(self, val: Any):
        self._obj = val


class CachePool:
    def __init__(self, cache_num: int = -1):
        self._cache_num = cache_num
        self._cache = OrderedDict()
        self.atomic = threading.RLock()

    def keys(self) -> List[str]:
        return list(self._cache.keys())

    def _check_count(self):
        if isinstance(self._cache_num, int) and self._cache_num > 0:
            while len(self._cache) > self._cache_num:
                self._cache.popitem(last=False)

    def get(self, key: str) -> ThreadSafeObject:
        if cache := self._cache.get(key):
            cache.wait_for_loading()
            return cache

    def set(self, key: str, obj: ThreadSafeObject) -> ThreadSafeObject:
        self._cache[key] = obj
        self._check_count()
        return obj

    def pop(self, key: str = None) -> ThreadSafeObject:
        if key is None:
            return self._cache.popitem(last=False)
        else:
            return self._cache.pop(key, None)

    def acquire(self, key: Union[str, Tuple], owner: str = "", msg: str = ""):
        cache = self.get(key)
        if cache is None:
            raise RuntimeError(f"请求的资源 {key} 不存在")
        elif isinstance(cache, ThreadSafeObject):
            self._cache.move_to_end(key)
            return cache.acquire(owner=owner, msg=msg)
        else:
            return cache
