import typing as T, pickle
import numpy as np
from .replay_memory import OversampleError, ReplayMemory
from numpy.lib.format import open_memmap
import zarr, caterva as cat
from pathlib import Path


class ZarrReplayMemory(ReplayMemory):
    """
    A variant of the replay memory which stores all experiences on disk.
    """

    def __init__(self, maxlen, batch_size, temporal_len, *, log_dir):
        kwargs = dict(locals())
        self.kwargs = dict(kwargs)
        del kwargs["self"], kwargs["log_dir"]
        super().__init__(**kwargs)
        self.memory: T.Dict[str, zarr.Array] = {}

    def _jit_initialize(self, experience_dict: dict):
        """This method is called the first time an experience is added.
        It initializes the memory data structure to have the right shape and dtype to store all subsequent entries"""
        metadata_path = Path(self.kwargs["log_dir"]) / "metadata.pkl"
        self.metadata = {}
        if metadata_path.exists():
            with open(metadata_path, mode="rb") as f:
                self.metadata = pickle.load(f)
            assert all([k in self.metadata for k in experience_dict]), "Memory doesn't match config!"
            for k, v in experience_dict.items():
                data_path = Path(self.kwargs["log_dir"]) / (k + ".npy")
                self.memory[k] = self.open_existing_memmap(data_path)
                shape = list(self.memory[k].shape)
                shape[0] = self._maxlen
                self.memory[k].resize(*shape)

        else:
            for k, v in experience_dict.items():
                if isinstance(v, np.ndarray):
                    shape = (self._maxlen,) + tuple(v.shape)
                    dtype = v.dtype
                else:
                    assert np.isclose(np.float32(v),
                                      v), "Anything thats not a numpy array must be representable as a float32 for numeric stability"
                    shape = (self._maxlen, 1)
                    dtype = np.float32

                data_path = Path(self.kwargs["log_dir"]) / (k + ".npy")
                assert not data_path.exists(), "[Memmap Replay]: Prior data exists but we have no metadata for it!"
                self.memory[k] = self.create_memmap(data_path, shape, dtype)

                self.metadata[k] = {"shape": shape, "dtype": dtype}

    def create_memmap(self, data_path, shape, dtype):
        return zarr.open(str(data_path), mode='a', shape=shape, dtype=dtype,
                         chunks=(self._temporal_len,) + tuple(shape[1:]))

    def open_existing_memmap(self, data_path):
        return zarr.open(str(data_path), mode='a')


class NpMmapReplayMemory(ZarrReplayMemory):
    def create_memmap(self, data_path, shape, dtype):
        return open_memmap(str(data_path), mode="w+", shape=shape, dtype=dtype)

    def open_existing_memmap(self, data_path):
        return np.load(str(data_path), mmap_mode="r+")


class CatReplayMemory(ZarrReplayMemory):
    def create_memmap(self, data_path, shape, dtype):
        fill_value = np.array(0).astype(dtype).tobytes()
        return cat.full(shape, fill_value=fill_value,
                        chunks=(self._temporal_len,) + tuple(shape[1:]),
                        blocks=(2,) + tuple(shape[1:]),
                        urlpath=str(data_path))

    def open_existing_memmap(self, data_path):
        return cat.open(str(data_path))
