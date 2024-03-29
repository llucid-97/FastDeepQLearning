import typing as T, pickle
import numpy as np
from .replay_memory import OversampleError, ReplayMemory
from numpy.lib.format import open_memmap

from pathlib import Path


class NpMmapReplayMemory(ReplayMemory):
    """
    A variant of the replay memory which stores all experiences on disk.
    """

    def __init__(self, maxlen, batch_size, temporal_len, *, log_dir):
        kwargs = dict(locals())
        self.kwargs = dict(kwargs)
        del kwargs["self"], kwargs["log_dir"]
        super().__init__(**kwargs)

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
                data_path = Path(self.kwargs["log_dir"]) / (k)
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
        return open_memmap(str(data_path)+".npy", mode="w+", shape=shape, dtype=dtype)

    def open_existing_memmap(self, data_path):
        return np.load(str(data_path)+".npy", mmap_mode="r+")


class ZarrReplayMemory(NpMmapReplayMemory):

    def create_memmap(self, data_path, shape, dtype):
        import zarr
        return zarr.open(str(data_path)+"_zarr", mode='a', shape=shape, dtype=dtype,
                         chunks=(self._temporal_len,) + tuple(shape[1:]))

    def open_existing_memmap(self, data_path):
        import zarr
        return zarr.open(str(data_path)+"_zarr", mode='a')

    def _temporal_sample_idxes(self, batch, _len):
        slices = [slice(b, b + self._temporal_len) for b in batch]
        outputs = {}
        for k, v in self.memory.items():
            v_slices = [v[s] for s in slices]
            outputs[k] = np.stack(v_slices, axis=1)
            # TODO: Numba optimize these loops G.
        return outputs

class CatReplayMemory(ZarrReplayMemory):
    def create_memmap(self, data_path, shape, dtype):
        import caterva as cat
        fill_value = np.array(0).astype(dtype).tobytes()
        return cat.full(shape, fill_value=fill_value,
                        chunks=(self._temporal_len,) + tuple(shape[1:]),
                        blocks=(2,) + tuple(shape[1:]),
                        urlpath=str(data_path))

    def open_existing_memmap(self, data_path):
        import caterva as cat
        return cat.open(str(data_path))
