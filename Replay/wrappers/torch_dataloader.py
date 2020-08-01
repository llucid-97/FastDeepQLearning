from queue import Queue
import threading
from .wrapper_base_class import ReplayMemoryWrapper
import torch
import typing


class ConfigurationError(Exception): ...


class TorchDataLoader(ReplayMemoryWrapper):
    def __init__(self, replay_buffer, device='cuda:0', precision=torch.float32, use_temporal=True, vectorized=True):
        # Loads data from replay memory in another thread and pre-fetches it to GPu so it's always ready
        ReplayMemoryWrapper.__init__(self, replay_buffer)
        self._temporal_load_q = Queue(maxsize=3)
        self.device, self.precision, self._use_temporal, self.vectorized = device, precision, use_temporal, vectorized
        if use_temporal:
            threading.Thread(target=self._infinite_loop_load).start()
        else:
            raise NotImplementedError("TODO: Add support for pre-fetching and batching non-temporal samples")

    def _infinite_loop_load(self):
        # Asynchronously load data to GPU and queue it
        while True:
            # sample data from replay memory
            experience = self.replay_buffer.temporal_sample()
            if not self.vectorized:
                # for not-Vectorized, experience data structure List[List[Dict[str,Value]]
                # We must transpose to shape {key : [Temporal, Batch, ...]}
                experience = {k: [[experience[i][j][k] for i in range(self.replay_buffer.batch_size)]
                                  for j in range(self.replay_buffer.temporal_len)]
                              for k in experience[0][0]}

            # convert to tensor and copy to device
            experience = {k: torch.tensor(v, dtype=self.precision, device=self.device)
                          for k, v in experience.items()}
            # queue output
            self._temporal_load_q.put(experience)

    def sample(self) -> typing.Dict[str, torch.Tensor]:
        if self._use_temporal:
            raise ConfigurationError("Incorrect Config! Unset `use_temporal` in init to support this feature")
        return self._temporal_load_q.get()

    def temporal_sample(self, *args, **kwargs):
        if not self._use_temporal:
            raise ConfigurationError("Incorrect config! Set `use_temporal` in init to support this feature")
        return self._temporal_load_q.get()
