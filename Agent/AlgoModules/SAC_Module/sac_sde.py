"""
MIT License

Copyright (c) 2018 Pranjal Tandon
Copyright (c) 2019 Antonin Raffin
Copyright (c) 2020 Gershom Agim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from .sac_baseline import SoftActorCriticModule
from .models import make_sde_actor

class SDESoftActorCriticModule(SoftActorCriticModule):
    """Soft actor critic with state dependent exploration"""

    def __init__(self, *args, **kwargs):
        super(SDESoftActorCriticModule, self).__init__(*args, **kwargs, actor_factory=make_sde_actor)
        NotImplementedError("SDE is not yet in a working state. DO not use. Only merged changes to stash progress")
        self._step = 0

    def update_target(self):
        super(SDESoftActorCriticModule, self).update_target()
        self.actor.reset_sde()
        self.target_actor.reset_sde()

    def act(self, state):
        if (self._step % self.conf.sde_update_interval) == 0: self.actor.reset_sde()
        return self.actor.forward(state)
