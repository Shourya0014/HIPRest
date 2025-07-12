# basicsr/utils/model_utils.py

import torch

class ExponentialMovingAverage:
    """EMA helper class that maintains a moving average of model parameters."""

    def __init__(self, parameters, decay=0.999):
        self.shadow = [p.clone().detach() for p in parameters]
        self.decay = decay

    def update(self, parameters):
        for s, p in zip(self.shadow, parameters):
            if p.requires_grad:
                s.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def copy_to(self, parameters):
        for s, p in zip(self.shadow, parameters):
            p.data.copy_(s.data)

    def state_dict(self):
        return {'shadow': self.shadow}

    def load_state_dict(self, state_dict):
        shadow = state_dict['shadow']
        assert len(shadow) == len(self.shadow)
        for s, saved in zip(self.shadow, shadow):
            s.data.copy_(saved.data)
