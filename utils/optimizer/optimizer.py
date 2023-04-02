from omegaconf import DictConfig
from torch.optim import Optimizer

class LitOptimizer(object):
    def __init__(
        self,
        cfg:DictConfig=None
    ):
        self.cfg = cfg.optimizer
        self.lr = self.cfg.lr if self.cfg.lr != None else self.cfg.default_lr
        self.weight_decay = self.cfg.weight_decay if self.cfg.weight_decay != None else self.cfg.default_weight_decay
    
    @property
    def optimizer(
        self,
        params
    ):
        if self.cfg.name == 'radam':
            from torch.optim import RAdam
            self.optimizer = RAdam(
                params=params,
                lr=self.lr,
                weight_decay=self.weight_decay   
            )
        elif self.cfg.name == 'adam':
            from torch.optim import Adam
            self.optimizer = Adam(
                params=params,
                lr=self.lr,
                weight_decay=self.weight_decay
            )

        return self.optimizer
