import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
class LitModel(pl.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        model:torch.nn.Module,
        optimizer,
    ):
        pass