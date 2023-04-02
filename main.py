# %%
import hydra
import logging
import sys
from utils.utils import *
from omegaconf import DictConfig

# %%
@hydra.main(
    version_base="1.2.0",
    config_path="config",
    config_name="config"
)
def main(
    cfg: DictConfig
)->None:
    # init
    init_configs = init(cfg)
    lit_model = get_lightning_model(cfg)    # 모델을 만들어야 함.

    # get dataset
    dataset = get_dataset(cfg)  # 미완성

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[cfg.devices] if type(cfg.devices) != list else cfg.devices,
        logger=init_configs['logger'],
        max_epochs=cfg.max_epochs
        # callbacks=[]
    )

    background_color_log('begin training')
    trainer.fit(lit_model)


# %%
if __name__ == "__main__":
    main()