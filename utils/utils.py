import sys
import logging
import pytorch_lightning as pl
import torch
import numpy
import random
import os
from .optimizer.optimizer import LitOptimizer
from omegaconf import DictConfig
from .bcolors import bcolors
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)
now = datetime.now()
now = now.strftime("%Y%m%d_%H%M%S")

def log_command_line():
    EXE_FILE = sys.argv[0]
    ARGS = ' '.join(sys.argv[1:])

    log.info(
        bcolors.HEADER +
        bcolors.OKGREEN + 
        'Execution file: ' + 
        bcolors.ENDC +
        bcolors.OKCYAN +
        EXE_FILE +
        bcolors.ENDC +        
        bcolors.ENDC
    )
    log.info(
        bcolors.HEADER +
        bcolors.OKGREEN + 
        'Arguments: ' + 
        bcolors.ENDC +
        bcolors.OKCYAN +
        ARGS +
        bcolors.ENDC +        
        bcolors.ENDC
    )

def log_optimizer_config(
    cfg: DictConfig
):
    log.info(
        bcolors.HEADER +
        bcolors.OKGREEN + 
        'Optimizer configs: ' + 
        bcolors.ENDC +
        bcolors.ENDC
    )
    log.info(
        bcolors.HEADER +
        bcolors.OKGREEN + 
        '\t\tname: ' + 
        bcolors.ENDC +
        bcolors.OKCYAN +
        str(cfg.optimizer.name) + 
        bcolors.ENDC +        
        bcolors.ENDC
    )
    log.info(
        bcolors.HEADER +
        bcolors.OKGREEN + 
        '\t\tlr: ' + 
        bcolors.ENDC +
        bcolors.OKCYAN +
        str(cfg.optimizer.lr) + 
        bcolors.ENDC +
        bcolors.ENDC
    )
    log.info(
        bcolors.HEADER +
        bcolors.OKGREEN + 
        '\t\tweight_decay: ' + 
        bcolors.ENDC +
        bcolors.OKCYAN +
        str(cfg.optimizer.weight_decay) + 
        bcolors.ENDC +
        bcolors.ENDC
    )
    if cfg.optimizer.name == 'adafactor':
        if cfg.optimizer.relative_step:
            log.info(
                bcolors.HEADER +
                bcolors.OKGREEN + 
                '\t\tmode: ' + 
                bcolors.ENDC +
                bcolors.OKCYAN +
                'relative step mode' +
                bcolors.ENDC +
                bcolors.ENDC
            )
        else:
            log.info(
                bcolors.HEADER +
                bcolors.OKGREEN + 
                '\t\tmode: ' + 
                bcolors.ENDC +
                bcolors.OKCYAN +
                'learning rate mode' +
                bcolors.ENDC +
                bcolors.ENDC
            )

def log_checkpoint(
    cfg: DictConfig
):
    checkpoint = get_checkpoint(cfg)

    log.info(
        bcolors.HEADER +
        bcolors.OKGREEN + 
        'Checkpoint: ' + 
        bcolors.ENDC +
        bcolors.OKCYAN +
        str(checkpoint) + 
        bcolors.ENDC +
        bcolors.ENDC
    )

def init_precision(
    cfg: DictConfig
):
    torch.set_float32_matmul_precision(cfg.set_float32_matmul_precision)

def init_random_seed(
    cfg: DictConfig
):
    random_seed = cfg.seed
    log.info(
        bcolors.HEADER +
        bcolors.OKGREEN + 
        'Initialize random seed: ' + 
        bcolors.ENDC +
        bcolors.OKCYAN +
        str(random_seed) + 
        bcolors.ENDC +
        bcolors.ENDC
    )
    if type(random_seed) == int:
        pl.seed_everything(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed) # if use multi-GPU

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        numpy.random.seed(random_seed)
        random.seed(random_seed)

def init_wandb(
    cfg: DictConfig
)->WandbLogger:
    
    
    if cfg.logger.version == None:
        version = str(now)
    else:
        version = cfg.logger.version
        
    log.info(
        bcolors.HEADER +
        bcolors.OKGREEN + 
        'Wandb configs: ' + 
        bcolors.ENDC +
        bcolors.ENDC
    )
    log.info(
        bcolors.HEADER +
        bcolors.OKGREEN + 
        '\t\tproject: ' + 
        bcolors.ENDC +
        bcolors.OKCYAN +
        str(cfg.logger.project) + 
        bcolors.ENDC +        
        bcolors.ENDC
    )
    log.info(
        bcolors.HEADER +
        bcolors.OKGREEN + 
        '\t\tname: ' + 
        bcolors.ENDC +
        bcolors.OKCYAN +
        str(cfg.logger.name) + 
        bcolors.ENDC +        
        bcolors.ENDC
    )
    log.info(
        bcolors.HEADER +
        bcolors.OKGREEN + 
        '\t\toffline mode: ' + 
        bcolors.ENDC +
        bcolors.OKCYAN +
        str(cfg.logger.offline) + 
        bcolors.ENDC +
        bcolors.ENDC
    )
    log.info(
        bcolors.HEADER +
        bcolors.OKGREEN + 
        '\t\tversion: ' + 
        bcolors.ENDC +
        bcolors.OKCYAN +
        str(version) + 
        bcolors.ENDC +
        bcolors.ENDC
    )
    
    return WandbLogger(
        name=cfg.logger.name,
        save_dir=os.path.join(DIR_PATH, cfg.logger.save_dir),
        project=cfg.logger.project,
        offline=cfg.logger.offline,
        checkpoint_name=get_checkpoint(cfg),
        version=version
    )

def get_checkpoint(
    cfg: DictConfig
)->str:
    checkpoint = os.path.join(cfg.checkpoint_path, now)

    return checkpoint

def get_lightning_model(
    cfg: DictConfig,
)->pl.LightningModule:
    r"""
        If you want to make new model, make if-else like:
        elif cfg.model == 'v2':
            from .models.lightning_models.v2 import LitModel
            from .models.raw_models.v2 import Model
    """
    if cfg.model == 'v1':
        from .models.lightning_models.v1 import LitModel
        from .models.raw_models.v1 import Model
    else:
        raise ValueError("Select model does not exist.")

    log.info(
        bcolors.HEADER +
        bcolors.OKGREEN + 
        'Model: ' + 
        bcolors.ENDC +
        bcolors.OKCYAN +
        f'{cfg.model}' + 
        f'(compile{cfg.compile_model})' + 
        bcolors.ENDC +
        bcolors.ENDC
    )
    
    if cfg.compile_model:
        return LitModel(
            cfg,
            model=torch.compile(Model(cfg)),
            optimizer=LitOptimizer(cfg)
        )
    return LitModel(
        cfg,
        model=Model(cfg),
        optimizer=LitOptimizer(cfg)
    )

def get_dataset(
    cfg: DictConfig
):
    background_color_log('get dataset')
    batch_size = cfg.dataset.batch_size

def background_color_log(msg, init=False):
    if init:
        log.info(
            bcolors.HEADER +
            bcolors.BACK_BLUE + 
            30 * '#' + f' {msg} ' + '#' * 30 +
            bcolors.ENDC +
            bcolors.ENDC
        )
        return

    log.info(
        bcolors.HEADER +
        bcolors.BACK_GREEN + 
        30 * '#' + f' {msg} ' + '#' * 30 +
        bcolors.ENDC +
        bcolors.ENDC
    )

def delayed_mode(
    cfg: DictConfig
):
    assert type(cfg.sleep) == int
    sleep_time = int(cfg.sleep)
    if sleep_time != 0:        
        log.info(
            bcolors.HEADER +
            bcolors.OKGREEN + 
            'Delayed execution: ' + 
            bcolors.ENDC +
            bcolors.OKCYAN +
            f'{sleep_time}(s)' + 
            bcolors.ENDC +
            bcolors.ENDC
        )
        from time import sleep
        sleep(sleep_time)

def init(
    cfg: DictConfig
):
    background_color_log('init', True)
    log_command_line()
    log_optimizer_config(cfg)
    log_checkpoint(cfg)
    init_random_seed(cfg)
    init_precision(cfg)
    logger = init_wandb(cfg)
    delayed_mode(cfg)

    if cfg.compile_model:
        raise ValueError("Compile model is not supported yet.")

    return {
        'logger': logger,
    }