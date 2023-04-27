import logging
import os

from omegaconf import DictConfig
from torch.optim import Optimizer

log = logging.getLogger(__name__)

class LitOptimizer(object):
    def __init__(
        self,
        cfg:DictConfig=None
    ):
        self.cfg = cfg.optimizer
        self.assister_cfg = cfg.optimizer_assister
    
    def get_optimizer(
        self,
        params
    )->Optimizer:
        if self.cfg.name == 'adadelta':
            from torch.optim import Adadelta
            self.optimizer = Adadelta(
                params=params,
                lr=self.cfg.lr,
                rho=self.cfg.rho,
                eps=self.cfg.eps,
                weight_decay=self.cfg.weight_decay,
                maximize=self.cfg.maximize,
                differentiable=self.cfg.differentiable
            )
        elif self.cfg.name == 'adagrad':
            from torch.optim import Adagrad
            self.optimizer = Adagrad(
                params=params,
                lr=self.cfg.lr,
                lr_decay=self.cfg.lr_decay,
                weight_decay=self.cfg.weight_decay,
                initial_accumulator_value=self.cfg.initial_accumulator_value,
                eps=self.cfg.eps,
                maximize=self.cfg.maximize,
                differentiable=self.cfg.differentiable
            )
        elif self.cfg.name == 'adam':
            from torch.optim import Adam
            self.optimizer = Adam(
                params=params,
                lr=self.cfg.lr,
                betas=(self.cfg.beta_1, self.cfg.beta_2),
                eps=self.cfg.eps,
                weight_decay=self.cfg.weight_decay,
                amsgrad=self.cfg.amsgrad,
                maximize=self.cfg.maximize,
                capturable=self.cfg.capturable,
                differentiable=self.cfg.differentiable,
            )
        elif self.cfg.name == 'adamw':
            from torch.optim import AdamW
            self.optimizer = AdamW(
                params=params,
                lr=self.cfg.lr,
                betas=(self.cfg.beta_1, self.cfg.beta_2),
                eps=self.cfg.eps,
                weight_decay=self.cfg.weight_decay,
                amsgrad=self.cfg.amsgrad,
                maximize=self.cfg.maximize,
                capturable=self.cfg.capturable,
                differentiable=self.cfg.differentiable,
            )
        elif self.cfg.name == 'sparseadam':
            from torch.optim import SparseAdam
            self.optimizer = SparseAdam(
                params=params,
                lr=self.cfg.lr,
                betas=(self.cfg.beta_1, self.cfg.beta_2),
                eps=self.cfg.eps,
                maximize=self.cfg.maximize,
            )
        elif self.cfg.name == 'adamax':
            from torch.optim import Adamax
            self.optimizer = Adamax(
                params=params,
                lr=self.cfg.lr,
                betas=(self.cfg.beta_1, self.cfg.beta_2),
                eps=self.cfg.eps,
                weight_decay=self.cfg.weight_decay,
                maximize=self.cfg.maximize,
                differentiable=self.cfg.differentiable,
            )
        elif self.cfg.name == 'asgd':
            from torch.optim import ASGD
            self.optimizer = ASGD(
                params=params,
                lr=self.cfg.lr,
                lambd=self.cfg.lambd,
                alpha=self.cfg.alpha,
                t0=self.cfg.t0,
                weight_decay=self.cfg.weight_decay,
                maximize=self.cfg.maximize,
                differentiable=self.cfg.differentiable,
            )
        elif self.cfg.name == 'lbfgs':
            from torch.optim import LBFGS
            self.optimizer = LBFGS(
                params=params,
                lr=self.cfg.lr,
                max_iter=self.cfg.max_iter,
                max_eval=self.cfg.max_eval,
                tolerance_grad=self.cfg.tolerance_grad,
                tolerance_change=self.cfg.tolerance_change,
                history_size=self.cfg.history_size,
                line_search_fn=self.cfg.line_search_fn,
            )
        elif self.cfg.name == 'nadam':
            from torch.optim import NAdam
            self.optimizer = NAdam(
                params=params,
                lr=self.cfg.lr,
                betas=(self.cfg.beta_1, self.cfg.beta_2),
                eps=self.cfg.eps,
                weight_decay=self.cfg.weight_decay,
                momentum_decay=self.cfg.momentum_decay,
                differentiable=self.cfg.differentiable,
            )
        elif self.cfg.name == 'radam':
            from torch.optim import RAdam
            self.optimizer = RAdam(
                params=params,
                lr=self.cfg.lr,
                betas=(self.cfg.beta_1, self.cfg.beta_2),
                eps=self.cfg.eps,
                weight_decay=self.cfg.weight_decay,
                differentiable=self.cfg.differentiable,
            )
        elif self.cfg.name == 'rmsprop':
            from torch.optim import RMSprop
            self.optimizer = RMSprop(
                params=params,
                lr=self.cfg.lr,
                alpha=self.cfg.alpha,
                eps=self.cfg.eps,
                weight_decay=self.cfg.weight_decay,
                momentum=self.cfg.momentum,
                centered=self.cfg.centered,
                maximize=self.cfg.maximize,
                differentiable=self.cfg.differentiable,
            )
        elif self.cfg.name == 'rprop':
            from torch.optim import Rprop
            self.optimizer = Rprop(
                params=params,
                lr=self.cfg.lr,
                etas=(self.cfg.eta_1, self.cfg.eta_2),
                step_sizes=(self.cfg.step_size_1, self.cfg.step_size_2),
                maximize=self.cfg.maximize,
                differentiable=self.cfg.differentiable,
            )
        elif self.cfg.name == 'sgd':
            from torch.optim import SGD
            self.optimizer = SGD(
                params=params,
                lr=self.cfg.lr,
                momentum=self.cfg.momentum,
                dampening=self.cfg.dampening,
                weight_decay=self.cfg.weight_decay,
                nesterov=self.cfg.nesterov,
                maximize=self.cfg.maximize,
                differentiable=self.cfg.differentiable,
            )
            
        elif self.cfg.name == 'adafactor':
            from .adafactor.adafactor import Adafactor
            self.optimizer = Adafactor(
                params=params,
                lr=self.cfg.lr,
                eps=(self.cfg.eps_1, self.cfg.eps_2),
                clip_threshold=self.cfg.clip_threshold,
                decay_rate=self.cfg.decay_rate,
                beta1=self.cfg.beta_1,
                weight_decay=self.cfg.weight_decay,
                scale_parameter=self.cfg.scale_parameter,
                relative_step=self.cfg.relative_step,
                warmup_init=self.cfg.warmup_init,
            )
        elif self.cfg.name == 'sm3':
            from .sm3.sm3 import SM3
            self.optimizer = SM3(
                params=params,
                lr=self.cfg.lr,
                momentum=self.cfg.momentum,
                beta=self.cfg.beta,
                eps=self.cfg.eps,
                weight_decay=self.cfg.weight_decay,
            )
        elif self.cfg.name == 'adabelief':
            try:
                from adabelief_pytorch import AdaBelief
            except:
                log.warning(
                    'AdaBelief is not installed. Try `pip install adabelief-pytorch`.'
                )
                os.system('pip install adabelief-pytorch==0.2.0')
                from adabelief_pytorch import AdaBelief
            self.optimizer = AdaBelief(
                params=params,
                lr=self.cfg.lr,
                betas=(self.cfg.beta_1, self.cfg.beta_2),
                eps=self.cfg.eps,
                weight_decay=self.cfg.weight_decay,
                amsgrad=self.cfg.amsgrad,
                weight_decouple=self.cfg.weight_decouple,
                fixed_decay=self.cfg.fixed_decay,
                rectify=self.cfg.rectify,
                degenerated_to_sgd=self.cfg.degenerated_to_sgd,
                print_change_log=self.cfg.print_change_log,
            )      
        elif self.cfg.name == 'shampoo':
            from .shampoo.shampoo import Shampoo
            self.optimizer = Shampoo(
                params=params,
                lr=self.cfg.lr,
                momentum=self.cfg.momentum,
                weight_decay=self.cfg.weight_decay,
                eps=self.cfg.eps,
                update_freq=self.cfg.update_freq,
            )                   
        else:
            raise ValueError('Unknown optimizer: {}'.format(self.cfg.name))

        if self.assister_cfg != None:
            if self.assister_cfg.name == 'sam':
                from .sam.sam import SAM
                self.optimizer = SAM(
                    params=params,
                    base_optimizer=self.optimizer,
                    rho=self.assister_cfg.rho,
                    adaptive=self.assister_cfg.adaptive,
                )
            elif self.assister_cfg.name == 'lookahead':
                from .lookahead.lookahead import Lookahead
                self.optimizer = Lookahead(
                    params=params,
                    k=self.assister_cfg.k,
                    alpha=self.assister_cfg.alpha,
                )
            else:
                raise ValueError('Unknown optimizer assister: {}'.format(self.assister_cfg.name))
        
        return self.optimizer
