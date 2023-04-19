from omegaconf import DictConfig
from torch.optim import Optimizer

class LitOptimizer(object):
    def __init__(
        self,
        cfg:DictConfig=None
    ):
        self.cfg = cfg.optimizer
    
    def get_optimizer(
        self,
        params
    ):
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
        else:
            raise ValueError('Unknown optimizer: {}'.format(self.cfg.name))

        return self.optimizer
