import copy
from .fedbase import BasicServer
from .fedbase import BasicClient
import flgo.utils.fmodule as fmodule
import torch

class Server(BasicServer):
    def initialize(self, *args, **kwargs):
        """
        lmbd: 0.001, 0.01, 0.1, 0.5
        alpha: 0.0, 0.5, 0.75, 0.875, 0.9375, 1.0
        rho: 0.1
        """
        self.init_algo_para({"lmbd": 0.1, 'alpha':0.9375, 'rho':0.1})
        self.aggregation_option = 'uniform'
class Client(BasicClient):
    def initialize(self, *args, **kwargs):
        self.local_grad_controller = self.server.model.zeros_like().to('cpu')

    @fmodule.with_multi_gpus
    def train(self, model):
        r"""
        Standard local_movielens_recommendation training procedure. Train the transmitted model with
        local_movielens_recommendation training dataset.

        Args:
            model (FModule): the global model
        """
        model.train()
        model_tmp = copy.deepcopy(model)
        global_model = copy.deepcopy(model)
        global_model.freeze_grad()
        self.local_grad_controller.to(self.device)
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.num_steps):
            # get a batch of data
            optimizer.zero_grad()
            batch_data = self.get_batch_data()
            loss1 = self.calculator.compute_loss(model_tmp, batch_data)['loss']
            loss1.backward()
            grad1 = [p.grad for p in model_tmp.parameters()]
            # grad1_norm = torch.norm(torch.cat([g.view(-1) for g in grad1 if g is not None]), 2)
            with torch.no_grad():
                for p in model_tmp.parameters():
                    if p.grad is not None: p.data = p.data + self.rho*p.grad
            model_tmp.zero_grad()
            loss2 = self.calculator.compute_loss(model_tmp, batch_data)['loss']
            loss2.backward()
            grad2 = [p.grad for p in model_tmp.parameters()]
            quasi_grad = [(1-self.alpha)*g1+self.alpha*g2 if g1 is not None else None for g1, g2 in zip(grad1, grad2)]
            with torch.no_grad():
                for p,qg,lg,gm in zip(model.parameters(), quasi_grad, self.local_grad_controller.parameters(), global_model.parameters()):
                    p.grad = qg - lg + 1.0/self.lmbd*(p - gm)
            if self.clip_grad > 0: torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.clip_grad)
            optimizer.step()
            model_tmp.load_state_dict(model.state_dict())
        self.local_grad_controller = self.local_grad_controller - 1.0/self.lmbd * (model - global_model)
        with torch.no_grad():
            for pm, lg in zip(model.parameters(), self.local_grad_controller.parameters()):
                pm.data = pm.data - self.lmbd*lg
        self.local_grad_controller.to('cpu')
        return
