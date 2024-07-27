from .fedbase import BasicServer
from .fedbase import BasicClient
import flgo.utils.fmodule as fmodule
import torch

class Server(BasicServer):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({"lmbd": 0.1})

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
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay,
                                                  momentum=self.momentum)
        for iter in range(self.num_steps):
            # get a batch of data
            batch_data = self.get_batch_data()
            model.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            loss = self.calculator.compute_loss(model, batch_data)['loss']
            loss.backward()
            if self.clip_grad > 0: torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                                                  max_norm=self.clip_grad)
            optimizer.step()
        return
