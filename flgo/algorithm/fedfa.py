"""
This is a non-official implementation of 'Fairness and Accuracy in Federated Learning' (http://arxiv.org/abs/2012.10069)
"""
from flgo.utils import fmodule
from flgo.algorithm.fedbase import BasicServer, BasicClient
import numpy as np

class Server(BasicServer):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'beta': 0.5, 'gamma': 0.9})
        self.m = self.model.zeros_like()
        self.alpha = 1.0 - self.beta
        self.eta = self.option['learning_rate']
        for c in self.clients: c.momentum = self.gamma

    def iterate(self):
        # sample clients
        self.selected_clients = self.sample()
        # training
        res = self.communicate(self.selected_clients)
        models, losses, ACC, F = res['model'], res['loss'], res['acc'], res['freq']
        # aggregate
        # calculate ACCi_inf, fi_inf
        sum_acc = np.sum(ACC)
        sum_f = np.sum(F)
        ACCinf = [-np.log2(1.0*acc/sum_acc+0.000001) for acc in ACC]
        Finf = [-np.log2(1-1.0*f/sum_f+0.00001) for f in F]
        sum_acc = np.sum(ACCinf)
        sum_f = np.sum(Finf)
        ACCinf = [acc/sum_acc for acc in ACCinf]
        Finf = [f/sum_f for f in Finf]
        # calculate weight = αACCi_inf+βfi_inf
        p = [self.alpha*accinf+self.beta*finf for accinf,finf in zip(ACCinf,Finf)]
        wnew = self.aggregate(models, p)
        dw = wnew -self.model
        # calculate m = γm+(1-γ)dw
        self.m = self.gamma*self.m + (1 - self.gamma)*dw
        self.model = wnew - self.m * self.eta
        return

    def aggregate(self, models, p):
        return fmodule._model_average(models, p)

    def save_checkpoint(self):
        cpt = super().save_checkpoint()
        cpt.update({
            'fs': [ci.frequency for ci in self.clients],
        })
        return cpt

    def load_checkpoint(self, cpt):
        super().load_checkpoint(cpt)
        fs = cpt.get('fs', [0 for _ in self.clients])
        for client_i, fi in zip(self.clients, fs):
            client_i.frequency = fi

class Client(BasicClient):
    def initialize(self, *args, **kwargs):
        self.frequency = 0

    def reply(self, svr_pkg):
        model = self.unpack(svr_pkg)
        metrics = self.test(model,'train')
        acc, loss = metrics['accuracy'], metrics['loss']
        self.train(model)
        cpkg = self.pack(model, loss, acc)
        return cpkg

    def pack(self, model, loss, acc):
        self.frequency += 1
        return {
            "model":model,
            "loss":loss,
            "acc":acc,
            "freq":self.frequency,
        }