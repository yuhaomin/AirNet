
import torch.optim as optim

class Optim(object):

    def _makeOptimizer(self):

        if self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, params, method, lr):
        self.params = list(params)
        self.lr = lr
        self.method = method
        self._makeOptimizer()

    def step(self):
        self.optimizer.step()

