import torch
import numpy as np;
from torch.autograd import Variable
from train_code.train import *


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

class Data_utility(object):

    def __init__(self, X_train, XY_train, X_val, XY_val, X_predict, XY_predict, Y_train,Y_val, Y_predict):
        self.X_train = X_train
        self.X_val = X_val
        self.X_predict = X_predict
        self.XY_train = XY_train
        self.XY_predict = XY_predict
        self.XY_val = XY_val
        self.Y_train = Y_train
        self.Y_val = Y_val
        self.Y_predict = Y_predict
        self._split();

        
    def _split(self):
        self.train = self._batchify(self.X_train, self.XY_train, self.Y_train);
        self.valid = self._batchify(self.X_val, self.XY_val,  self.Y_val);
        self.test = self._batchify(self.X_predict, self.XY_predict,  self.Y_predict);

        
        
    def _batchify(self, x, xy,y):
        X = torch.FloatTensor(x)
        XY = torch.FloatTensor(xy)
        Y = torch.FloatTensor(y)

        return [X,XY,Y];

    def get_batches(self, inputs, inputs_y, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt];
            XY = inputs_y[excerpt];
            Y = targets[excerpt];
            X_in = X[:, -1:, :]
            X = X.cuda();
            XY = XY.cuda();
            Y = Y.cuda();
            X_in = X_in.cuda();
            yield Variable(X), Variable(XY), Variable(Y), Variable(X_in);
            start_idx += batch_size