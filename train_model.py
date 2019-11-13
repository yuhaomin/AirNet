# encoding=gbk
import os
from train_code.common import *
from train_code.train import train
import ConfigParser
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


if __name__ == '__main__':
    wx_path = "data/example.csv"#location of dataset( Note that we have no right to directly publish the datasets in our paper.)
    wx_gk = pd.read_csv(wx_path)
    wx_title = wx_path.split('/')[-1].split('.')[0]
    model_path = 'res/model/ssname'
    check_or_create_path(model_path)
    train(wx_gk, wx_title, model_path)