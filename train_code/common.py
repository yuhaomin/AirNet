#coding:utf-8
import os
import pandas as pd
from datetime import datetime
import copy
import numpy as np
import cPickle as pickle
import math
from sklearn.cross_validation import train_test_split


def check_or_create_path(path):
    if not os.path.exists(path) or not os.path.isdir(path):
        os.mkdir(path)
    return True


def format_time(df_col):
    df_col = df_col.map(
        lambda x: str(x).replace('/', '-') if ':' in str(x) else str(x).replace('/', '-') + ' 0:00')
    df_col = df_col.map(lambda x: ':'.join(x.split(':')[:1]))
    df_col = df_col.map(lambda x: datetime.strptime(str(x), "%Y-%m-%d %H"))
    return df_col


