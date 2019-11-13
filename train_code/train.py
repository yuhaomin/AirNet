
# encoding=gbk
import argparse
from common import *
from sklearn import preprocessing



import string
import re
import random
import time
import math
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import random
import matplotlib.pyplot as plt

from utils import *
from utils import Data_utility
import Optim
from GlobalAttention import *

class EncoderRNN(nn.Module):
    def __init__(self, args, input_size):
        super(EncoderRNN, self).__init__()
        self.use_cuda = args.cuda

        self.m = input_size
        self.hidR = args.hidRNN;
        self.hidC = args.hidCNN;
        self.hidRx = args.hidRNNx;
        self.P = args.window;
        self.Ck = args.CNN_kernel;
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m), padding=(self.Ck/2, 0));
        self.GRU1 = nn.GRU(self.hidC, self.hidR, num_layers=1);
        self.GRUx = nn.GRU(1, self.hidRx, num_layers=1);
        self.dropout = nn.Dropout(p=args.dropout);



    def forward(self,ii, x):

        c = x.view(-1, 1, self.P, self.m);
        c = F.relu(self.conv1(c));
        c = self.dropout(c);
        c = torch.squeeze(c, 3);

        r = c.permute(2, 0, 1).contiguous();
        eo, r = self.GRU1(r);
        r = self.dropout(r);


        x_new = x[:,:,ii:ii+1]
        eox, rx = self.GRUx(x_new.permute(1, 0, 2))
        rx = self.dropout(rx)

        return  r, rx, eo, eox;



class EncoderyRNN(nn.Module):
    def __init__(self, args):
        super(EncoderyRNN, self).__init__()
        self.use_cuda = args.cuda
        self.hidRy = args.hidRNNy;
        self.GRU1 = nn.GRU(1, self.hidRy,num_layers=1);
        self.dropout = nn.Dropout(p=args.dropout);

    def forward(self,ii,xy):
        r = xy.permute(1, 0, 2).contiguous();

        output, r = self.GRU1(r);
        r = self.dropout(r);
        return output, r;



class DecoderRNN(nn.Module):
    def __init__(self, args, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.inp = 1
        self.hidden_size = hidden_size
        self.hidRy = args.hidRNNy;
        self.dim = args.dim
        self.attn = GlobalAttention(args, self.dim )
        self.gru1 = nn.GRU(self.inp, self.dim , num_layers=1)
        self.out = nn.Linear(self.dim , output_size)


    def forward(self, ii, input, hidden, h1,h2,e1output):
        attn_h2, align_vectors2 = self.attn(h2, h1,e1output,self.dim)
        input = input.permute(1,0, 2)
        hidden_attn = attn_h2.permute(1,0,2)
        output, hidden = self.gru1(input, hidden_attn)
        output = self.out(hidden[-1:,:,:])

        return output

    def initHidden(self):
        return torch.zeros(1, 128, self.hidden_size)


def draw_pic(ori_, pre_, path_pic):

    for i in range(1):
        fig = plt.figure()
        fig.set_figheight(30)
        fig.set_figwidth(100)
        ax = fig.add_subplot(111)
        plt.xticks(fontsize=45)
        plt.yticks(fontsize=60)
        ori = ori_
        pre = pre_
        x = range(len(ori))
        Oplot, = plt.plot(x, ori, 'r', linewidth=6, label='res' + str(i))

        Pplot, = plt.plot(x, pre, 'g', linewidth=6, label='gt' + str(i))
        font1 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 205,
                 }
        # plt.title(title + u'//tendency')
        ax.legend(handles=[Pplot, Oplot], labels=['gt' + str(i), 'res' + str(i)], prop=font1)
        plt.xlabel(u'time')
        plt.ylabel(u'mearurements', fontsize=60)
        plt.legend()
        plt.savefig(path_pic)
    plt.close('all')




def evaluate(ii, data, X,XY, Y, encoder,encoder2, decoder, evaluateL2, evaluateL1, batch_size, st_con, path_pic):
    encoder.eval()
    encoder2.eval()
    decoder.eval()
    total_loss = 0;
    total_loss_l1 = 0;
    n_samples = 0;
    predict = None;
    test = None;

    gt_res = [];
    pre_res =[];
    with torch.no_grad():
        for X,XY, Y, X_in in data.get_batches(X, XY, Y, batch_size, False):

            encoder_hidden, encoder_hiddenx,eo,exo = encoder(ii, X)
            encoder2_evryinput, encoder_hidden2= encoder2(ii,XY)
            encoder_evryinput = torch.cat([eo, exo], 2)
            decoder_hidden_init = torch.cat([encoder_hidden, encoder_hiddenx], 2)
            decoder_input = X_in[:,:,ii:ii+1]

            decoder_hidden = torch.cat([encoder_hidden, encoder_hiddenx ,encoder_hidden2],2)
            output = decoder( ii, decoder_input, decoder_hidden, decoder_hidden_init, encoder_hidden2, encoder_evryinput)
            output = output[0,:,:]
            pre_tmp = output.transpose(1,0).data.tolist()[0]
            Y_tmp = Y.transpose(1, 0).data.tolist()[0]
            pre_res.extend(pre_tmp)
            gt_res.extend(Y_tmp)
            if predict is None:
                predict = output;
                test = Y;
            else:
                predict = torch.cat((predict, output));
                test = torch.cat((test, Y));

            total_loss += evaluateL2(output , Y ).data[0]
            total_loss_l1 += evaluateL1(output , Y ).data[0]
            n_samples += (output.size(0) );

    smape, rae= compute_cost1(gt_res, pre_res)
    if st_con == 'test':
        draw_pic(pre_res, gt_res, path_pic)
    return  rae, smape, pre_res


def process_feature(wx_gk, train_window, train_val, val_test):
    gk_columns = wx_gk.columns[-6:].tolist()
    gk_columns.append('RECEIVETIME')
    wx_df = wx_gk[wx_gk.columns[:-6]]
    gk_df = wx_gk[gk_columns]
    gk_df['RECEIVETIME'] = format_time(gk_df['RECEIVETIME'])
    wx_df['RECEIVETIME'] = format_time(wx_df['RECEIVETIME'])

    wx_df['ZUFEN'] = wx_df['PM25_x'] / wx_df['PM10_x']
    wx_df['PM_SUM'] = wx_df['PM10N'] + wx_df['PM25N'] + wx_df['PM1N'] + wx_df['PM05N']
    wx_df['PM10N_ratio'] = wx_df['PM10N'] / wx_df['PM_SUM']
    wx_df['PM25N_ratio'] = wx_df['PM25N'] / wx_df['PM_SUM']



    all_df = pd.merge(wx_df, gk_df, on='RECEIVETIME')

    all_df_train = all_df.iloc[: train_val]
    all_df_val = all_df.iloc[train_val - train_window + 1:val_test]
    all_df_predict = all_df.iloc[val_test-train_window+1:]

    all_df_train = all_df_train.drop('RECEIVETIME', 1)
    all_df_val = all_df_val.drop('RECEIVETIME', 1)
    all_df_predict = all_df_predict.drop('RECEIVETIME', 1)



    X_train = all_df_train[all_df_train.columns[:-6]]
    y_train = all_df_train[all_df_train.columns[-6:]]

    X_val = all_df_val[all_df_val.columns[:-6]]
    y_val = all_df_val[all_df_val.columns[-6:]]

    X_predict = all_df_predict[all_df_predict.columns[:-6]]
    y_predict = all_df_predict[all_df_predict.columns[-6:]]


    return X_train, y_train, X_val, y_val, X_predict, y_predict



def seq_split(X,y,train_window, train_y_window):

    sample_x = []
    sample_y = []
    sample_y_en = []
    X_LEN = len(X) - train_window + 1
    START = 0
    END = train_window
    START_y = END - train_y_window-1
    for ii in range(X_LEN):
            sample_x.append(np.array(X.iloc[START:END]).T)
            sample_y_en.append(np.array(y.iloc[START_y:END-1].T))
            sample_y.append(np.array(y.iloc[END - 1:END]).T)
            START += 1
            START_y += 1
            END += 1
    source_int = np.array(sample_x).transpose((2, 0, 1))

    sourcey_int =np.expand_dims(np.array(sample_y_en),axis = 1).transpose((2, 0, 1))
    target_int =np.expand_dims(np.array(sample_y),axis = 1).transpose((2, 0, 1))

    return source_int, sourcey_int, target_int

def compute_cost1(real, predict):
    re=real
    pr = predict
    abs_ans=[]
    smape_ans = []
    for i in range(len(re)):
        smape_ans.append(2*abs(re[i] - pr[i]) / (abs(re[i]) + abs(pr[i])))

        abs_ans.append(abs(re[i] - pr[i]))



    return np.mean(smape_ans),np.mean(abs_ans)

def trainIters(ii, data, X, XY, Y,encoder,encoder1, decoder, batch_size,criterion, optim1,optim1_2, optim2):
    encoder.train()
    encoder1.train()
    decoder.train()
    n_samples = 0;
    print_loss_total = 0
    plot_loss_total = 0

    for X, XY, Y, X_in in data.get_batches(X, XY, Y, batch_size, True):
        encoder.zero_grad()
        encoder1.zero_grad()
        decoder.zero_grad()

        encoder_hidden, encoder_hidden_x,eo,exo = encoder(ii, X)
        encoder2_evryinput, encoder_hidden2  = encoder1(ii,XY)

        encoder_evryinput = torch.cat([eo,exo],2)
        decoder_input = X_in[:,:,ii:ii+1]

        decoder_hidden_init = torch.cat([encoder_hidden, encoder_hidden_x],2)
        decoder_hidden = torch.cat([encoder_hidden, encoder_hidden_x, encoder_hidden2],2)

        decoder_output = decoder(ii,
            decoder_input, decoder_hidden, decoder_hidden_init, encoder_hidden2,encoder_evryinput )

        loss = criterion(decoder_output[0,:,:], Y)
        loss.backward()

        optim1.step();
        optim1_2.step();
        optim2.step();


        print_loss_total += loss.item()
        plot_loss_total += loss.item()
        n_samples += decoder_output[0,:,:].size(0)

    return  print_loss_total/ n_samples

def train(wx_gk, wx_title, file_pre=''):
        #window length of mobile station
        win_ms = 168
        #window length of static station
        win_ss = 24
        train_window = win_ms
        train_y_window = win_ss

        wx_model_root_path = os.path.join(file_pre, wx_title)
        check_or_create_path(wx_model_root_path)



        train_val = int(len(wx_gk) * 0.8)
        val_test = int(len(wx_gk) * 0.9)

        X_train, y_train, X_val, y_val, X_predict, y_predict = process_feature(wx_gk, train_window, train_val, val_test)

        title = ['CO_x', 'NO2_x', 'SO2_x', 'O3_x', 'PM25_x', 'PM10_x', 'TEMPERATURE', 'HUMIDITY', 'PM05N', 'PM1N',
                 'PM25N', 'PM10N', 'PM_SUM', 'ZUFEN',
                 'PM10N_ratio', 'PM25N_ratio']

        scaler = preprocessing.MinMaxScaler().fit(X_train[title])

        X_train[title] = scaler.transform(X_train[title])

        X_val[title] = scaler.transform(X_val[title])
        X_predict[title] = scaler.transform(X_predict[title])

        titles = ['O3']  # or ,'CO'
        ii = 0
        f_log_all = open(os.path.join(wx_model_root_path, 'train_log_loss.txt'), 'a')
        predict_value_path = wx_model_root_path + '/' + 'value.csv'
        smape_loss_list_best = []
        mae_loss_list_best = []
        y_predict_value = []

        train_window = train_window
        parser = argparse.ArgumentParser(description='PyTorch Sensor calibration')
        parser.add_argument('--model', type=str, default='AirNet',
                            help='')
        parser.add_argument('--hidCNN', type=int, default=100,
                            help='number of CNN hidden units')
        parser.add_argument('--CNN_kernel', type=int, default=5,
                            help='the kernel size of the CNN layers')

        parser.add_argument('--hidRNN', type=int, default=80,
                            help='number of RNN hidden units (cross)')
        parser.add_argument('--hidRNNy', type=int, default=50,
                            help='number of RNN hidden units (global)')
        parser.add_argument('--hidRNNx', type=int, default=50,
                            help='number of RNN hidden units (basic)')
        parser.add_argument('--window', type=int, default=train_window,
                            help='window size')
        parser.add_argument('--epochs', type=int, default=100,
                            help='upper epoch limit')
        parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                            help='batch size')
        parser.add_argument('--dim', type=int, default=120,
                            help='fuse_dim')
        parser.add_argument('--dropout', type=float, default=0.2,
                            help='dropout applied to layers (0 = no dropout)')
        parser.add_argument('--cuda', type=str, default=True)
        parser.add_argument('--optim', type=str, default='adam')
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--L1Loss', type=bool, default=True)

        args = parser.parse_args()

        X_train_copy = copy.deepcopy(X_train)
        y_train_copy = copy.deepcopy(y_train[titles[ii] + '_y'])
        X_val_copy = copy.deepcopy(X_val)
        y_val_copy = copy.deepcopy(y_val[titles[ii] + '_y'])
        X_predict_copy = copy.deepcopy(X_predict)
        y_predict_copy = copy.deepcopy(y_predict[titles[ii] + '_y'])

        source_int_train, sourcey_int_train, targets_int_t = seq_split(X_train_copy, y_train_copy, train_window,
                                                                       train_y_window)

        source_int_val, sourcey_int_val, targets_int_v = seq_split(X_val_copy, y_val_copy, train_window,
                                                                   train_y_window)

        source_int_predict, sourcey_int_predict, targets_int_p = seq_split(X_predict_copy, y_predict_copy,
                                                                           train_window, train_y_window)

        targets_int_train = targets_int_t[0, :, :]
        targets_int_val = targets_int_v[0, :, :]
        targets_int_predict = targets_int_p[0, :, :]

        train_source = source_int_train
        train_sourcey = sourcey_int_train
        train_target = targets_int_train

        train_source_shuffle = []
        train_sourcey_shuffle = []

        train_target_shuffle = []
        num_shuffle = list(range(train_source.shape[1]))
        random.shuffle(num_shuffle)
        for i in num_shuffle:
            train_source_shuffle.append(train_source[:, i, :])
            train_sourcey_shuffle.append(train_sourcey[:, i, :])
            train_target_shuffle.append(train_target[i, :])

        train_source_shuffle = np.array(train_source_shuffle).transpose(1, 0, 2)
        train_sourcey_shuffle = np.array(train_sourcey_shuffle).transpose(1, 0, 2)

        Data = Data_utility(train_source_shuffle.transpose(1, 0, 2),
                            train_sourcey_shuffle.transpose(1, 0, 2),
                            source_int_val.transpose(1, 0, 2),
                            sourcey_int_val.transpose(1, 0, 2),
                            source_int_predict.transpose(1, 0, 2),
                            sourcey_int_predict.transpose(1, 0, 2),
                            train_target_shuffle,
                            targets_int_val, targets_int_predict);

        title = titles[ii]

        checkpoint_best_encoder = os.path.join(wx_model_root_path,
                                               title + 'trained_model_encoder_best.pt')
        checkpoint_best_encoder2 = os.path.join(wx_model_root_path,
                                                title + 'trained_model_encoder2_best.pt')
        checkpoint_best_decoder = os.path.join(wx_model_root_path,
                                               title + 'trained_model_decoder_best.pt')

        encoder1 = EncoderRNN(args, Data.train[0].shape[2]).cuda()
        encoder2 = EncoderyRNN(args).cuda()
        decoder1 = DecoderRNN(args, args.hidRNN, 1).cuda()
        optim1 = Optim.Optim(
            encoder1.parameters(), args.optim, args.lr
        )
        optim1_2 = Optim.Optim(
            encoder2.parameters(), args.optim, args.lr
        )
        optim2 = Optim.Optim(
            decoder1.parameters(), args.optim, args.lr
        )

        if args.L1Loss:
            criterion = nn.L1Loss(size_average=False);
        else:
            criterion = nn.MSELoss(size_average=False);
        evaluateL2 = nn.MSELoss(size_average=False);
        evaluateL1 = nn.L1Loss(size_average=False)

        best_val = 10000000;

        check_or_create_path(wx_model_root_path + '/' + 'pic')
        pic_root = wx_model_root_path + '/' + 'pic' + '/'
        print(pic_root)
        path_pic = pic_root + titles[ii] + '_pic_best' + '.png'
        try:
            print('begin training');
            plot_loss_total = 0

            for epoch in range(0, args.epochs):
                epoch_start_time = time.time()

                train_loss = trainIters(ii, Data, Data.train[0], Data.train[1], Data.train[2], encoder1, encoder2,
                                        decoder1, args.batch_size, criterion, optim1, optim1_2, optim2)
                val_rae, sm, out = evaluate(ii, Data, Data.valid[0], Data.valid[1], Data.valid[2], encoder1,
                                            encoder2, decoder1, evaluateL2,
                                            evaluateL1, args.batch_size, 'var', path_pic);

                plot_loss_total += train_loss
                print(
                    '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid mae {:5.4f} | valid smape {:5.4f} '.format(
                        epoch, (time.time() - epoch_start_time), train_loss, val_rae, sm))

                if val_rae < best_val:
                    with open(checkpoint_best_encoder, 'wb') as f:
                        torch.save(encoder1, f)
                    with open(checkpoint_best_encoder2, 'wb') as f:
                        torch.save(encoder2, f)
                    with open(checkpoint_best_decoder, 'wb') as f1:
                        torch.save(decoder1, f1)
                    best_val = val_rae



        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

        # Load the best saved model.
        with open(checkpoint_best_encoder, 'rb') as f:
            encoder1 = torch.load(f)
        with open(checkpoint_best_encoder2, 'rb') as f2:
            encoder2 = torch.load(f2)
        with open(checkpoint_best_decoder, 'rb') as f1:
            decoder1 = torch.load(f1)

        test_rae, smape, predict = evaluate(ii, Data, Data.test[0], Data.test[1], Data.test[2], encoder1, encoder2,
                                            decoder1, evaluateL2, evaluateL1,
                                            args.batch_size, 'test', path_pic);

        smape_loss_list_best.append(str(round(smape, 4)))
        mae_loss_list_best.append(str(round(test_rae, 3)))

        y_predict_value.append(predict)

        pd.DataFrame(np.array(y_predict_value).transpose((1, 0)), columns=titles).to_csv(predict_value_path)


        print >> f_log_all, '\t'.join(titles)

        print >> f_log_all, '\t'.join(smape_loss_list_best) + '\t' + 'smape_loss_best'

        print >> f_log_all, '\t'.join(mae_loss_list_best) + '\t' + 'mae_loss_best'

        f_log_all.close()




