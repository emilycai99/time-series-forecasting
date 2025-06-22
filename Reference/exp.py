import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import copy
import warnings
warnings.filterwarnings('ignore')
torch.autograd.set_detect_anomaly(True)

import sys 
sys.path.append('/home/r11user3/Yuxi/ParaRNN/ETT')
from exp_basic import Exp_Basic
from ptb_style.data.data_loader import Dataset_ETT_hour, StepSampler, Dataset_ETT_minute, Dataset_Custom
from utils.tools import adjust_learning_rate_no_change
from utils.metrics import metric
from ptb_style.model.rnn_model import vanilla_RNN, ParaRNN_torch
from ptb_style.model.pararnn_model import ParaRNN_c_forecast, ParaRNN_c_forecast_no_ffn
from ptb_style.model.lstm_model import PytorchLSTM, Script_ParaLSTM, ParaLSTM
from ptb_style.model.gru_model import PytorchGRU, Script_ParaGRU, ParaGRU

class Exp_RNN(Exp_Basic):
    def __init__(self, args):
        super(Exp_RNN, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            ############## rnn ###########################
            'pytorch_rnn': vanilla_RNN,
            'pararnn': ParaRNN_c_forecast,
            'pararnn_no_ffn': ParaRNN_c_forecast_no_ffn,
            'pararnn_torch': ParaRNN_torch,
            'pararnn_torch_no_ffn': ParaRNN_torch,
            ############## lstm ###########################
            'pytorch_lstm': PytorchLSTM,
            'paralstm': Script_ParaLSTM,
            'paralstm_no_ffn': Script_ParaLSTM,
            'paralstm_torch_no_ffn': ParaLSTM,
            'paralstm_torch': ParaLSTM,
            ############## gru ###########################
            'pytorch_gru': PytorchGRU,
            'paragru': Script_ParaGRU,
            'paragru_no_ffn': Script_ParaGRU,
            'paragru_torch': ParaGRU,
            'paragru_torch_no_ffn': ParaGRU
        }
        if self.args.model in ['pytorch_rnn', 'pytorch_lstm', 'pytorch_gru']:
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.embed_size,
                self.args.hidden_size,
                self.args.c_out,
                self.args.num_layers,
                self.args.update_nonlinearity,
                self.args.pos_enc,
                self.args.temp_enc,
                self.args.embed,
                self.args.freq,
                self.args.dropout_embed,
                self.args.dropout_rnn
            ).float()
        elif self.args.model in ['pararnn', 'pararnn_no_ffn']:
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.embed_size,
                self.args.hidden_size,
                self.args.c_out,
                self.args.num_layers,
                self.args.update_nonlinearity,
                self.args.pos_enc,
                self.args.temp_enc,
                self.args.embed,
                self.args.freq,
                self.args.dropout_embed,
                self.args.dropout_rnn,
                self.args.dropout,
                self.args.d_s,
                self.args.K,
                self.args.mlp_size,
                self.args.rnn_layernorm,
                torch.device('cuda:{}'.format(torch.cuda.current_device()))
            ).float()
        elif 'paralstm' in self.args.model or 'paragru' in self.args.model or 'pararnn_torch' in self.args.model:
            if 'no_ffn' in self.args.model:
                ffn_flag = False
            else:
                ffn_flag = True
            
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.embed_size,
                self.args.hidden_size,
                self.args.c_out,
                self.args.num_layers,
                self.args.update_nonlinearity,
                self.args.pos_enc,
                self.args.temp_enc,
                self.args.embed,
                self.args.freq,
                self.args.dropout_embed,
                self.args.dropout_rnn,
                self.args.dropout,
                self.args.d_s,
                self.args.K,
                self.args.mlp_size,
                self.args.rnn_layernorm,
                torch.device('cuda:{}'.format(torch.cuda.current_device())),
                ffn_flag,
                self.args.init_flag
            ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    def _get_data(self, flag):
        args = self.args
        
        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'WTH':Dataset_Custom,
        }
        
        Data = data_dict[self.args.data]
        # TODO: what is this for?
        timeenc = 0 if args.embed!='timeF' else 1
        
        if flag in ['test']:
            shuffle_flag = False; drop_last = True; batch_size = args.batch_test; freq = args.freq
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_train; freq = args.freq
        
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        
        if flag == 'val' or flag =='train':
            sampler = StepSampler(data_set, args.batch_step)
            print(flag, len(data_set))
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                # sampler=sampler,
                num_workers=args.num_workers,
                drop_last=drop_last)
        else:
            print(flag, len(data_set))
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last)

        return data_set, data_loader
    
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.lr)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion
    
    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        if 'paragru' in self.args.model:
            states = None
        else:
            states = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            pred, true, states = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark, states)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
            
            if self.args.model in ['pytorch_rnn', 'pararnn_torch'] or 'gru' in self.args.model:
                states = states.clone().detach()
            elif 'pararnn' in self.args.model or 'pytorch_lstm' in self.args.model or 'paralstm_torch' in self.args.model:
                states_copy = [h.clone().detach() for h in states]
                states = states_copy
            elif 'paralstm' in self.args.model:
                    states_copy = []
                    for (h, c) in states:
                        (h, c) = (h.clone().detach(), c.clone().detach())
                        states_copy.append((h, c))
                    states = states_copy
            else:
                raise NotImplementedError
            
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    
    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        # path = os.path.join(self.args.checkpoints, setting)
        # if not os.path.exists(path):
        #     os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
                
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        best_eval = float('inf')
        best_epoch = 0
        final_test_loss = float('inf')
        learning_rate = self.args.lr

        for epoch in range(self.args.epochs):
            iter_count = 0
            train_loss = []
            if 'paragru' in self.args.model:
                states = None
            else:
                states = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, true, states = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark, states)
                loss = criterion(pred, true)
                train_loss.append(loss.item())
                
                loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)

                model_optim.step()
                
                if self.args.model in ['pytorch_rnn', 'pararnn_torch'] or 'gru' in self.args.model:
                    states = states.clone().detach()
                elif 'pararnn' in self.args.model or 'pytorch_lstm' in self.args.model or 'paralstm_torch' in self.args.model:
                    states_copy = [h.clone().detach() for h in states]
                    states = states_copy
                elif 'paralstm' in self.args.model:
                    states_copy = []
                    for (h, c) in states:
                        (h, c) = (h.clone().detach(), c.clone().detach())
                        states_copy.append((h, c))
                    states = states_copy
                else:
                    raise NotImplementedError
                
                if (i+1) % self.args.log_interval==0:
                    # print("\titers: {0}, epoch: {1} | train loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.epochs - epoch)*train_steps - i)
                    # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                    # loss.backward(retain_graph=True)
                    # nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)

                    # model_optim.step()
            
            train_time = time.time() - epoch_time
            # print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("| Epoch: {0} | Steps: {1} | Time: {2:.2f} |  lr: {3:.7f} | Train Loss: {4:.4f} | Vali Loss: {5:.4f} | Test Loss: {6:.4f}".format(
                epoch + 1, train_steps, train_time, learning_rate, train_loss, vali_loss, test_loss))
            
            # result save
            folder_path = self.args.pth + '/results/'+self.args.model+'/'+self.args.data+'/' + setting +'/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            with open(folder_path+'log.txt', 'a+') as f:
                f.write("| Epoch: {0} | Steps: {1} | Time: {2:.2f} | lr: {3:.7f} | Train Loss: {4:.4f} | Vali Loss: {5:.4f} | Test Loss: {6:.4f} \n".format(
                epoch + 1, train_steps, train_time, learning_rate, train_loss, vali_loss, test_loss))
            
            if (vali_loss < best_eval):
                best_eval = vali_loss
                final_test_loss = test_loss
                best_epoch = epoch
                
                model_clone = copy.deepcopy(self.model.state_dict())
                opti_clone = copy.deepcopy(model_optim.state_dict())
                patience = 0
                
                ckp_folder = self.args.pth + '/checkpoints/'+self.args.model+'/'+self.args.data+'/' + setting +'/'
                if not os.path.exists(ckp_folder):
                    os.makedirs(ckp_folder)
                ckp_path = ckp_folder +'ckp_best.pt'
                torch.save(self.model, ckp_path)

            elif patience > self.args.patience:
                # if the validation loss does not drop for 3 epochs then reduce the learning rate by half
                self.model.load_state_dict(model_clone)
                model_optim.load_state_dict(opti_clone)
                patience = 0
                learning_rate = learning_rate * 0.5
                
                adjust_learning_rate_no_change(model_optim, learning_rate)
                # break the training if the learning rate is too small
                if learning_rate < 1e-6:
                    break
                print('| Reducing learning rate to {} | test loss {}'.format(learning_rate, str(round(test_loss, 4))))
            else: 
                patience += 1

        # best_model_path = path+'/'+'checkpoint.pt'
        # self.model.load_state_dict(torch.load(best_model_path))
        self.model.load_state_dict(model_clone)
        
        with open(folder_path+'log.txt', 'a+') as f:
            f.write('Best epoch {0} with test loss {1:.4f}'.format(best_epoch, final_test_loss))
        
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        if 'paragru' in self.args.model:
            states = None
        else:
            states = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            pred, true, states = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark, states)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
            
            if self.args.model in ['pytorch_rnn', 'pararnn_torch'] or 'gru' in self.args.model:
                states = states.clone().detach()
            elif 'pararnn' in self.args.model or 'pytorch_lstm' in self.args.model or 'paralstm_torch' in self.args.model:
                states_copy = [h.clone().detach() for h in states]
                states = states_copy
            elif 'paralstm' in self.args.model:
                    states_copy = []
                    for (h, c) in states:
                        (h, c) = (h.clone().detach(), c.clone().detach())
                        states_copy.append((h, c))
                    states = states_copy
            else:
                raise NotImplementedError

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        # folder_path = './results/'+self.args.model+'/' + setting +'/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        # np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path+'pred.npy', preds)
        # np.save(folder_path+'true.npy', trues)
        
        return np.array([mae, mse, rmse, mape, mspe])
    
    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, states):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        
        outputs, states = self.model(batch_x, batch_x_mark, states)

        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        
        outputs = outputs.permute(1, 0, 2)
        outputs = outputs[:,-self.args.pred_len:,:]

        return outputs, batch_y, states