import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import warnings
import time
import copy
import os
import numpy as np
from RNN import RNN_model
from data_loader import ts_dataset

warnings.filterwarnings('ignore')
torch.autograd.set_detect_anomaly(True)

def adjust_learning_rate_no_change(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

class Exp_RNN(object):
    def __init__(self, df, args):
        super(Exp_RNN, self).__init__(args)
        self.df = df
        self.args = args
        self.model = self._build_model()

    def _build_model(self):
        model_dict = {
            'rnn': RNN_model
        }

        model = model_dict[self.args.model](
            input_size=self.args.input_size,
            hidden_size=self.args.hidden_size,
            output_size=self.args.output_size,
            num_layers=self.args.num_layers,
            nonlinearity=self.args.nonlinearity,
            dropout=self.args.dropout
        )

        return model
    
    def _get_data(self, flag):
        if flag == 'test':
            shuffle_flag = False
            drop_last = False
            batch_size = self.args.batch_test
        else:
            shuffle_flag = True
            drop_last = False
            batch_size = self.args.batch_train

        dataset = ts_dataset(
            df=self.df,
            seq_len=self.args.seq_len,
            pred_len=self.args.pred_len,
            flag=flag,
            target_features=self.args.target_features,
            scale=self.args.scale,
            inverse=self.args.inverse,
            train_size=self.args.train_size,
            test_size=self.args.test_size
        )

        print(flag, len(dataset))
        data_loader = DataLoader(
            ts_dataset,
            batch_size=batch_size,
            shuffle_flag=shuffle_flag,
            drop_last=drop_last
        )

        return dataset, data_loader
    
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.lr)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion
    
    def val(self, val_data, val_loader, criterion):
        self.model.eval()

        total_loss = 0.0
        sample_count = 0

        for itr, (input, label) in enumerate(val_loader):
            output = self.model(input)

            # reverse standardization
            if self.args.scale and self.args.inverse:
                scaler = val_data.get_scaler()
                output = scaler.inverse_transform(output)

            # select the target features
            # batch size x pred_len x dim
            output = output[:, -self.args.pred_len, -len(self.args.target_features):]

            loss = criterion(output.detach(), label.detach())

            total_loss += loss.item()
            sample_count += output.shape[0]
        
        return total_loss / sample_count

    
    def train(self):
        train_data, train_loader = self._get_data('train')
        val_data, val_loader = self._get_data('val')
        test_data, test_loader = self._get_data('test')

        optimizer = self._select_optimizer()
        criterion = self._select_criterion()

        best_eval = float('inf')
        best_epoch = 0
        final_test_loss = float('inf')
        learning_rate = self.args.lr
        
        for epoch in range(self.args.total_epochs):
            total_loss = 0.0
            sample_count = 0
            time_start = time.time()

            for itr, (input, label) in enumerate(train_loader):
                self.model.train()
                optimizer.zero_grad()
                output = self.model(input)

                # reverse standardization
                if self.args.scale and self.args.inverse:
                    scaler = train_data.get_scaler()
                    output = scaler.inverse_transform(output)

                # select the target features
                # batch size x pred_len x dim
                output = output[:, -self.args.pred_len, -len(self.args.target_features):]

                loss = criterion(output, label)
                loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                optimizer.step()

                total_loss += loss.item()
                sample_count += output.shape[0]

                if (itr + 1) % self.args.log_interval == 0:
                    print("     Batch {0} | Train loss: {1:.4f}".format(itr+1, total_loss/sample_count))

            train_loss = total_loss / sample_count
            val_loss = self.val(val_data, val_loader, criterion)
            test_loss = self.val(test_data, test_loader, criterion)

            end_time = time.time()

            print("Epoch {0} | Time: {1:.2f} | LR: {2:.6f} | Train loss: {3:.4f} | Val loss: {4:.4f} | Test loss: {5:.4f}").format(
                epoch+1, end_time-time_start, learning_rate, train_loss, val_loss, test_loss
            )

            if val_loss < best_eval:
                best_eval = val_loss
                final_test_loss = test_loss
                best_epoch = epoch

                model_clone = copy.deepcopy(self.model.state_dict())
                opti_clone = copy.deepcopy(optimizer.state_dict())
                patience = 0

                ckp_folder = self.args.pth + '/checkpoints/' + self.args.model +'/'
                if not os.path.exists(ckp_folder):
                    os.makedirs(ckp_folder)
                ckp_path = ckp_folder +'ckp_best.pt'
                torch.save(self.model, ckp_path)

            elif patience > self.args.patience:
                # if the validation loss does not drop for 3 epochs then reduce the learning rate by half
                self.model.load_state_dict(model_clone)
                optimizer.load_state_dict(opti_clone)
                patience = 0
                learning_rate = learning_rate * 0.5
                
                adjust_learning_rate_no_change(optimizer, learning_rate)
                # break the training if the learning rate is too small
                if learning_rate < 1e-6:
                    break
                print('Reducing learning rate to {0:6f} | Test loss: {1:4f}'.format(learning_rate, test_loss))
            else: 
                patience += 1


        self.model.load_state_dict(model_clone)
        print('Best epoch {0} with test loss {1:.4f}'.format(best_epoch, final_test_loss)) 

        return self.model
    
    def test(self):
        test_data, test_loader = self._get_data('test')
        self.model.eval()

        preds = []
        trues = []

        for itr, (input, label) in enumerate(test_loader):
            output = self.model(input)

            # reverse standardization
            if self.args.scale and self.args.inverse:
                scaler = test_data.get_scaler()
                output = scaler.inverse_transform(output)

            # select the target features
            # batch size x pred_len x dim
            output = output[:, -self.args.pred_len, -len(self.args.target_features):]

            preds.append(output.detach().cpu().numpy())
            trues.append(label.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)

        # TODO: check
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        mae = MAE(preds, trues)
        mse = MSE(preds, trues)

        print('mse:{0:.4f}, mae:{1:.4f}'.format(mse, mae))

        return mse, mae





                
