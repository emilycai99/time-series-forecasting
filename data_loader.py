import torch
from torch.utils.data import Dataset
import pandas as pd

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

        print('mean: ', self.mean)
        print('std: ', self.std)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean

class ts_dataset(Dataset):
    def __init__(self, df:pd.DataFrame, seq_len:int, pred_len:int, flag:str='train',
                 target_features:list=None, scale:bool=False, inverse:bool=False, 
                 train_size:float=0.7, test_size:float=0.2):
        super(ts_dataset, self).__init__()
        
        self.seq_len = seq_len
        self.pred_len = pred_len

        assert flag in ['train', 'val', 'test']
        self.flag = flag
        self.target_features = target_features
        self.scale = scale
        self.inverse = inverse

        self.train_size = train_size
        self.test_size = test_size

        self.__read_data__(df)

    def get_scaler(self):
        if self.scale:
            return self.scaler
        else:
            raise NotImplementedError

    def __read_data__(self, df_raw):
        # determine the train/val/test split
        assert self.train_size + self.test_size <= 1.0, 'train + test should be smaller than 1'

        num_train = int(df_raw.shape[0] * self.train_size)
        num_test = int(df_raw.shape[0] * self.test_size)
        num_val = df_raw.shape[0] - num_train - num_test

        border_bgn = [0, num_train - self.seq_len - self.pred_len + 1, len(df_raw) - num_test - self.seq_len - self.pred_len + 1]
        border_end = [num_train, num_train + num_val, len(df_raw)]
        flag_map = {'train': 0, 'val': 1, 'test': 2}

        if self.scale:
            self.scaler = StandardScaler()
            train_data = df_raw.iloc[border_bgn[0]:border_end[0]]
            self.scaler.fit(train_data)
            df_scaled = self.scaler.transform(df_raw)
            self.data_x = df_scaled.iloc[border_bgn[flag_map[self.flag]]:border_end[flag_map[self.flag]]]
        else:
            self.data_x = df_raw.iloc[border_bgn[flag_map[self.flag]]:border_end[flag_map[self.flag]]]

        if self.scale and not self.inverse:
            self.data_y = df_scaled.iloc[border_bgn[flag_map[self.flag]]:border_end[flag_map[self.flag]]][self.target_features]
        else:
            self.data_y = df_raw.iloc[border_bgn[flag_map[self.flag]]:border_end[flag_map[self.flag]]][self.target_features]

    def get_dataframe(self):
        return self.data_x

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx):
        predictor = torch.tensor(self.data_x.iloc[idx: idx+self.seq_len].values, dtype=torch.float32)
        response = torch.tensor(self.data_y.iloc[idx+self.seq_len: idx+self.seq_len+self.pred_len].values, dtype=torch.float32) 

        return predictor, response  

