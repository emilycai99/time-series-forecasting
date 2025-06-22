import os
import torch
import time
import numpy as np

import sys
sys.path.append('/home/r11user3/Yuxi/ParaRNN/ETT')
from ptb_style.exp import Exp_RNN
from ptb_style.exp_xl import Exp_TransformerXL
from ptb_style.parser import get_parser

parser = get_parser()
args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
    'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'WTH':{'data':'WTH.csv','T':'WetBulbCelsius','M':[12,12,12],'S':[1,1,1],'MS':[12,12,1]},
    'ECL':{'data':'ECL.csv','T':'MT_320','M':[321,321,321],'S':[1,1,1],'MS':[321,321,1]},
    'Solar':{'data':'solar_AL.csv','T':'POWER_136','M':[137,137,137],'S':[1,1,1],'MS':[137,137,1]},
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

args.detail_freq = args.freq
args.freq = args.freq[-1:]

print('Args in experiment:')
print(args)

# if 'rnn' in args.model:
#     Exp = Exp_RNN
# else:
#     raise NotImplementedError

if 'brt' in args.model:
    Exp = Exp_TransformerXL
    setting = '{m}_ft{ft}_nl{l}_rl{rl}_sl{sl}_el{el}_pl{pl}_nh{h}_ne{ne}_nm{nm}_ds{d}_K{k}_dp{dp}_bsz{b}_lr{lr}_pe{pe}_in{ini}'.format(
                m=args.model, l=args.num_layers, h=args.d_model, d=args.d_s, k=args.K,
                dp=args.dropout, b=args.batch_train, lr=args.lr,
                nm=args.mlp_size, ne=args.embed_size,
                sl=args.seq_len, pl=args.pred_len, pe=args.pos_enc,
                ft=args.features, ini=args.init_flag,
                rl=args.recurrent_layer, el=args.ext_len)
else:
    Exp = Exp_RNN
    # setting record of experiments
    setting = '{m}_ft{ft}_nl{l}_sl{sl}_pl{pl}_nh{h}_ne{ne}_nm{nm}_ds{d}_K{k}_de{de}_dp{dp}_dr{dr}_bsz{b}_lr{lr}_ln{ln}_pe{pe}_in{ini}'.format(
                m=args.model, l=args.num_layers, h=args.hidden_size, d=args.d_s, k=args.K,
                dp=args.dropout, b=args.batch_train, lr=args.lr, dr=args.dropout_rnn,
                nm=args.mlp_size, ln=args.rnn_layernorm, ne=args.embed_size,
                de=args.dropout_embed, sl=args.seq_len, pl=args.pred_len, pe=args.pos_enc,
                ft=args.features, ini=args.init_flag)

log_time = []
log_metric = []
for ii in range(args.itr):
    
    exp = Exp(args) # set experiments
    
    folder_path = args.pth + '/results/'+ args.model+ '/' + args.data + '/' + setting +'/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    with open(folder_path+'log.txt', 'a+') as f:
        print(args, file=f)
        
    train_start = time.time()
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
    train_time = time.time()-train_start
    print('>>>>>>>end training : time cost {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(train_time))

    test_start = time.time()
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    metric = exp.test(setting)
    log_metric.append(metric)
    test_time = time.time()-test_start
    print('>>>>>>>end testing : time cost {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(test_time))
    log_time.append(np.array([train_time,test_time]))

    # folder_path = args.pth + '/results/'+args.model+'/' + setting +'/'
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path)
    # np.save(folder_path+'time.npy', np.array([train_time,test_time]))

    torch.cuda.empty_cache()

# folder_pth = args.pth + '/results/'+args.model +'/'+ args.data
# if not os.path.exists(folder_pth):
#         os.makedirs(folder_pth)
with open(args.pth + '/results/'+args.model +'/' + args.data + '/' + setting +'/metric.txt','a') as f:
    for item in log_metric:
        f.write('metric: mae, mse, rmse, mape, mspe\n')
        f.write("%s\n" % item)