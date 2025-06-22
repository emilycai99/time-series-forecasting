import configargparse

def get_parser():
    parser = configargparse.ArgumentParser(description='[RNN] Long Sequences Forecasting')
    ########################### config #############################################################
    parser.add_argument('--config', required=False,
                        is_config_file_arg=True,
                        help='Path of the main YAML config file.')
    ########################### data #############################################################
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='data')
    parser.add_argument('--root_path', type=str, default='/home/r11user3/Yuxi/ParaRNN/ETT/data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')    
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--sep', type=str, default=',', help='csv file separator')
    # embedding
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--pos_enc', action='store_true', default=False, help='include positional encoding or not')
    parser.add_argument('--temp_enc', action='store_true',default=False, help='include temporal encoding or not')
    # length
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of Informer encoder')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
    ########################### model parameter #############################################################
    parser.add_argument('--model', type=str, default='pararnn',
                        help='which model to use')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='hidden size of RNN')
    parser.add_argument('--embed_size', type=int, default=128,
                        help='embedding size')
    parser.add_argument('--mlp_size', type=int, default=None,
                        help='hidden size for FFN')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of RNN (bidirectional) layers')
    parser.add_argument('--update_nonlinearity', type=str, default='tanh', choices=['tanh', 'relu', 'linear'],
                        help='update nonlinearity for RNN')
    # parser.add_argument('--bias', action='store_false',
    #                     help='whether to add bias in RNN')
    parser.add_argument('--d_s', type=int, default=2,
                        help="size of hidden state for each small RNN")
    parser.add_argument('--K', type=int, default=None,
                        help="number of small RNNs")
    parser.add_argument('--dropout', type=float, default=0.1,
                        help="dropout rate")
    parser.add_argument('--dropout_rnn', type=float, default=0.0,
                        help='dropout rate after each rnn layer')
    parser.add_argument('--rnn_layernorm', action='store_true', 
                        help='whether to use layernorm in rnn layer')
    parser.add_argument('--dropout_embed', type=float, default=0.1,
                        help="dropout rate")
    parser.add_argument('--init_flag', action='store_true', help='whether to use initialization', default=False)

    ########################### training parameter ##########################################################
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of total epochs')
    parser.add_argument('--batch_train', type=int, default=32,
                        help='batch size of training')
    parser.add_argument('--batch_test', type=int, default=1000,
                        help='batch size of testing')
    parser.add_argument('--lr', type=float, default=0.00251,
                        help='learning rate')
    parser.add_argument('--cuda', action='store_false',
                        help='use CUDA (default: True)')
    parser.add_argument('--use_gpu', action='store_false', default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                            help='report interval (default: 20')
    parser.add_argument('--pth', type=str, default='/home/r11user3/Yuxi/ParaRNN/ETT/ptb_style',
                        help="where to write the results")
    parser.add_argument('--checkpoints', type=str, default='/home/r11user3/Yuxi/ParaRNN/ETT/ptb_style/checkpoints/', help='location of model checkpoints')
    parser.add_argument('--clip', type=float, default=1,
                        help='grad_clip_value')
    parser.add_argument('--optim', type=str, default='Adam',
                            help='optimizer to use (default: Adam)') 
    parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--batch_step', type=int, default=1, help='step of selecting a batch of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    
    ########################### Transformer ##########################################################################################################
    ########################### model parameter #############################################################
    parser.add_argument('--init', default='normal', type=str, help='parameter initializer to use.')
    parser.add_argument('--init_range', type=float, default=0.1, help='parameters initialized by U(-init_range, init_range)')
    parser.add_argument('--init_std', type=float, default=0.02, help='parameters initialized by N(0, init_std)')
    
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads (including seasonal and non-seasonal)')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--d_head', type=int, default=50, help='head dimension')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--attn_dropout', type=float, default=0.05, help='attention dropout')
    parser.add_argument('--same_length', action='store_true', help='use the same attn length for all tokens')
    parser.add_argument('--recurrent_layer', type=int, default=1, help='specify which layer to be recurrent layer')
    parser.add_argument('--d_state', type=int, default=512, help='state dimension')
    parser.add_argument('--n_state', type=int, default=8, help='number of states')
    parser.add_argument('--n_chunk', type=int, default=8, help='number of chunks')
    parser.add_argument('--pre_lnorm', action='store_true', help='apply LayerNorm to the input instead of the output')

    
    ########################### data #############################################################
    parser.add_argument('--ext_len', type=int, default=0, help='length of the extended context')
    parser.add_argument('--mem_len', type=int, default=0, help='length of the retained previous heads')
    
    ########################### training parameter ##########################################################
    parser.add_argument('--scheduler', default='default', type=str, choices=['cosine', 'default'],
                    help='lr scheduler to use.')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--to_clip', type=bool, default=False, help='use gradient clipping')
    parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
    parser.add_argument('--warmup_step', type=int, default=0, help='upper epoch limit')




    
    return parser