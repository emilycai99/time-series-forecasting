import torch
import torch.nn as nn

class RNN_model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, nonlinearity='tanh', dropout=0.1):
        super(RNN_model, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, nonlinearity=nonlinearity,
                          dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size, bias=True)
    
    def forward(self, input):
        # rnn_out.shape = bsz x seq_len x input_size
        rnn_out, hidden_state = self.rnn(input)
        out = self.fc(rnn_out)
        
        return out

