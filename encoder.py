import torch
import torch.nn as nn
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, input_size, config):
        super().__init__()
        self.embedding = nn.Embedding(input_size, config['embed_size'])
        self.dropout = nn.Dropout(config['dropout'])
        self.rnn = self._create_rnn_layer(
            config['cell_type'],
            config['embed_size'],
            config['hidden_size'],
            config['num_layers'],
            config['dropout']
        )

    def _create_rnn_layer(self, cell_type, in_size, hid_size, n_layers, dropout):
        cells = {
            'RNN': nn.RNN,
            'LSTM': nn.LSTM,
            'GRU': nn.GRU
        }
        return cells[cell_type](
            in_size, hid_size, n_layers,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=False
        )

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden