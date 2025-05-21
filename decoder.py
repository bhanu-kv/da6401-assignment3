import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from encoder import Encoder

class Decoder(nn.Module):
    def __init__(self, output_size, config):
        super().__init__()
        self.embedding = nn.Embedding(output_size, config['embed_size'])
        self.dropout = nn.Dropout(config['dropout'])
        self.rnn = self._create_rnn_layer(
            config['cell_type'],
            config['embed_size'],
            config['hidden_size'],
            config['num_layers'],
            config['dropout']
        )
        self.out = nn.Linear(config['hidden_size'], output_size)
        self.output_size = output_size

    def _create_rnn_layer(self, *args, **kwargs):
        return Encoder._create_rnn_layer(self, *args, **kwargs)

    def forward(self, input, hidden, encoder_outputs=None):
        embedded = self.dropout(self.embedding(input)).unsqueeze(0)
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.out(output.squeeze(0))
        return prediction, hidden

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.hid_dim = hid_dim

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (batch, hid_dim)
        # encoder_outputs: (batch, src_len, hid_dim)
        src_len = encoder_outputs.shape[1]
        # Repeat decoder hidden for each src position
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)  # (batch, src_len, hid_dim)
        # Energy: dot product
        energy = torch.sum(encoder_outputs * decoder_hidden, dim=2)  # (batch, src_len)
        attn_weights = F.softmax(energy, dim=1)  # (batch, src_len)
        # Context vector
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch, hid_dim)
        return context, attn_weights

class DecoderWithAttention(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, num_layers, 
                 cell_type, dropout, device):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.device = device
        
        self.embedding = nn.Embedding(output_size, embed_size)
        self.attention = Attention(hidden_size)
        self.rnn = self._create_rnn_layer(
            cell_type=cell_type,
            input_size=embed_size + hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        self.fc_out = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)

    def _create_rnn_layer(self, cell_type, input_size, hidden_size, num_layers, dropout):
        rnn_class = getattr(nn, cell_type.upper())
        return rnn_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

    def forward(self, input, hidden, encoder_outputs):        
        # Convert encoder outputs to batch-first
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # (batch, src_len, hidden)
        
        # Get decoder hidden state for attention
        if isinstance(hidden, tuple):
            hidden_for_attn = hidden[0][-1]  # (batch, hidden)
        else:
            hidden_for_attn = hidden[-1]     # (batch, hidden)
        
        # Calculate attention context
        context, attn_weights = self.attention(hidden_for_attn, encoder_outputs)
        
        # Embed input and combine with context
        embedded = self.dropout(self.embedding(input.unsqueeze(0)))  # (1, batch, embed)
        context = context.unsqueeze(0)  # (1, batch, hidden)
        rnn_input = torch.cat((embedded, context), dim=2)
        
        # RNN forward
        output, new_hidden = self.rnn(rnn_input, hidden)
        
        # Final prediction
        output = self.fc_out(torch.cat((output.squeeze(0), context.squeeze(0)), dim=1))
        
        return output, new_hidden, attn_weights
