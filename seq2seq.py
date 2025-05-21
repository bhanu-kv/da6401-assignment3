from encoder import Encoder
from decoder import Decoder, DecoderWithAttention
import torch
import torch.nn as nn

class Seq2SeqTransliterator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config['device']
        self.encoder = Encoder(config['src_vocab_size'], config)
        self.decoder = Decoder(config['tgt_vocab_size'], config)

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.size(1)
        tgt_len = tgt.size(0)
        outputs = torch.zeros(tgt_len, batch_size, self.decoder.output_size).to(self.device)
        encoder_outputs, encoder_hidden = self.encoder(src)
        decoder_input = tgt[0]
        decoder_hidden = encoder_hidden
        for t in range(1, tgt_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            outputs[t] = decoder_output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = decoder_output.argmax(1)
            decoder_input = tgt[t] if teacher_force else top1
        return outputs

class Seq2SeqTransliteratorWithAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config['src_vocab_size'], config)
        self.decoder = DecoderWithAttention(
            output_size=config['tgt_vocab_size'],
            embed_size=config['embed_size'],
            hidden_size=config['hidden_size'],
            device=config['device'],
            num_layers=config['num_layers'],
            cell_type=config['cell_type'],
            dropout=config['dropout']
        )
        self.device = config['device']

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        # src: (src_len, batch_size)
        # tgt: (tgt_len, batch_size)
        
        encoder_outputs, encoder_hidden = self.encoder(src)
        decoder_hidden = encoder_hidden
        
        tgt_len, batch_size = tgt.shape
        outputs = torch.zeros(tgt_len, batch_size, self.decoder.fc_out.out_features).to(self.device)
        attention_weights = torch.zeros(tgt_len, batch_size, src.size(0)).to(self.device)
        
        decoder_input = tgt[0]  # <sos>
        
        for t in range(1, tgt_len):
            decoder_output, decoder_hidden, attn_weights = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            outputs[t] = decoder_output
            attention_weights[t] = attn_weights
            
            # Teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = decoder_output.argmax(1)
            decoder_input = tgt[t] if teacher_force else top1
        
        return outputs, attention_weights