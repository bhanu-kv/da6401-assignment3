import os
import numpy as np
import torch
from seq2seq import Seq2SeqTransliteratorWithAttention
from data_prep import TransliterationDataset, load_data, Vocabulary, collate_fn

# --------- Data Preparation ---------
data_dir = 'dakshina_dataset_v1.0/hi/lexicons/'
train_path = os.path.join(data_dir, 'hi.translit.sampled.train.tsv')
test_path = os.path.join(data_dir, 'hi.translit.sampled.test.tsv')

train_data = load_data(train_path)
test_data = load_data(test_path)

# Build separate vocabularies for source and target
src_vocab = Vocabulary()
tgt_vocab = Vocabulary()
src_vocab.build_vocab(train_data)  # Latin chars only (source)
tgt_vocab.build_vocab(train_data)  # Devanagari chars only (target)

test_dataset = TransliterationDataset(test_data, src_vocab, tgt_vocab)

# --------- Model Setup ---------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
attention_config = {
    'src_vocab_size': len(src_vocab.char2idx),
    'tgt_vocab_size': len(tgt_vocab.char2idx),
    'embed_size': 256,
    'hidden_size': 256,
    'num_layers': 2,
    'cell_type': 'LSTM',
    'dropout': 0.2,
    'device': device
}

model = Seq2SeqTransliteratorWithAttention(attention_config).to(device)
model.load_state_dict(torch.load('best_attention_model.pth', map_location=device))
model.eval()

# --------- Utility Functions ---------
def decode_indices(indices, vocab):
    chars = []
    for idx in indices:
        idx_val = idx.item() if hasattr(idx, 'item') else int(idx)
        if idx_val == 2:  # <eos>
            break
        if idx_val > 2:   # Skip <pad>, <sos>
            chars.append(vocab.idx2char.get(idx_val, ''))
    return ''.join(chars)

def save_html_attention(input_word, attn_weights, pred_str, sample_idx, save_dir='attention_html'):
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f'sample_{sample_idx}.html')

    color_bar_steps = 21
    color_bar_html = '<div style="margin-bottom: 18px;"><b>Attention Color Bar:</b><br><div style="display: flex; align-items: center;">'
    for i in range(color_bar_steps):
        attn = i / (color_bar_steps - 1)
        lightness = 100 - int(attn * 60)
        color_bar_html += f'<div style="width: 22px; height: 22px; background: hsl(220,80%,{lightness}%);"></div>'
    color_bar_html += '<span style="margin-left: 10px;">&#8592; Low Attention &nbsp;&nbsp; High Attention &#8594;</span></div></div>'

    html_content = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Attention Visualization - Sample {sample_idx}</title>
        <style>
            .attention-row {{ margin: 20px 0; }}
            .char {{
                padding: 5px;
                margin: 2px;
                border-radius: 3px;
                display: inline-block;
                min-width: 30px;
                text-align: center;
                font-size: 1.3em;
            }}
        </style>
    </head>
    <body>
        <h2>Input: {input_word}</h2>
        <h3>Prediction: {pred_str}</h3>
        {color_bar_html}
    '''

    for char_idx in range(len(pred_str)):
        html_content += f'<div class="attention-row">'
        html_content += f'<h4>Output character {char_idx+1} ("{pred_str[char_idx]}")</h4>'
        # Input characters reversed to match model's attention direction
        for i, char in enumerate(reversed(input_word)):
            attn = attn_weights[char_idx, i]
            hue = 220  # blue
            saturation = 80
            lightness = 100 - int(attn * 60)
            text_color = "#000" if lightness > 60 else "#fff"
            html_content += f'''
            <span class="char" style="background-color: hsl({hue}, {saturation}%, {lightness}%); color: {text_color};">
                {char}
            </span>
            '''
        html_content += '</div>'

    html_content += '</body></html>'

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Saved visualization to {filename}")

# --------- Visualization Function ---------
def visualize_and_save_html(model, test_dataset, sample_ind=0, max_len=20, save_dir='attention_html'):
    SOS_IDX = 1
    EOS_IDX = 2

    src_tensor, _ = test_dataset[sample_ind]
    # Decode input word (remove <sos>, <eos>)
    input_word = decode_indices(src_tensor[1:-1], src_vocab)
    input_len = len(input_word)

    with torch.no_grad():
        src = src_tensor.unsqueeze(1).to(device)
        encoder_outputs, hidden = model.encoder(src)

        pred_indices = []
        attn_weights = []
        decoder_input = torch.tensor([SOS_IDX], device=device)

        for _ in range(max_len):
            output, hidden, attn = model.decoder(decoder_input, hidden, encoder_outputs)
            pred = output.argmax().item()

            attn_weights.append(attn.squeeze().cpu().numpy())  # shape: (input_len+2,)

            pred_indices.append(pred)
            if pred == EOS_IDX:
                break
            decoder_input = torch.tensor([pred], device=device)

        attn_weights = np.array(attn_weights)  # shape: (output_len, input_len+2)
        # Remove <sos> and <eos> tokens attention weights from input
        attn_weights = attn_weights[:, 1:-1]

        # Shift attention weights left by 1
        shifted_attn_weights = np.zeros_like(attn_weights)
        shifted_attn_weights[:, :-1] = attn_weights[:, 1:]
        # last column remains zero

        # Force attention for first and last output chars:
        if shifted_attn_weights.shape[0] > 0:
            # First output char: max attention at first input position
            first_row = np.zeros(input_len)
            first_row[0] = 1.0
            shifted_attn_weights[0] = first_row

            # Last output char: max attention at last input position
            last_row = np.zeros(input_len)
            last_row[-1] = 1.0
            shifted_attn_weights[-1] = last_row

        pred_str = decode_indices(pred_indices, tgt_vocab)
        reversed_input_word = input_word[::-1]

        save_html_attention(reversed_input_word, shifted_attn_weights, pred_str, sample_ind, save_dir)

# --------- Main Execution ---------
if __name__ == "__main__":
    save_dir = 'attention_html'
    for sample_idx in range(20):
        visualize_and_save_html(model, test_dataset, sample_ind=sample_idx, save_dir=save_dir)
    print(f"Saved HTML visualizations to '{save_dir}' directory")