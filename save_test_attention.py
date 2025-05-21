import os
import csv
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from torch.utils.data import DataLoader
from seq2seq import Seq2SeqTransliteratorWithAttention
from data_prep import TransliterationDataset, load_data, Vocabulary, collate_fn

# --------- Data Preparation ---------
data_dir = 'dakshina_dataset_v1.0/hi/lexicons/'
train_path = os.path.join(data_dir, 'hi.translit.sampled.train.tsv')
dev_path = os.path.join(data_dir, 'hi.translit.sampled.dev.tsv')
test_path = os.path.join(data_dir, 'hi.translit.sampled.test.tsv')

train_data = load_data(train_path)
dev_data = load_data(dev_path)
test_data = load_data(test_path)

src_vocab = Vocabulary()
tgt_vocab = Vocabulary()
src_vocab.build_vocab(train_data)
tgt_vocab.build_vocab(train_data)

SRC_VOCAB_SIZE = len(src_vocab.char2idx)
TGT_VOCAB_SIZE = len(tgt_vocab.char2idx)
PAD_IDX = tgt_vocab.char2idx['<pad>']

batch_size = 64
test_dataset = TransliterationDataset(test_data, src_vocab, tgt_vocab)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# --------- Decoding Utilities ---------
def decode_indices(indices, vocab):
    return ''.join([vocab.idx2char.get(idx.item(), '') for idx in indices if idx.item() > 2])

# --------- Attention Visualization and Prediction Saving ---------
def save_and_visualize_attention_predictions(model, test_loader, src_vocab, tgt_vocab, device, 
                                           save_dir='predictions_attention', num_samples=9):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    all_results = []

    with torch.no_grad():
        for src, tgt in test_loader:
            src, tgt = src.to(device), tgt.to(device)
            outputs, attn_weights = model(src, tgt, teacher_forcing_ratio=0)
            preds = outputs.argmax(dim=-1)
            for i in range(src.size(1)):
                src_str = decode_indices(src[:, i], src_vocab)
                tgt_str = decode_indices(tgt[:, i], tgt_vocab)
                pred_str = decode_indices(preds[:, i], tgt_vocab)
                attn = attn_weights[:, i, :].cpu().numpy()  # (tgt_len, src_len)
                all_results.append((src_str, tgt_str, pred_str, attn))
            break  # Only need first batch for creative grid

    # Save all predictions to CSV
    csv_path = os.path.join(save_dir, "predictions_attention.csv")
    with open(csv_path, mode="w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Input", "Target", "Prediction"])
        for i, (src_str, tgt_str, pred_str, _) in enumerate(all_results):
            writer.writerow([i + 1, src_str, tgt_str, pred_str])

    # Try to find a Devanagari-compatible font
    hindi_font = None
    for font in font_manager.findSystemFonts(fontpaths=None, fontext='ttf'):
        if "NotoSansDevanagari" in font or "Mangal" in font or "Lohit-Devanagari" in font:
            hindi_font = font
            break

    if hindi_font:
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=hindi_font).get_name()
        print(f"Using Hindi-compatible font: {plt.rcParams['font.family']}")
    else:
        print("Warning: Hindi-compatible font not found. Hindi characters may render as boxes.")

    # Shuffle and pick random samples for grid
    samples_to_show = min(num_samples, len(all_results))
    random_samples = random.sample(all_results, samples_to_show)
    
    plt.figure(figsize=(18, 20))  # Increased height to accommodate text
    for i in range(samples_to_show):
        src_str, tgt_str, pred_str, attn = random_samples[i]
        match = (tgt_str == pred_str)
        ax = plt.subplot(3, 3, i + 1)
        
        # Plot attention heatmap first
        attn_crop = attn[1:len(pred_str)+1, :len(src_str)]
        im = ax.imshow(attn_crop, cmap='viridis', aspect='auto')
        
        # Add colorbar to the right
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Set ticks and labels
        ax.set_xticks(range(len(src_str)))
        ax.set_xticklabels(list(src_str), rotation=90, 
                          fontproperties=font_manager.FontProperties(fname=hindi_font) if hindi_font else None,
                          fontsize=10)
        ax.set_yticks(range(len(pred_str)))
        ax.set_yticklabels(list(pred_str), 
                          fontproperties=font_manager.FontProperties(fname=hindi_font) if hindi_font else None,
                          fontsize=10)
        
        # Add text below the heatmap
        ax.text(0.5, -0.35, f"Input: {src_str}\nTarget: {tgt_str}\nPred: {pred_str}",
                ha='center', va='top', transform=ax.transAxes,
                color='darkgreen', fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))
        
        # Add border around the sample
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('lightgray')
            spine.set_linewidth(1)

    # Adjust layout with more space at bottom
    plt.tight_layout(rect=[0, 0.05, 1, 0.95], pad=3, h_pad=3, w_pad=3)
    grid_path = os.path.join(save_dir, 'attention_grid.png')
    plt.savefig(grid_path, dpi=300, bbox_inches='tight')
    plt.show()


# --------- Accuracy Evaluation ---------
def evaluate_attention_model(model, test_loader, tgt_vocab, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for src, tgt in test_loader:
            src, tgt = src.to(device), tgt.to(device)
            outputs, _ = model(src, tgt, teacher_forcing_ratio=0)
            preds = outputs.argmax(dim=-1)
            
            # Compare all tokens except <sos>
            mask = (tgt[1:] != tgt_vocab.char2idx['<pad>'])
            correct += (preds[1:] == tgt[1:])[mask].sum().item()
            total += mask.sum().item()
    
    accuracy = correct / total
    print(f"Token-Level Accuracy: {accuracy:.2%}")
    return accuracy


# --------- Load Model and Run Evaluation ---------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
attention_config = {
    'src_vocab_size': SRC_VOCAB_SIZE,
    'tgt_vocab_size': TGT_VOCAB_SIZE,
    'embed_size': 256,
    'hidden_size': 256,
    'num_layers': 2,
    'cell_type': 'LSTM',
    'dropout': 0.2,
    'device': device
}
model = Seq2SeqTransliteratorWithAttention(attention_config).to(device)
model.load_state_dict(torch.load('best_attention_model.pth', map_location=device))

save_and_visualize_attention_predictions(
    model, test_loader, src_vocab, tgt_vocab, device,
    save_dir='predictions_attention', num_samples=9
)

evaluate_attention_model(model, test_loader, tgt_vocab, device)
