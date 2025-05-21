import os
import csv
import random
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from seq2seq import Seq2SeqTransliterator
from data_prep import TransliterationDataset, load_data, Vocabulary, collate_fn

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
train_dataset = TransliterationDataset(train_data, src_vocab, tgt_vocab)
dev_dataset = TransliterationDataset(dev_data, src_vocab, tgt_vocab)
test_dataset = TransliterationDataset(test_data, src_vocab, tgt_vocab)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

def decode_indices(indices, vocab):
    # Remove <sos>, <eos>, <pad>
    return ''.join([vocab.idx2char.get(idx.item(), '') for idx in indices if idx.item() > 2])

def save_and_visualize_predictions(model, test_loader, src_vocab, tgt_vocab, device, save_dir='predictions_vanilla', num_samples=9):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    all_results = []
    
    with torch.no_grad():
        for src, tgt in test_loader:
            src, tgt = src.to(device), tgt.to(device)
            outputs = model(src, tgt, teacher_forcing_ratio=0)
            preds = outputs.argmax(dim=-1)
            for i in range(src.size(1)):
                src_str = decode_indices(src[:, i], src_vocab)
                tgt_str = decode_indices(tgt[:, i], tgt_vocab)
                pred_str = decode_indices(preds[:, i], tgt_vocab)
                all_results.append((src_str, tgt_str, pred_str))
            break  # Only need first batch for creative grid

    # Save all predictions to file
    all_preds_path = os.path.join(save_dir, "predictions_vanilla.txt")
    with open(all_preds_path, "w", encoding="utf-8") as f:
        for i, (src_str, tgt_str, pred_str) in enumerate(all_results):
            f.write(f"{i+1}\tInput: {src_str}\tTarget: {tgt_str}\tPrediction: {pred_str}\n")

    all_preds_path = os.path.join(save_dir, "predictions_vanilla.csv")
    with open(all_preds_path, mode="w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["Index", "Input", "Target", "Prediction"])
        
        # Write each prediction
        for i, (src_str, tgt_str, pred_str) in enumerate(all_results):
            writer.writerow([i + 1, src_str, tgt_str, pred_str])

    # Creative 3x3 grid for first 9 predictions

    os.makedirs(save_dir, exist_ok=True)
    grid_path = os.path.join(save_dir, 'prediction_table.png')

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

    # Shuffle the results to pick random samples
    samples_to_show = min(num_samples, len(all_results))
    random_samples = random.sample(all_results, samples_to_show)

    # Setup the figure
    plt.figure(figsize=(15, 15))

    for i in range(samples_to_show):
        src_str, tgt_str, pred_str = random_samples[i]
        match = (tgt_str == pred_str)
        ax = plt.subplot(3, 3, i + 1)
        ax.set_title(f"Sample {i+1}", color='navy', fontsize=16, pad=15)

        # Format text with styling
        ax.text(0.5, 0.8, f"Input:\n{src_str}", ha='center', va='center',
                color='darkgreen', fontsize=14, wrap=True)
        ax.text(0.5, 0.5, f"Target:\n{tgt_str}", ha='center', va='center',
                color='mediumblue', fontsize=14, wrap=True)
        ax.text(0.5, 0.2, f"Prediction:\n{pred_str}", ha='center', va='center',
                color='green' if match else 'red', fontsize=14, fontweight='bold', wrap=True)

        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

        # Add a box around each subplot
        rect = patches.FancyBboxPatch(
            (0, 0), 1, 1,
            boxstyle="round,pad=0.03",
            linewidth=2,
            edgecolor='lightgray',
            facecolor='none',
            transform=ax.transAxes,
            clip_on=False
        )
        ax.add_patch(rect)

    plt.tight_layout(pad=3)
    plt.savefig(grid_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Saved prediction grid with random samples to: {grid_path}")

def evaluate_test_accuracy(model, test_loader, tgt_vocab, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for src, tgt in test_loader:
            src, tgt = src.to(device), tgt.to(device)
            outputs = model(src, tgt, teacher_forcing_ratio=0)
            preds = outputs.argmax(dim=-1)
            
            # Compare all tokens except <sos>
            mask = (tgt[1:] != tgt_vocab.char2idx['<pad>'])
            correct += (preds[1:] == tgt[1:])[mask].sum().item()
            total += mask.sum().item()
    
    accuracy = correct / total
    print(f"Token-Level Accuracy: {accuracy:.2%}")
    return accuracy


# Load your best model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

best_config = {
    'src_vocab_size': len(src_vocab.char2idx),
    'tgt_vocab_size': len(tgt_vocab.char2idx),
    'embed_size': 64,
    'hidden_size': 128,
    'num_layers': 3,
    'cell_type': 'LSTM',
    'dropout': 0.3,
    'device': device,
    'batch_size': 64
}

model = Seq2SeqTransliterator(best_config).to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.to(device)

# Run predictions and visualization
save_and_visualize_predictions(model, test_loader, src_vocab, tgt_vocab, device, save_dir='predictions_vanilla', num_samples=9)

# Evaluate accuracy on test set
evaluate_test_accuracy(model, test_loader, tgt_vocab, device)
