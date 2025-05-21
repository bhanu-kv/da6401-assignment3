import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import wandb

from data_prep import load_data, Vocabulary, TransliterationDataset, collate_fn
from eval import evaluate, evaluate_attention
from seq2seq import Seq2SeqTransliteratorWithAttention  # Your attention model

# ======== DATA PREP ========
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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ======== SWEEP CONFIG (Attention Model) ========
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
    'parameters': {
        'embed_size': {'values': [64, 128, 256]},
        'hidden_size': {'values': [128, 256]},
        'num_layers': {'values': [1, 2, 3]},
        'cell_type': {'values': ['RNN', 'GRU', 'LSTM']},
        'dropout': {'values': [0.2, 0.3]},
        'learning_rate': {'min': 1e-4, 'max': 1e-3},
        'epochs': {'value': 10},
        'batch_size': {'values': [64]}
    }
}


# ======== TRAIN FUNCTION WITH ATTENTION ========
def train_attention_model(config=None):
    with wandb.init(config=config):
        config = wandb.config
        name = f'emb_{config.embed_size}_hid_{config.hidden_size}_numl_{config.num_layers}_dropout_{config.dropout}_lr_{config.learning_rate}_attn'
        wandb.run.name = name
        wandb.run.save()

        model = Seq2SeqTransliteratorWithAttention({
            'src_vocab_size': SRC_VOCAB_SIZE,
            'tgt_vocab_size': TGT_VOCAB_SIZE,
            'embed_size': config.embed_size,
            'hidden_size': config.hidden_size,
            'num_layers': config.num_layers,
            'cell_type': config.cell_type,
            'dropout': config.dropout,
            'device': DEVICE
        }).to(DEVICE)

        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        best_val_acc = 0.0

        for epoch in range(config.epochs):
            model.train()
            train_loss = 0.0
            for src_batch, tgt_batch in train_loader:
                src_batch = src_batch.to(DEVICE)
                tgt_batch = tgt_batch.to(DEVICE)
                optimizer.zero_grad()

                # Model should return output logits of shape (seq_len, batch, vocab_size)
                outputs, attn_weights = model(src_batch, tgt_batch, teacher_forcing_ratio=0.5)

                # outputs: (tgt_len, batch, vocab_size) -> reshape for loss
                outputs = outputs[1:].reshape(-1, outputs.size(-1))
                tgt = tgt_batch[1:].reshape(-1)
                loss = criterion(outputs, tgt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            val_loss, val_acc = evaluate_attention(model, dev_loader, criterion, PAD_IDX)
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss / len(train_loader),
                "val_loss": val_loss,
                "val_accuracy": val_acc
            })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "best_attention_model.pth")

        wandb.finish()


# ======== FINAL TRAINING FUNCTION ========
def train_attention_final_model(model, train_loader, test_loader, config):
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0
        for src_batch, tgt_batch in train_loader:
            src_batch, tgt_batch = src_batch.to(DEVICE), tgt_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs, attn_weights = model(src_batch, tgt_batch, teacher_forcing_ratio=0.5)
            outputs = outputs[1:].reshape(-1, outputs.size(-1))
            tgt = tgt_batch[1:].reshape(-1)
            loss = criterion(outputs[1:].view(-1, outputs.size(-1)), tgt_batch[1:].view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        test_loss, test_acc = evaluate_attention(model, test_loader, criterion, PAD_IDX)
        print(f"Epoch {epoch + 1} | Train Loss: {train_loss / len(train_loader):.4f} | Test Accuracy: {test_acc:.4f}")



# ======== RUN SWEEP ========
project_name = 'DA6401 - Assignment3'
entity = 'CE21B031'

# To start sweep
sweep_id = wandb.sweep(sweep_config, project=project_name, entity=entity)
wandb.agent(sweep_id, train_attention_model, count=30)
