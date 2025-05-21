import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(model, dataloader, criterion, PAD_IDX):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for src_batch, tgt_batch in dataloader:
            src_batch = src_batch.to(DEVICE)
            tgt_batch = tgt_batch.to(DEVICE)
            outputs = model(src_batch, tgt_batch, teacher_forcing_ratio=0)
            loss = criterion(
                outputs[1:].view(-1, outputs.size(-1)),
                tgt_batch[1:].view(-1)
            )
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 2)
            correct += (predicted[1:] == tgt_batch[1:]).sum().item()
            total += (tgt_batch[1:] != PAD_IDX).sum().item()
    return total_loss/len(dataloader), correct/total if total > 0 else 0

def evaluate_attention(model, dataloader, criterion, pad_idx):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for src_batch, tgt_batch in dataloader:
            src_batch = src_batch.to(DEVICE)
            tgt_batch = tgt_batch.to(DEVICE)
            
            # Unpack the tuple: outputs is first element
            outputs, _ = model(src_batch, tgt_batch, teacher_forcing_ratio=0)
            
            # Now outputs is a tensor, not a tuple
            loss = criterion(
                outputs[1:].view(-1, outputs.size(-1)),
                tgt_batch[1:].view(-1)
            )
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 2)
            correct += (predicted[1:] == tgt_batch[1:]).sum().item()
            total += (tgt_batch[1:] != pad_idx).sum().item()
    
    return total_loss/len(dataloader), correct/total
