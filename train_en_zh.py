import torch
from torch import nn
from model import Seq2Seq, Decoder, Encoder, EncoderLayer, DecoderLayer, SelfAttention, PositionwiseFeedforward, NoamOpt
from data_gen import AiChallenger2017Dataset, data, pad_collate
import time
import os

batch_size = 256
num_workers = 4
epoch = 100
clip = 1
print_freq = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pad_idx = 0
vocab_size = 15000
hid_dim = 512
n_layers = 6
n_heads = 8
pf_dim = 2048
drop_out = 0.1


def train():
    start_epoch = 0
    checkpoint_path = 'BEST_checkpoint.tar'
    best_loss = float('inf')
    epochs_since_improvement = 0

    if os.path.exists(checkpoint_path):
        print('load checkpoint...')
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        optimizer = checkpoint['optimizer']
    else:
        print('train from begining...')
        encoder = Encoder(vocab_size, hid_dim, n_layers, n_heads, pf_dim, EncoderLayer, SelfAttention, PositionwiseFeedforward, drop_out, device)
        decoder = Decoder(vocab_size, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, drop_out, device)

        model = Seq2Seq(encoder, decoder, pad_idx, device).to(device)
        optimizer = NoamOpt(hid_dim, 1, 2000, torch.optim.Adam(model.parameters()))

    train_dataset = AiChallenger2017Dataset('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=pad_collate,
                                               shuffle=True, num_workers=num_workers)
    valid_dataset = AiChallenger2017Dataset('valid')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, collate_fn=pad_collate,
                                               shuffle=True, num_workers=num_workers)
    print('train size', len(train_dataset), 'valid size', len(valid_dataset))

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    for i in range(start_epoch, epoch):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        valid_loss = value_epoch(model, valid_loader, criterion)
        print('epoch', i, 'avg train loss', train_loss, 'avg valid loss', valid_loss)
        if valid_loss < best_loss:
            best_loss = valid_loss
            epochs_since_improvement = 0
            save_checkpoint(i, epochs_since_improvement, model, optimizer, best_loss)
        else:
            epochs_since_improvement += 1


def train_epoch(model, train_loader, optimizer, criteriaon):
    model.train()
    epoch_loss = 0
    for i, (batch) in enumerate(train_loader):
        src, tgt, length = batch
        src = src.to(device)
        tgt = tgt.to(device)
        optimizer.optimizer.zero_grad()
        output = model(src, tgt[:, :-1])
        output = output.contiguous().view(-1, output.shape[-1])
        tgt = tgt[:, 1:].contiguous().view(-1)
        loss = criteriaon(output, tgt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        avg_loss = epoch_loss / (i + 1)
        if i % print_freq == 0:
            print('{}/{} avg loss'.format(i, len(train_loader)), avg_loss)
    return avg_loss


def value_epoch(model, valid_loader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            src, tgt, length = batch
            src = src.to(device)
            tgt = tgt.to(device)
            output = model(src, tgt[:, :-1])
            output = output.contiguous().view(-1, output.shape[-1])
            tgt = tgt[:, 1:].contiguous().view(-1)
            loss = criterion(output, tgt)
            epoch_loss += loss.item()
        return epoch_loss / len(valid_loader)


def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'loss': best_loss,
             'model': model,
             'optimizer': optimizer}

    filename = 'checkpoint.tar'
    torch.save(state, filename)
    torch.save(state, 'BEST_checkpoint.tar')


if __name__ == '__main__':
    train()
