import torch
from torch import nn
from model import Seq2Seq, Decoder, Encoder, EncoderLayer, DecoderLayer, SelfAttention, PositionwiseFeedforward, NoamOpt
from data_gen import AiChallenger2017Dataset, data, pad_collate
import time

batch_size = 1
num_workers = 4
epoch = 1
clip = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pad_idx = 0
vocab_size = 15000
hid_dim = 512
n_layers = 6
n_heads = 8
pf_dim = 2048
drop_out = 0.1


def train():
    encoder = Encoder(vocab_size, hid_dim, n_layers, n_heads, pf_dim, EncoderLayer, SelfAttention, PositionwiseFeedforward, drop_out, device)
    decoder = Decoder(vocab_size, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, drop_out, device)

    model = Seq2Seq(encoder, decoder, pad_idx, device).to(device)

    train_dataset = AiChallenger2017Dataset('valid')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=pad_collate,
                                               shuffle=True, num_workers=num_workers)
    valid_dataset = AiChallenger2017Dataset('valid')
    tvalid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, collate_fn=pad_collate,
                                                shuffle=True, num_workers=num_workers)
    print('train size', len(train_dataset), 'valid size', len(valid_dataset))

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    optimizer = NoamOpt(hid_dim, 1, 2000, torch.optim.Adam(model.parameters()))
    criteriaon = nn.CrossEntropyLoss(ignore_index=pad_idx)

    for i in range(epoch):
        train_epoch(model, train_loader, optimizer, criteriaon)


def train_epoch(model, train_dataset, optimizer, criteriaon):
    model.train()
    epoch_loss = 0
    for i, (batch) in enumerate(train_dataset):
        print(batch.shape)
        src, tgt, length = batch
        src = src.to(device)
        tgt = tgt.to(device)
        length = length.to(device)
        optimizer.optimizer.zero_grad()
        output = model(src, tgt)
        print(output.shape)
        output = output.contiguous().view(-1, output.shape[-1])
        print(output.shape)
        tgt = tgt[:, 1:].contiguous().view(-1)
        loss = criteriaon(output, tgt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_dataset)
        print('avg loss', avg_loss)
        time.sleep(100)
    return avg_loss


train()
