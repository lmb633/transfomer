import pickle
import os
import torch
from data_gen import AiChallenger2017Dataset, pad_collate
import numpy as np
import time


def sequence_to_text(seq, idx2char):
    result = [idx2char[idx] for idx in seq]
    return result


vocab_file = 'vocab.pkl'
checkpoint_path = 'BEST_checkpoint.tar'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(vocab_file, 'rb') as file:
    data = pickle.load(file)

src_idx2char = data['dict']['src_idx2char']
tgt_idx2char = data['dict']['tgt_idx2char']

if os.path.exists(checkpoint_path):
    print('load checkpoint...')
    checkpoint = torch.load(checkpoint_path)
    model = checkpoint['model']
    model.eval()
    encoder = model.encoder
    decoder = model.decoder
    make_mask = model.make_masks

valid_dataset = AiChallenger2017Dataset('valid')
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, collate_fn=pad_collate,
                                           shuffle=True, num_workers=1)

with torch.no_grad():
    print(len(valid_loader))
    for i, batch in enumerate(valid_loader):
        src, trg, length = batch
        src = src.to(device)
        trg = trg.to(device)
        print(src, trg)
        src_mask, trg_mask = make_mask(src, trg)
        print(src_mask)
        enc_src = encoder(src, src_mask)
        print(enc_src.shape)
        result_sentence = torch.tensor([[1]]).to(device)
        print(result_sentence)
        while True:
            src_mask, trg_mask = make_mask(src, result_sentence)
            temp_output = decoder(result_sentence, enc_src, trg_mask, src_mask)
            print(temp_output.reshape(-1))
            temp_vocab = np.argmax(temp_output.reshape(-1).cpu())
            print(temp_vocab)
            if temp_vocab == 2:
                break
            result_sentence.data[0].append(temp_vocab)
            time.sleep(100)
        print(i, 'result', result_sentence)

        src_text = src
        tgt_text = trg
        result_text = result_sentence[1:]

        src_text = sequence_to_text(src_text, src_idx2char)
        src_text = ' '.join(src_text)
        print('src_text: ' + src_text)

        tgt_text = sequence_to_text(tgt_text, tgt_idx2char)
        tgt_text = ' '.join(tgt_text)
        print('tgt_text: ' + tgt_text)

        result_text = sequence_to_text(result_text, tgt_idx2char)
        result_text = ' '.join(result_text)
        print('tgt_text: ' + result_text)
