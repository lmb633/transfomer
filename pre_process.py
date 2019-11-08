import os
import pickle
from collections import Counter

import jieba
import nltk
from tqdm import tqdm
import unicodedata
import re

data_file = 'data.pkl'
vocab_file = 'vocab.pkl'
n_src_vocab = 15000
n_tgt_vocab = 15000  # target
maxlen_in = 50
maxlen_out = 100

sos_id = 1
eos_id = 2
unk_id = 3

train_translation_en_filename = 'data/ai_challenger_translation_train_20170904/translation_train_data_20170904/train.en'
train_translation_zh_filename = 'data/ai_challenger_translation_train_20170904/translation_train_data_20170904/train.zh'
valid_translation_en_filename = 'data/ai_challenger_translation_validation_20170912/translation_validation_20170912/valid.en'
valid_translation_zh_filename = 'data/ai_challenger_translation_validation_20170912/translation_validation_20170912/valid.zh'


def encode_text(word_map, c):
    return [word_map.get(word, word_map['<unk>']) for word in c]


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def build_vocab(token, word2idx, idx2char):
    if token not in word2idx:
        next_index = len(word2idx)
        word2idx[token] = next_index
        idx2char[next_index] = token


def process(file, lang='zh'):
    print('processing {}...'.format(file))
    with open(file, 'r', encoding='utf-8') as f:
        data = f.readlines()

    word_freq = Counter()
    lengths = []

    for line in tqdm(data):
        sentence = line.strip()
        if lang == 'en':
            sentence_en = sentence.lower()
            tokens = [normalizeString(s) for s in nltk.word_tokenize(sentence_en)]
            word_freq.update(list(tokens))
            vocab_size = n_src_vocab
        else:
            seg_list = jieba.cut(sentence.strip())
            tokens = list(seg_list)
            word_freq.update(list(tokens))
            vocab_size = n_tgt_vocab

        lengths.append(len(tokens))

    words = word_freq.most_common(vocab_size - 4)
    word_map = {k[0]: v + 4 for v, k in enumerate(words)}
    word_map['<pad>'] = 0
    word_map['<sos>'] = 1
    word_map['<eos>'] = 2
    word_map['<unk>'] = 3
    print(len(word_map))
    print(words[:100])
    #
    # n, bins, patches = plt.hist(lengths, 50, density=True, facecolor='g', alpha=0.75)
    #
    # plt.xlabel('Lengths')
    # plt.ylabel('Probability')
    # plt.title('Histogram of Lengths')
    # plt.grid(True)
    # plt.show()

    word2idx = word_map
    idx2char = {v: k for k, v in word2idx.items()}

    return word2idx, idx2char


def get_data(in_file, out_file):
    print('getting data {}->{}...'.format(in_file, out_file))
    with open(in_file, 'r', encoding='utf-8') as file:
        in_lines = file.readlines()
    with open(out_file, 'r', encoding='utf-8') as file:
        out_lines = file.readlines()

    samples = []

    for i in tqdm(range(len(in_lines))):
        sentence_zh = in_lines[i].strip()
        tokens = jieba.cut(sentence_zh.strip())
        in_data = encode_text(src_char2idx, tokens)

        sentence_en = out_lines[i].strip().lower()
        tokens = [normalizeString(s.strip()) for s in nltk.word_tokenize(sentence_en)]
        out_data = [sos_id] + encode_text(tgt_char2idx, tokens) + [eos_id]

        if len(in_data) < maxlen_in and len(out_data) < maxlen_out and unk_id not in in_data and unk_id not in out_data:
            samples.append({'in': in_data, 'out': out_data})
    return samples


if __name__ == '__main__':
    if os.path.isfile(vocab_file):
        with open(vocab_file, 'rb') as file:
            data = pickle.load(file)

        src_char2idx = data['dict']['src_char2idx']
        src_idx2char = data['dict']['src_idx2char']
        tgt_char2idx = data['dict']['tgt_char2idx']
        tgt_idx2char = data['dict']['tgt_idx2char']

    else:
        src_char2idx, src_idx2char = process(train_translation_zh_filename, lang='zh')
        tgt_char2idx, tgt_idx2char = process(train_translation_en_filename, lang='en')

        print(len(src_char2idx))
        print(len(tgt_char2idx))

        data = {
            'dict': {
                'src_char2idx': src_char2idx,
                'src_idx2char': src_idx2char,
                'tgt_char2idx': tgt_char2idx,
                'tgt_idx2char': tgt_idx2char
            }
        }
        with open(vocab_file, 'wb') as file:
            pickle.dump(data, file)

    train = get_data(train_translation_zh_filename, train_translation_en_filename)
    valid = get_data(valid_translation_zh_filename, valid_translation_en_filename)

    data = {
        'train': train,
        'valid': valid
    }

    print('num_train: ' + str(len(train)))
    print('num_valid: ' + str(len(valid)))

    with open(data_file, 'wb') as file:
        pickle.dump(data, file)
