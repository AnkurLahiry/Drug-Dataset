from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
from torchtext.vocab import build_vocab_from_iterator
from collections import Counter


def build_vocab(data):
    counter = Counter()
    for text in data:
        counter.update(text)
    vocab = build_vocab_from_iterator([counter.keys()], specials=['<unk>', '<pad>'])
    vocab.set_default_index(vocab['<unk>'])
    return vocab

def tokens_to_indices(tokens, vocab):
    return [vocab[token] for token in tokens]

