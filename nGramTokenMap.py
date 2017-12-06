from os.path import isfile, split
from collections import defaultdict, Counter
from string import punctuation
import pickle
import gzip
import re


def tokenise(review):
    # punctuation string: '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    punctuation_regex = re.compile('[{}]'.format(punctuation.replace("'", "")))
    review = review.strip()
    review = punctuation_regex.sub(' ', review)
    review = re.sub(r"'+", "", review)
    tokens = review.lower().split()
    tokens = ['6'*len(token) if token.isdigit() else token for token in tokens]
    return tokens


def save_vocab_object_to_pickle(vocab_object):
    filename = vocab_object.get_vocab_object_file_name()
    print('Pickling Vocab Object as {}'.format(filename))
    with open(filename, 'wb') as handle:
        pickle.dump(vocab_object, handle, pickle.HIGHEST_PROTOCOL)
    print('Pickled!')
    return None


class Vocab(object):
    def __init__(self, path, unk='<UNK>'):
        self.path = path
        self.unknown_word = unk
        if self.pickle_exists():
            self.load_vocab_object_from_pickle(self, self.get_vocab_object_file_name())
        else:
            self.vocab_freq = self.get_vocab_freq()
            self.sorted_vocab_freq = self.get_sorted_vocab_freq()
            self.token2idx = {token: idx for idx, (token, freq) in enumerate(self.sorted_vocab_freq)}
            self.idx2token = {idx: token for idx, (token, freq) in enumerate(self.sorted_vocab_freq)}

    def get_vocab_freq(self):
        self.vocab_freq = defaultdict(int)
        for doc in self.parse():
            review = doc['reviewText']
            tokens = tokenise(review)
            for token in tokens:
                self.vocab_freq[token] += 1
        self.vocab_freq = Counter(self.vocab_freq)
        return self.vocab_freq

    # Reference: http://jmcauley.ucsd.edu/data/amazon/links.html
    def parse(self):
        g = gzip.open(self.path, 'rb')
        for l in g:
            yield eval(l)

    def get_sorted_vocab_freq(self):
        self.sorted_vocab_freq = sorted(self.vocab_freq.items(), key=lambda x: x[1], reverse=True)
        return self.sorted_vocab_freq

    def get_vocab_object_file_name(self):
        path_to, filename = split(self.path)
        start = 'reviews'
        end = '_5.json.gz'
        assert filename.startswith(start) and filename.endswith(end)
        subset = filename[len(start): -len(end)]
        picklename = 'Vocab_object{}.pickle'.format(subset)
        if len(path_to) == 0:
            return picklename
        else:
            return path_to + '/' + picklename

    def pickle_exists(self):
        filename = self.get_vocab_object_file_name()
        return isfile(filename)

    @staticmethod
    def load_vocab_object_from_pickle(vocab, pickle_file):
        print("Unpickling:", pickle_file)
        with open(pickle_file, 'rb') as handle:
            vocab_object = pickle.load(handle)
        vocab.vocab_freq = vocab_object.vocab_freq
        vocab.sorted_vocab_freq = vocab_object.sorted_vocab_freq
        vocab.token2idx = vocab_object.token2idx
        vocab.idx2token = vocab_object.idx2token
        return vocab_object


if __name__ == '__main__':
    path = 'reviews_Home_and_Kitchen_5.json.gz'
    vocabulary = Vocab(path)
    save_vocab_object_to_pickle(vocabulary)

