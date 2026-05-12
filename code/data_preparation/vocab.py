"""Handles vocabulary words and embeddings. Contains method sentence_to_indices which is used by CustomDataset"""

import warnings
from pathlib import Path
# import pyconll
from collections import Counter, defaultdict
import pickle
import numpy as np


warnings.simplefilter('default', DeprecationWarning)


def get_vocab(conf, train_data, rewrite=False):
    if rewrite:
        print('Rewriting Vocab because rewrite option was set to True')
        return Vocab(conf, train_data)
    elif Path(conf['vocab_file']).exists():
        print('Loading existing Vocab from file')
        with open(conf['vocab_file'], 'rb') as vf:
            return pickle.load(vf)
    else:
        print('Creating Vocab for the first time for this language and order')
        return Vocab(conf, train_data)


class Vocab:
    """Vocab class. More info will be added later.

    Possible keys for vocab are
    'word-index', 'index-word',
    'grammeme-index', 'index-grammeme',
    'char-index', 'index-char'
    """

    def __init__(self, conf, train_data):
        self.conf = conf
        self.train_data = train_data
        self.vocab = {} # dictionary of dictionaries, main object of the class

        self.grammemes_by_freq = {} # dictionary of grammemes and their numbers of occurrences
        self.categories_by_freq = {} # dictionary of categories and their number of occurrences
        self.sorting_order = {} # dictionary used to sort grammemes

        self.embeddings = None # word embeddings

        self.create_vocab()


    def create_vocab(self):
        assert self.conf['train_files'], f'Directory {self.conf["language_folder"]} doesn\'t contain train files!'

        print(f'\t{len(self.train_data)} training sentences for Vocab')

        self.get_all(self.train_data)

        self.get_sorting_order()

        with open(self.conf['vocab_file'], 'wb+') as f:
            pickle.dump(self, f)
        print(f'\tSaved Vocab: {self.conf["vocab_file"]}')


    def get_all(self, sentences):
        """Method for getting all tokens, chars, grammemes, and singletons for vocab."""

        tokens = set()
        chars = set()
        grammemes = set()
        categories = set()
        grammemes_frequencies = defaultdict(int)
        categories_frequencies = defaultdict(int)

        tokens.add(self.conf['NUM'])
        for sentence in sentences:
            for token in sentence['tokens']:
                # Update tokens and chars
                if not token.isdigit(): # NUM token is already added separately
                    tokens.add(token)
                chars.update(token)

            for tag in sentence['tags']:
                for g in tag.split('|'):
                    if 'POS=' in g:
                        categories.add('POS')
                        categories_frequencies['POS'] += 1
                        grammemes.add(g)
                        grammemes_frequencies[g] += 1
                    else:
                        cat, feat = g.split('=')
                        categories.add(cat)
                        categories_frequencies[cat] += 1
                        grammemes.add(g)
                        grammemes_frequencies[g] += 1


        tokens = [self.conf['PAD'], self.conf['UNK']] + sorted(list(tokens))
        chars = [self.conf['PAD'], self.conf['UNK']] + sorted(list(chars))
        grammemes = [self.conf['PAD'], self.conf['SOS'], self.conf['EOS'], self.conf['UNK']] + sorted(list(grammemes))

        counter = Counter([token for sentence in sentences for token in sentence['tokens']])
        singletons = [token for token, cnt in counter.items() if cnt == 1]

        self.grammemes_by_freq = dict(sorted(grammemes_frequencies.items(), key=lambda i: i[1], reverse=True))
        self.categories_by_freq = dict(sorted(categories_frequencies.items(), key=lambda i: i[1], reverse=True))
        self.vocab['word-index'], self.vocab['index-word'] = get_dictionaries(tokens)
        self.vocab['grammeme-index'], self.vocab['index-grammeme'] = get_dictionaries(grammemes)
        self.vocab['char-index'], self.vocab['index-char'] = get_dictionaries(chars)
        self.vocab['singleton-index'], self.vocab['index-singleton'] = get_dictionaries(singletons)


    def create_embeddings(self, ft=None):
        """Creates embeddings, possibly using fastText. Only used by LSTMEncoder."""

        # these numbers were in original code
        dimension = self.conf['word_embeddings_dimension']
        self.embeddings = np.random.normal(scale=2.0 / (dimension + len(self.vocab['word-index'])),
                                      size=(len(self.vocab['word-index']), dimension))

        if ft is not None:
            for word in ft.words:
                if word.lower() in self.vocab['word-index'].keys():
                    self.embeddings[self.vocab['word-index'][word.lower()]] = ft[word]


    def get_sorting_order(self):
        """Sorts grammemes according to config."""

        # there are actually 10 unique orders for xe loss because POS category is the most frequent

        grammemes = [g for g in self.vocab['grammeme-index'].keys() if '$' not in g]

        # 1. Standard
        if self.conf['order'] == 'standard':
            sorting_order = [g for g in grammemes if 'POS' in g] + [g for g in grammemes if 'POS' not in g]

        # 2. Reverse
        elif self.conf['order'] == 'reverse':
            sorting_order = ([g for g in grammemes if 'POS' in g] + [g for g in grammemes if 'POS' not in g])[::-1]

        # 3. "POS, reverse"
        elif self.conf['order'] == 'pos,reverse':
            sorting_order = [g for g in grammemes if 'POS' in g][::-1] + [g for g in grammemes if 'POS' not in g][::-1]

        # 4. "Grammemes-down"
        elif self.conf['order'] == 'grammemes-down':
            sorting_order = list(self.grammemes_by_freq.keys())

        # 5. "POS,grammemes-down"
        elif self.conf['order'] == 'pos,grammemes-down':
            sorting_order = [g for g in list(self.grammemes_by_freq.keys()) if 'POS' in g] + \
                            [g for g in list(self.grammemes_by_freq.keys()) if 'POS' not in g]

        # 6. "Categories-down"
        elif self.conf['order'] == 'categories-down':
            sorting_order = sorted(grammemes,
                                   key=lambda i: self.categories_by_freq[i.split("=")[0]],
                                   reverse=True)
        # 7. "Grammemes-up"
        elif self.conf['order'] == 'grammemes-up':
            sorting_order = list(self.grammemes_by_freq.keys())[::-1]

        # 8. "POS,grammemes-up"
        elif self.conf['order'] == 'pos,grammemes-up':
            sorting_order = [g for g in list(self.grammemes_by_freq.keys()) if 'POS' in g][::-1] + \
                            [g for g in list(self.grammemes_by_freq.keys()) if 'POS' not in g][::-1]

        # 9. "Categories-up"
        elif self.conf['order'] == 'categories-up':
            sorting_order = sorted(grammemes,
                                   key=lambda i: self.categories_by_freq[i.split("=")[0]])

        # 10. "POS,categories-up"
        elif self.conf['order'] == 'pos,categories-up':
            if_pos_were_false = sorted(grammemes,
                                       key=lambda i: self.categories_by_freq[i.split("=")[0]])
            sorting_order = [g for g in if_pos_were_false if 'POS' in g] + \
                            [g for g in if_pos_were_false if 'POS' not in g]

        else:
            raise ValueError(f'Unknown order of grammemes: {self.conf["order"]}')
        self.sorting_order = {g: i for i, g in enumerate(sorting_order)}


    def collate_fn(self, batch, train_mode=True):
        """Collate method.

            Args:
                batch (dict): Batch returned by dataset
                train_mode (bool): If True, singletons will be randomly substituted by UNK
            """

        # dataset returns dict with keys 'id', 'tokens', 'tags'
        # batch is dict that stores a list for each key

        tokens = [item['tokens'] for item in batch] # list of lists
        tags = [[tag.split('|') for tag in item['tags']] for item in batch]
        max_sentence_length = max(map(len, tokens))
        max_word_length = max([max(map(len, sentence)) for sentence in tokens])
        max_label_length = 2 + max([max(map(len, item)) for item in tags])

        # pad tokens and chars
        new_tokens = []
        new_chars = []
        for item in tokens:
            word_indices = []
            for token in item:
                if token.isdigit():
                    word_indices.append(self.vocab['word-index'][self.conf['NUM']])
                elif train_mode and token in self.vocab['singleton-index'] and np.random.rand() < self.conf['singleton_substitution']:
                    word_indices.append(self.vocab['word-index'][self.conf['UNK']])
                else:
                    word_indices.append(self.vocab['word-index'].get(token, self.vocab['word-index'][self.conf['UNK']]))

            char_indices = []
            for token in item:
                indices = [self.vocab['char-index'].get(c, self.vocab['char-index'][self.conf['UNK']]) for c in token]
                indices.extend([self.vocab['char-index'][self.conf['PAD']]] * (max_word_length - len(indices)))
                char_indices += [indices]

            word_indices.extend([self.vocab['word-index'][self.conf['PAD']]] * (max_sentence_length - len(item)))
            char_indices.extend([[self.vocab['char-index'][self.conf['PAD']]] * max_word_length] * (max_sentence_length - len(item)))

            new_tokens.append(word_indices)
            new_chars.append(char_indices)

        # pad tags
        new_tags = []
        for item in tags:
            tag_indices = []
            for tag in item:
                ordered_tag = sorted(tag, key=lambda g: self.sorting_order.get(g, len(self.sorting_order)))
                indices = [self.vocab['grammeme-index'].get(g, self.vocab['grammeme-index'][self.conf['UNK']]) for g in ordered_tag]
                indices.insert(0, self.vocab['grammeme-index'][self.conf['SOS']])
                indices.append(self.vocab['grammeme-index'][self.conf['EOS']])
                indices.extend([self.vocab['grammeme-index'][self.conf['PAD']]] * (max_label_length - len(indices)))
                tag_indices += [indices]
            tag_indices.extend([[self.vocab['grammeme-index'][self.conf['PAD']]] * max_label_length] * (max_sentence_length - len(item)))
            new_tags.append(tag_indices)


        return new_tokens, new_chars, new_tags


    def length(self, what='grammeme'):
        return len(self.vocab[f'{what}-index'])


def get_dictionaries(data):
    """
    Create two dictionaries from list:
    first with element:index pairs, second with index:element pairs.

    Args:
        data (list): List with elements to be turned into dictionaries.

    Returns:
        tuple: Dictionaries with element->index and index->element pairs.
    """

    stoi = {element: index for index, element in enumerate(data)}
    itos = {index: element for index, element in enumerate(data)}
    return stoi, itos


