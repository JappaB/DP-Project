from collections import Counter
from urllib.parse import urlparse
from email.utils import parseaddr
import itertools
import numpy as np

def email_validator(x):
    if parseaddr(x) == ('', ''):
        return True
    return False

def uri_validator(x):
    try:
        result = urlparse(x)
        return result.scheme and result.netloc and result.path
    except:
        return False

class Reader(object):
    
    def get_label_vocabulary(self, file_path):
        with open(file_path) as f:
            label_to_index = {'ROOT': 0}
            for line in [line for line in f.read().split('\n\n')]:
                if line == "":
                    continue
                labels = [x.split('\t')[7].lower() for x in line.split('\n') if x[0] not in ['#','r\n','\n']]
                # add words to vocabulary, possibly do pre-proccesing here
                for label in labels:
                    if label not in label_to_index:
                        label_to_index[label] = len(label_to_index)
        return label_to_index
    
    def count_words(self, file_path):
        cnt = Counter()
        with open(file_path) as f:
            for line in [line for line in f.read().split('\n\n')]:
                if line == "":
                    continue
                words = [x.split('\t')[2].lower() for x in line.split('\n') if x[0] not in ['#','r\n','\n']]
                for word in words:
                    cnt[word] += 1
        return cnt
    
    def get_vocabulary(self, file_path):
        with open(file_path) as f:
            word_to_index = {'root': 0, '<unk>': 1, '<uri>': 2, '<email>': 3}
            for line in [line for line in f.read().split('\n\n')]:
                if line == "":
                    continue
                words = [x.split('\t')[2].lower() for x in line.split('\n') if x[0] not in ['#','r\n','\n']]
                # add words to vocabulary, possibly do pre-proccesing here
                for word in words:
                    if word not in word_to_index:
                        word_to_index[word] = len(word_to_index)
        return word_to_index

    def get_tag_vocabulary(self, file_path):
        with open(file_path) as f:
            tag_to_index = {'ROOT': 0}
            for line in [line for line in f.read().split('\n\n')]:
                if line == "":
                    continue
                tags = [x.split('\t')[4] for x in line.split('\n') if x[0] not in ['#','r\n','\n']]
                for tag in tags:
                    if tag not in tag_to_index:
                        tag_to_index[tag] = len(tag_to_index)
        return tag_to_index
    
    def get_char_vocab(self, word_vocab):
        chars = list(set(itertools.chain(*[[c for c in w] for w in word_vocab.keys()])))
        vocab = {}
        for char in chars:
            if char not in vocab:
                vocab[char] = len(vocab)
        return vocab
        

    def aggregate_training_data(self, file_path, cnt, vocabulary, tag_vocabulary, label_vocabulary, char_vocab):
        with open(file_path) as f:
            X = []
            T = []
            for line in [line for line in f.read().split('\n\n')]:
                if line == "":
                    continue

                words =  ['root'] + [x.split('\t')[2] for x in line.split('\n') if x[0] not in ['#','r\n','\n']]

                # to lowercase
                words = [w.lower() for w in words]

                # set uris to tag uri
                words = ['<uri>' if uri_validator(w) else w for w in words]

                # set email to @
                words = ['<email>' if '@' in w else w for w in words]

                # set words that occur ones in the training data to unknown
                words = [w if cnt[w] != 1 else '<unk>' for w in words]

                chars = [[char_vocab[c] for c in w] for w in words]
                
                tags = ['ROOT'] + [x.split('\t')[4] for x in line.split('\n') if x[0] not in ['#','r\n','\n']]

                labels = ['ROOT'] + [x.split('\t')[7].lower() for x in line.split('\n') if x[0] not in ['#','r\n','\n']]

                # possibly should add root.
                tag_idx = [tag_vocabulary[tag] for tag in tags]

                label_idx = [label_vocabulary[label] for label in labels]

                arcs_indices = [0] + [x.split('\t')[6] for x in line.split('\n') if x[0] not in ['#','r\n','\n']]

                arcs = [(vocabulary[words[i]], int(j)) # j might be off 
                        for i, j in enumerate(arcs_indices, start=0) if j != '_'] 

                idx, target = zip(*arcs)
                if len(idx) != len(tags):             
                    # weird format, no arc
                    continue
                X.append({'word_idx': idx, 'tag_idx': tag_idx, 'chars': chars})
                T.append({'arc_target': target, 'label_target': label_idx})
                # should check why j sometimes is '_'
        return X, T


def average_sentence_len(X):
    return np.mean([len(x['word_idx']) for x in X])

r = Reader()
# counted_words = r.count_words('../Data/cs-ud-train-c.conllu.txt')
# vocabulary = r.get_vocabulary('../Data/cs-ud-train-c.conllu.txt')
# tag_vocabulary = r.get_tag_vocabulary('../Data/cs-ud-train-c.conllu.txt')

label_vocabulary = r.get_label_vocabulary('../Data/ar-ud-train.conllu.txt')
# char_vocabulary = r.get_char_vocab(vocabulary)
# X, T = r.aggregate_training_data('../Data/cs-ud-train-c.conllu.txt', 
#                                  counted_words, vocabulary, tag_vocabulary, 
#                                  label_vocabulary, char_vocabulary)
# print(len(X))
print(len(label_vocabulary))