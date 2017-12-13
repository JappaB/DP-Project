from collections import Counter
from urllib.parse import urlparse
from email.utils import parseaddr

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
            label_to_index = {}
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

    def aggregate_training_data(self, file_path, cnt, vocabulary, tag_vocabulary, label_vocabulary):
        with open(file_path) as f:
            X = []
            T = []
            for line in [line for line in f.read().split('\n\n')]:
                if line == "":
                    continue

                words = ['root'] + [x.split('\t')[2] for x in line.split('\n') if x[0] not in ['#','r\n','\n']]

                # to lowercase
                words = [w.lower() for w in words]

                # set uris to tag uri
                words = ['<uri>' if uri_validator(w) else w for w in words]

                # set email to @
                words = ['<email>' if '@' in w else w for w in words]

                # set words that occur ones in the training data to unknown
                words = [w if cnt[w] != 1 else '<unk>' for w in words]
                
                tags = [x.split('\t')[4] for x in line.split('\n') if x[0] not in ['#','r\n','\n']]

                labels = [x.split('\t')[7].lower() for x in line.split('\n') if x[0] not in ['#','r\n','\n']]

                # possibly should add root.
                tag_idx = [tag_vocabulary[tag] for tag in tags]

                label_idx = [label_vocabulary[label] for label in labels]

                arcs_indices = [x.split('\t')[6] for x in line.split('\n') if x[0] not in ['#','r\n','\n']]

                print(arcs_indices)

                arcs = [(vocabulary[words[i]], int(j)) # j might be off 
                        for i, j in enumerate(arcs_indices, start=0) if j != '_'] 

                idx, target = zip(*arcs)
                if len(idx) != len(tags):
                    
                    # weird format, no arc
                    continue
                X.append({'word_idx': idx, 'tag_idx': tag_idx})
                T.append({'arc_target': target, 'label_target': label_idx})
                # should check why j sometimes is '_'
        return X, T

r = Reader()

counted_words = r.count_words('../Data/en-ud-train.conllu')
vocabulary = r.get_vocabulary('../Data/en-ud-train.conllu')
tag_vocabulary = r.get_tag_vocabulary('../Data/en-ud-train.conllu')

label_vocabulary = r.get_label_vocabulary('../Data/en-ud-train.conllu')
X, T = r.aggregate_training_data('../Data/en-ud-train.conllu', 
                                 counted_words, vocabulary, tag_vocabulary, 
                                 label_vocabulary)