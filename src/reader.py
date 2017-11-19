class Reader(object):
    
    def get_vocabulary(self, file_path):
        with open(file_path) as f:
            word_to_index = {'root': 0}
            for line in [line for line in f.read().split('\n\n')]:
                if line == "":
                    continue
                words = [x.split('\t')[2] for x in line.split('\n') if x[0] not in ['#','r\n','\n']]
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

    def aggregate_training_data(self, file_path, vocabulary, tag_vocabulary):
        with open(file_path) as f:
            X = []
            T = []
            for line in [line for line in f.read().split('\n\n')]:
                if line == "":
                    continue
                words = ['root'] + [x.split('\t')[2] for x in line.split('\n') if x[0] not in ['#','r\n','\n']]
                tags = ['ROOT'] + [x.split('\t')[4] for x in line.split('\n') if x[0] not in ['#','r\n','\n']]
                
                tag_idx = [tag_vocabulary[tag] for tag in tags]
                arcs_indices = [x.split('\t')[6] for x in line.split('\n') if x[0] not in ['#','r\n','\n']]
                print(len(arcs_indices), len(tag_idx))
                arcs = [(vocabulary[words[i]], int(j)) # j might be off 
                        for i, j in enumerate(arcs_indices, start=0) if j != '_'] 

                # print()
                idx, target = zip(*arcs)
                # print(len(idx), len(tag_idx))
                X.append({'word_idx': idx, 'tag_idx': tag_idx})
                T.append(target)
                # should check why j sometimes is '_'
        return X, T
