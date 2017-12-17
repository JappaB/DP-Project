from reader import Reader
from model import DependencyParser
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
from random import shuffle
import json


def train(X_train, X_test, T_train, T_test, model, loss_function, optimizer, scheduler):
    train_loss = 0.0
    train_arc_acc = 0
    train_label_acc = 0
    i = 0
    skipped = 0
    mistakes = []
    for x, t in zip(X_train, T_train):
        if (len(t['arc_target']) in t['arc_target'] or len(t['label_target'])+1 
            in t['arc_target'] or len(t['label_target']) in t['label_target']):
            skipped += 1
            continue 
        model.zero_grad()
        word_tensor = autograd.Variable(torch.LongTensor(x['word_idx']))
        tag_tensor = autograd.Variable(torch.LongTensor(x['tag_idx']))

        arc_tensor = autograd.Variable(torch.LongTensor(t['arc_target']))
        label_tensor = autograd.Variable(torch.LongTensor(t['label_target']))

        arc_scores, label_scores = model(word_tensor, tag_tensor, x['chars'], best_arcs=t['arc_target'])
        train_arc_acc += (arc_scores.max(dim=1)[1] == arc_tensor).sum().data[0] / len(arc_tensor)
        train_label_acc += (label_scores.max(dim=0)[1] == label_tensor).sum().data[0] / len(label_tensor)
        loss = loss_function(arc_scores, arc_tensor)  + loss_function(label_scores.transpose(0, 1), label_tensor)

        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]
        i += 1
    
    test_loss = 0
    test_arc_acc = 0
    test_label_acc = 0
    las_acc = 0
    uas_acc = 0
    j = 0
    for x, t in zip(X_test, T_test):
        if (len(t['arc_target']) in t['arc_target'] or len(t['label_target'])+1 
            in t['arc_target'] or len(t['label_target']) in t['label_target']):
            skipped += 1
            continue
        
        word_tensor = autograd.Variable(torch.LongTensor(x['word_idx']))
        tag_tensor = autograd.Variable(torch.LongTensor(x['tag_idx']))

        arc_tensor = torch.LongTensor(t['arc_target'])
        label_tensor = autograd.Variable(torch.LongTensor(t['label_target']))

        best_arcs, label_scores = model(word_tensor, tag_tensor,x['chars'], best_arcs=None)
        # print(best_arcs.max(dim=1)[1].data == arc_tensor)
        # test_arc_acc += (best_arcs.max(dim=1)[1].data == arc_tensor).sum() / len(arc_tensor)
        best_arcs = torch.LongTensor(best_arcs)
        test_arc_acc += (best_arcs == arc_tensor).sum() / len(arc_tensor)
        
        # correct_arc_idx = best_arcs.max(dim=1)[1].data == arc_tensor
        correct_arc_idx = best_arcs == arc_tensor
        # las_acc += best_arcs.max(dim=1)[1] == arc_tensor
        c = (label_scores.max(dim=0)[1] == label_tensor)
        las = c[correct_arc_idx].sum().data[0] / len(arc_tensor)
        uas = correct_arc_idx.sum() / len(arc_tensor)

        print(las, uas, '@@@@@@@')
        las_acc += las
        uas_acc += uas
        test_label_acc += (label_scores.max(dim=0)[1] == label_tensor).sum().data[0] / len(label_tensor)
        # mistakes.append({
        #     'sentence': list(word_tensor.data), 'acc': uas,
        #     'scores': best_arcs.data.numpy().tolist()
        # })
        j += 1
    print('train loss: ', train_loss / i,  # 'test_loss :', test_loss / j,
          'train arc acc:', train_arc_acc / i, 'train label acc: ', train_label_acc / i,
          'test las acc: ', las_acc / j, 'test uas acc: ', uas_acc / j)
    scheduler.step(test_label_acc / len(X_test))
    return test_arc_acc / j, sorted(mistakes, key=lambda x: x['acc'])

def split(X, T):
    split_index = round(len(X) / 5)
    indices = [i for i in range(len(X))]
    shuffle(indices)
    X_train = [X[i] for i in indices[split_index:]]
    X_test = [X[i] for i in indices[:split_index]]
    T_train = [T[i] for i in indices[split_index:]]
    T_test = [T[i] for i in indices[:split_index]]
    return X_train, X_test, T_train, T_test

def full_training(epochs, filename):
    r = Reader()

    counted_words = r.count_words(filename)
    vocabulary = r.get_vocabulary(filename)
    tag_vocabulary = r.get_tag_vocabulary(filename)

    label_vocabulary = r.get_label_vocabulary(filename)
    char_vocabulary = r.get_char_vocab(vocabulary)
    X, T = r.aggregate_training_data(filename, 
                                    counted_words, vocabulary, tag_vocabulary, 
                                    label_vocabulary, char_vocabulary)
    model = DependencyParser(100, 100, len(vocabulary), len(tag_vocabulary), 25, len(label_vocabulary), 0.3, 20, len(char_vocabulary))  # bad performance :()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=[0.9, 0.9], weight_decay=1e-7)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=3)
    counted_words = r.count_words(filename)
    X, T = r.aggregate_training_data(filename, 
                                     counted_words, vocabulary, tag_vocabulary, 
                                     label_vocabulary, char_vocabulary)
    print('fffff', len(X))
    X_train, X_test, T_train, T_test = split(X, T)

    reverse_vocab = {value: key for key, value in vocabulary.items()}
    for i in range(epochs):
        print('iteration: {0}'.format(i+1))
        test_loss, mistakes = train(X_train, X_test, T_train, T_test, model, loss_function, optimizer, scheduler)
        for mistake in mistakes[:10]:
            print(mistake)
            print('mistake: {0} sentence: {1}'.format(mistake['acc'], str([
                reverse_vocab[i] for i in mistake['sentence']
            ])))
    # with open('mistakes.json'.format(filename), 'w') as f:
    #     json.dump({
    #         'bad': [{'scores': x['scores'], 'sentence': [reverse_vocab[i] for i in x['sentence']]} for x in mistakes if  len(x['sentence']) > 8][:3],
    #         'good': [{'scores': x['scores'], 'sentence': [reverse_vocab[i] for i in x['sentence']]} for x in mistakes[-3:] if  len(x['sentence']) > 8][:3]
    #     }, f)

    with open('model_eng.model'.format(test_loss), 'wb') as f:
        torch.save(model, f)

full_training(10, '../Data/ar-ud-train.conllu.txt')