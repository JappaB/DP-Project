from reader import Reader
from model import DependencyParser
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from random import shuffle


def train(X_train, T_train, X_test, T_test, model, loss_function, 
          optimizer, scheduler):
    train_loss = 0.0
    train_arc_acc = 0
    train_label_acc = 0
    i = 0
    skipped = 0
    for x, t in zip(X_train, T_train):
        if i > 100:
            continue

        if (len(t['arc_target']) in t['arc_target'] or len(t['label_target'])+1 
            in t['arc_target'] or len(t['label_target']) in t['label_target']):
            skipped += 1
            continue # this a bug from reading in the data, should check it ;)
        model.zero_grad()
        word_tensor = autograd.Variable(torch.LongTensor(x['word_idx']))
        tag_tensor = autograd.Variable(torch.LongTensor(x['tag_idx']))

        arc_tensor = autograd.Variable(torch.LongTensor(t['arc_target']))
        label_tensor = autograd.Variable(torch.LongTensor(t['label_target']))

        arc_scores, label_scores = model(word_tensor, tag_tensor, best_arcs=t['arc_target'])
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
    j = 0
    for x, t in zip(X_test, T_test):
        if j> 50:
            continue
        if (len(t['arc_target']) in t['arc_target'] or len(t['label_target'])+1 
            in t['arc_target'] or len(t['label_target']) in t['label_target']):
            skipped += 1
            continue
        word_tensor = autograd.Variable(torch.LongTensor(x['word_idx']))
        tag_tensor = autograd.Variable(torch.LongTensor(x['tag_idx']))

        arc_tensor = autograd.Variable(torch.LongTensor(t['arc_target']))
        label_tensor = autograd.Variable(torch.LongTensor(t['label_target']))

        arc_scores, label_scores = model(word_tensor, tag_tensor, best_arcs=None)
        test_arc_acc += (arc_scores.max(dim=1)[1] == arc_tensor).sum().data[0] / len(arc_tensor)
        test_label_acc += (label_scores.max(dim=0)[1] == label_tensor).sum().data[0] / len(label_tensor)
        j += 1
    print('train loss: ', train_loss / i,  # 'test_loss :', test_loss / j,
          'train arc acc:', train_arc_acc / i, 'train label acc: ', train_label_acc / i,
          'test arc acc: ', test_arc_acc / j, 'test label acc: ', test_label_acc / j, i, j, skipped)
    scheduler.step(test_label_acc / len(X_test))
    return test_arc_acc / j

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
    vocabulary = r.get_vocabulary(filename)
    tag_vocabulary = r.get_tag_vocabulary(filename)
    label_vocabulary = r.get_label_vocabulary(filename)
    model = DependencyParser(100, 100, len(vocabulary), len(tag_vocabulary), 10, len(label_vocabulary), 0.3)  # bad performance :()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=[0.9, 0.9])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=3)
    counted_words = r.count_words(filename)
    X, T = r.aggregate_training_data(filename, 
                                     counted_words, vocabulary, tag_vocabulary, 
                                     label_vocabulary)

   
    X_train, X_test, T_train, T_test = split(X, T)
    
    for i in range(epochs):
        print('iteration: {0}'.format(i+1))
        test_loss = train(X_train, T_train, X_test, T_test, model, loss_function, optimizer, scheduler)

    with open('model_{}.model'.format(test_loss), 'wb') as f:
        torch.save(model, f)

full_training(10, '../Data/en-ud-train.conllu')
