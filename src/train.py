from reader import Reader
from model import DependencyParser
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from random import shuffle

def train(model, X_train, T_train, X_test, T_test):
    train_loss = 0.0
    train_acc = 0
    i = 0
    for x, t in zip(X_train, T_train):
        if len(t) in t:
            continue # this a bug from reading in the data, should check it ;)
        model.zero_grad()
        word_tensor = autograd.Variable(torch.LongTensor(x['word_idx']))
        tag_tensor = autograd.Variable(torch.LongTensor(x['tag_idx']))
        t_tensor = autograd.Variable(torch.LongTensor(t))
        scores = model(word_tensor, tag_tensor)
        train_acc += (scores.max(dim=1)[1] == t_tensor).sum().data[0] / len(t_tensor)
        loss = loss_function(scores, t_tensor)
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]
        i += 1
    
    test_loss = 0
    test_acc = 0
    j = 0
    for x, t in zip(X_test, T_test):
        if len(t) in t:
            continue
        word_tensor = autograd.Variable(torch.LongTensor(x['word_idx']))
        tag_tensor = autograd.Variable(torch.LongTensor(x['tag_idx']))
        t_tensor = autograd.Variable(torch.LongTensor(t))
        scores = model(word_tensor, tag_tensor)
        test_acc += (scores.max(dim=1)[1] == t_tensor).sum().data[0] / len(t_tensor)
        loss = loss_function(scores, t_tensor)
        test_loss += loss.data[0]
        j += 1
    print('train loss: ', train_loss / i, 'test_loss :', test_loss / j,
          'train acc:', train_acc / i, 'test acc:', test_acc / j)
    scheduler.step(test_loss / len(X_test))
    return test_acc / j

def split(X, T):
    split_index = round(len(X) / 5)
    indices = [i for i in range(len(X))]
    shuffle(indices)
    X_train = [X[i] for i in indices[split_index:]]
    X_test = [X[i] for i in indices[:split_index]]
    T_train = [T[i] for i in indices[split_index:]]
    T_test = [T[i] for i in indices[:split_index]]
    return X_train, X_test, T_train, T_test

r = Reader()
counted_words = r.count_words('../Data/en-ud-train.conllu')
vocabulary = r.get_vocabulary('../Data/en-ud-train.conllu')
tag_vocabulary = r.get_tag_vocabulary('../Data/en-ud-train.conllu')
X, T = r.aggregate_training_data('../Data/en-ud-train.conllu', 
                                 counted_words, vocabulary, tag_vocabulary)

model = DependencyParser(100, 100, len(vocabulary), len(tag_vocabulary), 10, 0.3)  # bad performance :()
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002, betas=[0.9, 0.9])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=3)
X_train, X_test, T_train, T_test = split(X, T)

for i in range(100):
    print('iteration: {0}'.format(i+1))
    test_loss = train(model, X, T, X_test, T_test)

with open('model_{}.model'.format(test_loss), 'wb') as f:
    torch.save(model, f)
