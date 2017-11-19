from reader import Reader
from model import LSTM
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

def train(model, X, T):
    i = 1
    train_loss = 0.0
    for x, t in zip(X, T):
        if len(t) in t:
            continue # this a bug from reading in the data, should check it ;)
        model.zero_grad()
        model.hidden = model.init_hidden()
        word_tensor = autograd.Variable(torch.LongTensor(x['word_idx']))
        tag_tensor =  autograd.Variable(torch.LongTensor(x['tag_idx']))
        t_tensor = autograd.Variable(torch.LongTensor(t))
        
        scores = model(word_tensor, tag_tensor)
        loss = loss_function(scores, t_tensor)
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]
        i += 1
    print('loss!', train_loss/i)

def evaluate(X, T):
    i = 1
    train_loss = 0.0
    for x, t in zip(X, T):
        if len(t) in t:
            continue
        x_tensor = autograd.Variable(torch.LongTensor(x))
        t_tensor = autograd.Variable(torch.LongTensor(t))
        scores = self(x_tensor)
        loss = loss_function(scores, t_tensor)
        train_loss += loss.data[0]
        i += 1
    return (train_loss/i)


r = Reader()
vocabulary = r.get_vocabulary('../Data/en-ud-train.conllu')
tag_vocabulary = r.get_tag_vocabulary('../Data/en-ud-train.conllu')

X, T = r.aggregate_training_data('../Data/en-ud-train.conllu', vocabulary, tag_vocabulary)

model = LSTM(10, 10, len(vocabulary), len(tag_vocabulary), 10)  # bad performance :()
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for _ in range(100):
    train(model, X, T)
