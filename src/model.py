import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import TimeDistributed
from utils import MLP

class DependencyParser(nn.Module):
    
    def __init__(self, word_embedding_dim, hidden_dim, vocab_size, tag_vocab_size, tag_embedding_dim,
                dropout):
        super(DependencyParser, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim)
        self.tag_embeddings = nn.Embedding(tag_vocab_size, tag_embedding_dim)
        self.lstm = nn.LSTM(word_embedding_dim + tag_embedding_dim, hidden_dim ,
                            num_layers=1, bidirectional=True, dropout=dropout)

        self.hidden_to_relu_dep = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_to_relu_head = nn.Linear(hidden_dim, hidden_dim)

        self.h_dep = TimeDistributed(MLP(hidden_dim*2, hidden_dim, 1,  0.2))
        self.h_head = TimeDistributed(MLP(hidden_dim*2, hidden_dim, 1, 0.2))

        # add 1 for bias
        self.bi_affine = nn.Linear(hidden_dim+1, hidden_dim)

    def forward(self, sentence, tags):
        word_embeds = self.word_embeddings(sentence)
        tag_embeds = self.tag_embeddings(tags)

        # print('size!', word_embeds.size(), tag_embeds.size())
        embeds = self.dropout(torch.cat((word_embeds, tag_embeds), dim=1))
        size = len(sentence)

        lstm_out, _ = self.lstm(embeds.view(size , 1, -1))

        H_dep = self.h_dep(lstm_out).view(size, -1)

        H_head = self.h_head(lstm_out).view(size, -1)

        H_head = torch.cat((H_head, autograd.Variable(torch.ones(size, 1))), -1)

        scores = F.log_softmax(self.bi_affine(H_head).mm(H_dep.transpose(0, 1)), dim=1)  # have to check dimension of softmax!
        return scores
