import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import TimeDistributed
from utils import MLP

class DependencyParser(nn.Module):
    
    def __init__(self, word_embedding_dim, hidden_dim, vocab_size, tag_vocab_size, tag_embedding_dim,
                label_amount, dropout):
        super(DependencyParser, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim)
        self.tag_embeddings = nn.Embedding(tag_vocab_size, tag_embedding_dim)
        self.lstm = nn.LSTM(word_embedding_dim + tag_embedding_dim, hidden_dim ,
                            num_layers=1, bidirectional=True, dropout=dropout)

        self.hidden_to_relu_dep = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_to_relu_head = nn.Linear(hidden_dim, hidden_dim)

        self.arc_dep = MLP(hidden_dim*2, hidden_dim, 1,  0.2)
        self.arc_head = MLP(hidden_dim*2, hidden_dim, 1, 0.2)

        self.label_dep = MLP(hidden_dim*2, hidden_dim, 1,  0.2)
        self.label_head = MLP(hidden_dim*2, hidden_dim, 1,  0.2)

        # add 1 for bias
        self.bi_affine_arcs = nn.Linear(hidden_dim+1, hidden_dim, bias=False)
        self.bi_affine_labels_weights = nn.Parameter(torch.Tensor(label_amount, hidden_dim+1, hidden_dim+1))
        self.bi_affine_labels_weights.data.normal_(0, 1)

    def forward(self, sentence, tags, best_arcs=None):
        word_embeds = self.word_embeddings(sentence)
        tag_embeds = self.tag_embeddings(tags)

        # print('size!', word_embeds.size(), tag_embeds.size())
        embeds = self.dropout(torch.cat((word_embeds, tag_embeds), dim=1))
        size = len(sentence)

        lstm_out, _ = self.lstm(embeds.view(size , 1, -1))

        arc_dep = self.arc_dep(lstm_out).view(size, -1)

        arc_head = self.arc_head(lstm_out).view(size, -1)

        label_dep = self.label_dep(lstm_out).view(size, -1)

        label_dep = torch.cat((label_dep, autograd.Variable(torch.ones(size, 1))), -1)

        label_head = self.label_head(lstm_out).view(size, -1) 

        label_head = torch.cat((label_head, autograd.Variable(torch.ones(size, 1))), -1)

        arc_head = torch.cat((arc_head, autograd.Variable(torch.ones(size, 1))), -1)

        arc_scores = F.log_softmax(self.bi_affine_arcs(arc_head).mm(arc_dep.transpose(0, 1)), dim=1)  # have to check dimension of softmax!

        # best arcs should be provided during training.
        if best_arcs == None:
            best_arcs = arc_scores.max(dim=1)[1] 
        label_scores = label_head @ self.bi_affine_labels_weights @ label_dep.transpose(0, 1)
        #print(label_head @ self.bi_affine_labels_weights)

        label_scores = F.log_softmax(label_scores[:, best_arcs, [i for i in range(len(best_arcs))]], dim=0)

        return arc_scores, label_scores
