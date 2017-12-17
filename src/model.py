import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import MLP
from mst_daan import mst

class DependencyParser(nn.Module):
    
    def __init__(self, word_embedding_dim, hidden_dim, vocab_size, tag_vocab_size, tag_embedding_dim,
                label_amount, dropout, char_emb_dim, char_vocab_size):
        super(DependencyParser, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim)
        self.tag_embeddings = nn.Embedding(tag_vocab_size, tag_embedding_dim)

        self.char_embeddings = nn.Embedding(char_vocab_size, char_emb_dim)
        self.char_attention = nn.Linear(char_emb_dim, 1)
        self.lstm = nn.LSTM(word_embedding_dim + tag_embedding_dim , hidden_dim ,
                            num_layers=1, bidirectional=True, dropout=dropout)

        self.char_lstm = nn.LSTM(char_emb_dim, 20, num_layers=1, dropout=dropout)
        self.hidden_to_relu_dep = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_to_relu_head = nn.Linear(hidden_dim, hidden_dim)

        self.arc_dep = MLP(hidden_dim*2, hidden_dim, 1,  dropout)
        self.arc_head = MLP(hidden_dim*2, hidden_dim, 1, dropout)

        self.label_dep = MLP(hidden_dim*2, hidden_dim, 1,  dropout)
        self.label_head = MLP(hidden_dim*2, hidden_dim, 1,  dropout)

        # add 1 for bias
        self.bi_affine_arcs = nn.Linear(hidden_dim+1, hidden_dim, bias=False)
        self.bi_affine_labels_weights = nn.Parameter(torch.Tensor(label_amount, hidden_dim+1, hidden_dim+1))
        self.bi_affine_labels_weights.data.normal_(0, 1)

    def forward(self, sentence, tags, charachters, best_arcs=None):
        word_embeds = self.word_embeddings(sentence)
        tag_embeds = self.tag_embeddings(tags)
        size = len(sentence)

        def warp_word(word):
            vec = autograd.Variable(torch.LongTensor(word))
            embeds = self.char_embeddings(vec)
            lstm_out, _ = self.char_lstm(embeds.view(len(word), 1, -1))
            lstm_out = lstm_out.squeeze(dim=1)
            return self.char_attention(lstm_out.transpose(0, 1) @ lstm_out)

        char_embeds = torch.cat([warp_word(word) for word in charachters], dim=1)
        embeds = self.dropout(torch.cat((word_embeds, tag_embeds), dim=1))
        lstm_out, _ = self.lstm(embeds.view(size, 1, -1))
        arc_dep = self.arc_dep(lstm_out).view(size, -1)
        arc_head = self.arc_head(lstm_out).view(size, -1)
        label_dep = self.label_dep(lstm_out).view(size, -1)
        label_dep = torch.cat((label_dep, autograd.Variable(torch.ones(size, 1))), -1)
        label_head = self.label_head(lstm_out).view(size, -1) 
        label_head = torch.cat((label_head, autograd.Variable(torch.ones(size, 1))), -1)
        arc_head = torch.cat((arc_head, autograd.Variable(torch.ones(size, 1))), -1)
        arc_scores = F.softmax(self.bi_affine_arcs(arc_head).mm(arc_dep.transpose(0, 1)), dim=1)  

        if best_arcs == None:
            if len(arc_scores.data.numpy().T) == 1:
                best_arcs = [0]
            else:     
                best_arcs = mst(arc_scores.data.numpy())

            label_scores = label_head @ self.bi_affine_labels_weights @ label_dep.transpose(0, 1)
            label_scores = F.log_softmax(label_scores[:, best_arcs, [i for i in range(len(best_arcs))]], dim=0)
            return best_arcs, label_scores
            
        label_scores = label_head @ self.bi_affine_labels_weights @ label_dep.transpose(0, 1)
        label_scores = F.log_softmax(label_scores[:, best_arcs, [i for i in range(len(best_arcs))]], dim=0)

        return arc_scores, label_scores
