import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LSTM(nn.Module):
    
    def __init__(self, word_embedding_dim, hidden_dim, vocab_size, tag_vocab_size, tag_embedding_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim)
        self.tag_embeddings = nn.Embedding(tag_vocab_size, tag_embedding_dim)
        self.lstm = nn.LSTM(word_embedding_dim + tag_embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)

        self.hidden2dep = nn.Linear(hidden_dim, vocab_size)

        self.hidden_to_relu_dep = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_to_relu_head = nn.Linear(hidden_dim, hidden_dim)

        self.bi_affine = nn.Linear(hidden_dim, hidden_dim)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(2, 1, self.hidden_dim // 2)),
                autograd.Variable(torch.zeros(2, 1, self.hidden_dim // 2)))

    def forward(self, sentence, tags):
        word_embeds = self.word_embeddings(sentence)
        tag_embeds = self.tag_embeddings(tags)
        embeds = torch.cat((word_embeds, tag_embeds))
        size = len(sentence) + len(tags)
        print(len(sentence), len(tags))
        lstm_out, self.hidden = self.lstm(embeds.view(size , 1, -1), self.hidden)
        dep_space = self.hidden2dep(lstm_out.view(size, -1))
        H_dep = F.relu(self.hidden_to_relu_dep(lstm_out.view(size, -1)))
        H_head = F.relu(self.hidden_to_relu_head(lstm_out.view(size, -1)))
        scores = F.log_softmax(self.bi_affine(H_head).mm(H_dep.transpose(0, 1)), dim=1)  # have to check dimension of softmax!
        return scores
