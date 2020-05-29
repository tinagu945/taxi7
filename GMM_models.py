import torch.nn as nn

class StateEmbedding(nn.Module):
    def __init__(self, num_state=2000, embedding_dim=1):
        super(StateEmbedding, self).__init__()
        self.embeddings = nn.Embedding(num_state, embedding_dim)

    def forward(self, inputs):
        return self.embeddings(inputs)


class StateEmbeddingAdversary(nn.Module):
    def __init__(self, num_state=2000, embedding_dim=1):
        super(StateEmbeddingAdversary, self).__init__()
        self.embeddings = nn.Embedding(num_state, embedding_dim)
        self.c = nn.Parameter(torch.ones(2))

    def forward(self, inputs):
        return self.embeddings(inputs)

    def get_coef(self):
        return self.c

    
class simple_CNN(nn.Module):
    def __init__(self, num_state=2000, embedding_dim=1):