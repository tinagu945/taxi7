import torch.nn as nn


class StateEmbeddingModel(nn.Module):
    def __init__(self, num_s, num_out):
        super(StateEmbeddingModel, self).__init__()
        self.embeddings = nn.Embedding(num_s, num_out)

    def forward(self, inputs):
        return self.embeddings(inputs)


class QTableModel(StateEmbeddingModel):
    def __init__(self, q_table):
        num_s, num_a = q_table.shape
        super(QTableModel, self).__init__(num_s=num_s, num_out=num_a)
        self.embeddings.weight.data = q_table
