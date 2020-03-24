import torch
import torch.nn as nn
from .utils import _calc_padding, _unpack_from_convolution, _pack_for_convolution


class GraphAndConv(nn.Module):
    def __init__(self, input_dim, output_dim, conv_kernel_size, intermediate_dim=None):
        super(GraphAndConv, self).__init__()
        if intermediate_dim is None:
            intermediate_dim = output_dim
        self.lin = nn.Linear(2*input_dim, intermediate_dim)
        padding = _calc_padding(1, conv_kernel_size)
        self.conv = nn.Conv1d(intermediate_dim, output_dim, conv_kernel_size, padding=padding)

        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim

    def forward(self, adj, inputs):
        batch_size = inputs.shape[0]
        mask = adj.sum(dim=2).bool()
        x = torch.einsum('bilc,bij->bjlc', inputs, adj)
        x = torch.cat((x, inputs), dim=-1)
        x = self.lin(x)
        x = _pack_for_convolution(x)
        x = self.conv(x)
        x = _unpack_from_convolution(x, batch_size)
        x[~mask] = 0.
        return x


class SiameseMLP(nn.Module):
    def __init__(self, input_size, emb_dim):
        """ Initialize model parameters for Siamese network.

        Parameters
        ----------
        input_size: int
            Input dimension size
        emb_dim: int
            Embedding dimension for both datasets

        Note
        ----
        This implicitly assumes that the embedding dimension for
        both datasets are the same.
        """
        # See here: https://adoni.github.io/2017/11/08/word2vec-pytorch/
        super(SiameseMLP, self).__init__()
        self.input_size = input_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Linear(input_size, emb_dim)
        self.v_embeddings = nn.Linear(input_size, emb_dim)
        self.init_emb()

    def init_emb(self):
        initstd = 1 / math.sqrt(self.emb_dimension)
        self.u_embeddings.weight.data.normal_(0, initstd)
        self.v_embeddings.weight.data.normal_(0, initstd)

    def forward(self, pos_u, pos_v, neg_v):
        """
        Parameters
        ----------
        pos_u : torch.Tensor
           Protein representation vector
        pos_v : torch.Tensor
           Positive molecular representation vector
        neg_v : torch.Tensor
           Negative molecular representation vector(s).
           There can be multiple negative examples (~5 according to NCE).
        """

        losses = 0
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        score = torch.mul(emb_u, emb_v).squeeze()
        score = torch.sum(score, -1)
        score = F.logsigmoid(score)
        if score.dim() >= 1:
            losses += sum(score)
        else:
            losses += score
        neg_emb_v = self.v_embeddings(neg_v)
        neg_score = torch.bmm(neg_emb_v.unsqueeze(1),
                              emb_u.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-1 * neg_score)
        if neg_score.dim() >= 1:
            losses += sum(neg_score)
        else:
            losses += neg_score
        return -1 * losses

    def predict(self, x1, x2):
        emb_u = self.u_embeddings(x1)
        emb_v = self.v_embeddings(x2)
        score = torch.mul(emb_u, emb_v).squeeze()
        score = F.logsigmoid(torch.sum(score, -1))
        return score

