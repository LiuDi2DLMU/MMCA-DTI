# -*- coding: utf-8 -*-
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from module import Encode, MoleculeTransformer, GatedCoAttention


class MMCADTI(nn.Module):
    """
    GatedCoAttentionwithSigmoid
    """

    def __init__(self,
                 dim=64,
                 conv=32,
                 dropout=0.2,
                 return_one=True,
                 n_layer=3,
                 protein_kernel_size=None,
                 k=2):
        super(MMCADTI, self).__init__()
        self.n_layer = n_layer
        if protein_kernel_size is None:
            protein_kernel_size = [5, 9, 13]
        self.dim = dim
        if return_one:
            self.return_dim = 1
        else:
            self.return_dim = 2
        self.conv = conv
        self.k = k

        self.proteinEmbed = nn.Embedding(22, self.dim, padding_idx=0)

        self.atomLinear = nn.Sequential(
            nn.Linear(28, self.dim),
            nn.ReLU()
        )
        self.atomTrans = MoleculeTransformer(8, [50, self.dim])

        self.bitattn = GatedCoAttention3(self.dim, self.dim)

        self.proteinEncode = Encode(n_layer, self.conv, [1000, self.dim], protein_kernel_size, activate=nn.ReLU,
                                    k=self.k)

        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.conv * n_layer * 1 * k + self.dim * 2 * k + 1024, 512),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, self.return_dim)
        )

    def __convert_graph__(self, graph, MAX=50):
        graphs = dgl.unbatch(graph)
        adjs = []
        atoms = []
        for graph in graphs:
            adj = graph.adjacency_matrix().to_dense()
            adj = torch.split(torch.split(adj, 50, dim=-2)[0], 50, dim=-1)[0]
            adj = F.pad(adj, (0, MAX - adj.shape[1], 0, MAX - adj.shape[0]))
            adjs.append(adj)
            atom = graph.ndata['x']
            atom = torch.split(atom, 50, dim=-2)[0]
            atom = F.pad(atom, (0, 0, 0, MAX - atom.shape[0]))
            atoms.append(atom)
        return torch.stack(adjs).to(torch.float32), torch.stack(atoms).to(torch.float32)

    def forward(self, drug, protein):
        coulombMatrix = drug[0].to(torch.float32)
        adjs, atoms = self.__convert_graph__(drug[1])
        morgan = drug[3].to(torch.float32)

        proteinembed = self.proteinEmbed(protein)

        atoms = self.atomLinear(atoms)
        atoms = self.atomTrans(atoms, adjs.cuda(), coulombMatrix)
        protein_b, b = self.proteinEncode(proteinembed)

        drug, proteinembed = self.bitattn(atoms, proteinembed)
        c, d = drug.topk(k=self.k, dim=1)[0].flatten(1), proteinembed.topk(k=self.k, dim=1)[0].flatten(1)

        state = self.flatten(torch.cat([b, c, d], dim=-1))
        state = torch.cat([state, morgan], dim=-1)
        state = self.classifier(state)
        return state
