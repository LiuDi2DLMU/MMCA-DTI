# -*- coding: utf-8 -*-
import torch
from torch import nn


class Encode(nn.Module):
    def __init__(self, n_layer, conv, input_shape, kernel_size: list, activate=nn.ReLU, k=1):
        super(Encode, self).__init__()
        self.n_layer = n_layer
        self.conv = conv
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.k = k

        self.relu = activate()

        self.conv_layer = nn.ModuleList()

        self.res_layer = nn.ModuleList()

        self.__init_layer__()

    def __init_layer__(self):
        for i in range(self.n_layer):
            if i == 0:
                self.conv_layer.append(
                    nn.Conv1d(in_channels=self.input_shape[1],
                              out_channels=self.conv,
                              kernel_size=self.kernel_size[i],
                              padding=self.kernel_size[i] // 2))
                self.res_layer.append(None)
            else:
                self.conv_layer.append(
                    nn.Conv1d(in_channels=self.conv * i,
                              out_channels=self.conv * (i + 1),
                              kernel_size=self.kernel_size[i],
                              padding=self.kernel_size[i] // 2))
                self.res_layer.append(
                    nn.Conv1d(in_channels=self.input_shape[1],
                              out_channels=self.conv * (i + 1),
                              kernel_size=1))

    def forward(self, state):
        state = torch.transpose(state, 1, 2)
        copy_state = state
        for i, (layer, res) in enumerate(zip(self.conv_layer, self.res_layer)):
            if i == 0:
                state = self.relu(layer(state))
            else:
                state = self.relu(layer(state) + res(copy_state))
        state = state.transpose(1, 2)
        return state, state.topk(k=self.k, dim=1)[0].flatten(1)


class MoleculeMultiHeadSelfAttention(nn.Module):
    def __init__(self, num_heads, dim, N=50):
        super().__init__()
        # Q, K, V 转换矩阵，这里假设输入和输出的特征维度相同
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        self.a = nn.Linear(N, N * num_heads)
        self.m = nn.Linear(N, N * num_heads)
        self.num_heads = num_heads

        self.lambdas = torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0]] * num_heads).requires_grad_())

        self.norm = nn.Sigmoid()

    def forward(self, x, adj, matrix):
        B, N, C = x.shape
        # 生成转换矩阵并分多头
        q = self.q(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.k(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        # 点积得到attention score
        attn = q @ k.transpose(2, 3) * (x.shape[-1] ** -0.5)
        attn = attn.softmax(dim=-1).transpose(0, 1)

        adj = self.a(adj).reshape(B, N, self.num_heads, N).permute(0, 2, 1, 3).softmax(dim=-1).transpose(0, 1)
        matrix = self.m(matrix).reshape(B, N, self.num_heads, N).permute(0, 2, 1, 3).softmax(dim=-1).transpose(0, 1)

        lambdas = torch.exp(self.lambdas) / torch.sum(torch.exp(self.lambdas), dim=-1, keepdim=True).repeat(1, 3)
        attn_sum = []
        for i in range(self.num_heads):
            lambdas_i = lambdas[i]
            # x = torch.mul(attn[i], lambdas_i[0]) + torch.mul(adj, lambdas_i[1]) + torch.mul(
            #     self.norm(matrix), lambdas_i[2])
            x = torch.mul(attn[i], lambdas_i[0]) + torch.mul(adj[i], lambdas_i[1]) + torch.mul(matrix[i], lambdas_i[2])
            attn_sum.append(x)
        attn_sum = torch.stack(attn_sum, dim=0).transpose(0, 1)

        # 乘上attention score并输出
        v = (attn_sum @ v).permute(0, 2, 1, 3).reshape(B, N, C)
        return v


class MoleculeTransformer(nn.Module):
    def __init__(self, heads, inputShape):
        super(MoleculeTransformer, self).__init__()
        self.selfAttention = MoleculeMultiHeadSelfAttention(heads, inputShape[-1])
        self.norm = nn.LayerNorm(inputShape)
        self.norm1 = nn.LayerNorm(inputShape)
        self.linear = nn.Linear(inputShape[-1], inputShape[-1])

    def forward(self, x, adj, colomb):
        state = self.selfAttention(x, adj, colomb)
        state = self.norm(x + state)
        state = self.norm1(self.linear(state) + state)
        return state


class GatedCoAttention(nn.Module):
    """
    func保持为sigmoid
    """

    def __init__(self, d_dim, p_dim):
        super().__init__()
        self.d_dim = d_dim
        self.p_dim = p_dim

        self.softmax = nn.Sigmoid()
        self.Wb = nn.Linear(d_dim, p_dim)

        self.Wd = nn.Linear(p_dim, d_dim)
        self.Wp = nn.Linear(d_dim, p_dim)

        class Door(nn.Module):
            def __init__(self, dim):
                super(Door, self).__init__()
                self.sigmoid = nn.Sigmoid()
                self.w1 = nn.Linear(dim, dim)
                self.w2 = nn.Linear(dim, dim)
                self.tanh = nn.Tanh()
                self.w3 = nn.Linear(dim, dim)
                self.w4 = nn.Linear(dim, dim)

            def forward(self, x1, x2):
                f = self.sigmoid(self.w1(x1) + self.w2(x2))
                return f * self.tanh(self.w3(x1)) + (1 - f) * self.tanh(self.w4(x2))

        self.d_door = Door(d_dim)
        self.p_door = Door(p_dim)

    def forward(self, drug, protein):
        C = torch.bmm(self.Wb(drug), protein.permute(0, 2, 1))

        drug2protein = self.softmax(C * (self.p_dim ** -0.5))
        protein2drug = self.softmax(C.permute(0, 2, 1) * (self.d_dim ** -0.5))

        a = torch.bmm(drug2protein, self.Wd(protein))
        b = torch.bmm(protein2drug, self.Wp(drug))

        a = self.d_door(drug, a)
        b = self.p_door(protein, b)

        return a, b
