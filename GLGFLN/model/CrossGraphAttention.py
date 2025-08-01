import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class CrossGraphAttention(nn.Module):
    def __init__(self, in_features, out_features):
        super(CrossGraphAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 权重矩阵用于计算 Query, Key, Value
        self.query_weight_g1 = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.key_weight_g1 = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.value_weight_g1 = nn.Parameter(torch.FloatTensor(in_features, out_features))

        self.query_weight_g2 = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.key_weight_g2 = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.value_weight_g2 = nn.Parameter(torch.FloatTensor(in_features, out_features))

        # 初始化权重
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.xavier_uniform_(self.query_weight_g1)
        # nn.init.xavier_uniform_(self.key_weight_g1)
        # nn.init.xavier_uniform_(self.value_weight_g1)
        # nn.init.xavier_uniform_(self.query_weight_g2)
        # nn.init.xavier_uniform_(self.key_weight_g2)
        # nn.init.xavier_uniform_(self.value_weight_g2)
        stdv = 1. / math.sqrt(self.query_weight_g1.size(1))
        self.query_weight_g1.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.key_weight_g1.size(1))
        self.key_weight_g1.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.value_weight_g1.size(1))
        self.value_weight_g1.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.query_weight_g2.size(1))
        self.query_weight_g2.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.key_weight_g2.size(1))
        self.key_weight_g2.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.value_weight_g2.size(1))
        self.value_weight_g2.data.uniform_(-stdv, stdv)


    def forward(self, x_g1, adj_g1, x_g2, adj_g2):
        # 计算 Q, K, V
        Q_g1 = torch.mm(x_g1, self.query_weight_g1)
        K_g1 = torch.mm(x_g1, self.key_weight_g1)
        V_g1 = torch.mm(x_g1, self.value_weight_g1)

        Q_g2 = torch.mm(x_g2, self.query_weight_g2)
        K_g2 = torch.mm(x_g2, self.key_weight_g2)
        V_g2 = torch.mm(x_g2, self.value_weight_g2)

        # 计算交叉注意力
        attention_scores_g1_to_g2 = torch.mm(Q_g1, K_g2.t())
        attention_scores_g1_to_g2 = F.softmax(attention_scores_g1_to_g2, dim=-1)
        attended_features_g1 = torch.mm(attention_scores_g1_to_g2, V_g2)

        attention_scores_g2_to_g1 = torch.mm(Q_g2, K_g1.t())
        attention_scores_g2_to_g1 = F.softmax(attention_scores_g2_to_g1, dim=-1)
        attended_features_g2 = torch.mm(attention_scores_g2_to_g1, V_g1)

        # 将两个图的特征融合
        output_g1 = torch.cat([attended_features_g1, x_g1], dim=-1)
        output_g2 = torch.cat([attended_features_g2, x_g2], dim=-1)
        #output_g1 = torch.sigmoid(attended_features_g1+x_g1)
        #output_g2 = torch.sigmoid(attended_features_g2 + x_g2)


        return output_g1, output_g2

# 示例用法
if __name__ == "__main__":
    # 示例参数
    num_nodes_g1 = 10
    num_nodes_g2 = 8
    in_features = 16
    out_features = 8

    # 随机生成输入数据
    x_g1 = torch.randn(num_nodes_g1, in_features)
    adj_g1 = torch.randn(num_nodes_g1, num_nodes_g1)
    x_g2 = torch.randn(num_nodes_g2, in_features)
    adj_g2 = torch.randn(num_nodes_g2, num_nodes_g2)

    # 初始化模型
    model = CrossGraphAttention(in_features, out_features)

    # 前向传播
    output_g1, output_g2 = model(x_g1, adj_g1, x_g2, adj_g2)

    print("Output for G1:", output_g1)
    print("Output for G2:", output_g2)
