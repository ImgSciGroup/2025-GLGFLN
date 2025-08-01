import torch
import torch.nn as nn

from .GraphConv import GraphConvolution
from .CrossGraphAttention import CrossGraphAttention
import torch.nn.functional as F


class GraphFeatureLearning_external(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GraphFeatureLearning_external, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, 2 * nhid)
        self.crat = CrossGraphAttention(2 *nhid,2*nhid)
        #self.gc3 = GraphConvolution(2 * nhid, nclass)

    def forward(self, x1, adj1,x2,adj2):
        x11 = torch.sigmoid(self.gc1(x1, adj1))
        x12= torch.sigmoid(self.gc2(x11, adj1))
        x21=torch.sigmoid(self.gc1(x2, adj2))
        x22 = torch.sigmoid(self.gc2(x21, adj2))
        feat1,feat2=self.crat(x12,adj1,x22,adj2)

        return feat1,feat2


class GraphFeatureLearning_external_1(nn.Module):#SG
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GraphFeatureLearning_external_1, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, 2*nhid)
        self.crat = CrossGraphAttention(2*nhid,2*nhid)
        self.gc3 = GraphConvolution( 4*nhid, nclass)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self, x1, adj1,x2,adj2):
        x1 = torch.sigmoid(self.gc1(x1, adj1))
        x1= torch.sigmoid(self.gc2(x1, adj1))
        x2 = self.dropout(x2)
        x1 = self.dropout(x1)
        x2=torch.sigmoid(self.gc1(x2, adj2))
        x2 = torch.sigmoid(self.gc2(x2, adj2))

        x1 = torch.sigmoid(self.gc2(x1, adj1))
        x2 = torch.sigmoid(self.gc2(x2, adj1))
        x2 = self.dropout(x2)
        x1 = self.dropout(x1)

        feat1,feat2=self.crat(x1,adj1,x2,adj2)
        #feat1, feat2 =x12,x22
        rec_feat1 = torch.sigmoid(self.gc3(feat1, adj1))
        rec_feat2 = torch.sigmoid(self.gc3(feat2, adj2))
        return rec_feat1,rec_feat2

class GraphFeatureLearning_external_2(nn.Module):#*******************************************************
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GraphFeatureLearning_external_2, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, 2*nhid)
        self.crat = CrossGraphAttention(nhid,nhid)


    def forward(self, x1, adj1,x2,adj2):
        x1_1 = torch.sigmoid(self.gc1(x1, adj1))
        #x1_1 = self.dropout(x1_1)
        x1_2 = torch.sigmoid(self.gc2(x1_1, adj1))
        #x1_2 = self.dropout(x1_2)
        x2_1 = torch.sigmoid(self.gc1(x2, adj2))
        x2_2 = torch.sigmoid(self.gc2(x2_1, adj2))
        #x2 = self.dropout(x2)
        #x1 = self.dropout(x1)
        #x1 = torch.sigmoid(self.gc2(x1, adj1))
        #x2 = torch.sigmoid(self.gc2(x2, adj1))
        #x2 = self.dropout(x2)
        #x1 = self.dropout(x1)
        #feat1,feat2=self.crat(x1,adj1,x2,adj2)
        #feat1, feat2 =x12,x22
        #rec_feat1 = torch.sigmoid(self.gc3(feat1, adj1))
        #rec_feat2 = torch.sigmoid(self.gc3(feat2, adj2))
        return x1_2,x2_2

class GraphFeatureLearning_external_1_x(nn.Module):#重构顶点
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GraphFeatureLearning_external_1, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, 2*nhid)
        self.crat = CrossGraphAttention(2*nhid,2*nhid)
        self.gc3 = GraphConvolution( 2*nhid, nclass)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self, x1, adj1,x2,adj2):
        x1 = torch.sigmoid(self.gc1(x1, adj1))
        #x1= torch.sigmoid(self.gc2(x1, adj1))
        x2=torch.sigmoid(self.gc1(x2, adj2))
        #x2 = torch.sigmoid(self.gc2(x2, adj2))
        x2 = self.dropout(x2)
        x1 = self.dropout(x1)
        x1 = torch.sigmoid(self.gc2(x1, adj1))
        x2 = torch.sigmoid(self.gc2(x2, adj1))

        #feat1,feat2=self.crat(x1,adj1,x2,adj2)
        #feat1, feat2 =x12,x22
        rec_feat1 = torch.sigmoid(self.gc3(x1, adj1))
        rec_feat2 = torch.sigmoid(self.gc3(x2, adj2))
        return rec_feat1,rec_feat2
#



class GraphFeatureLearning_internal(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GraphFeatureLearning_internal, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, 2 * nhid)
        self.crat = CrossGraphAttention(nhid,nhid)
        self.dropout = nn.Dropout(p=dropout)
        #self.gc3 = GraphConvolution(2 * nhid, nclass)

    def forward(self, x1, adj1,x2,adj2):
        x1 = torch.sigmoid(self.gc1(x1, adj1))
        x1= torch.sigmoid(self.gc2(x1, adj1))
        x2=torch.sigmoid(self.gc1(x2, adj2))
        x2 = torch.sigmoid(self.gc2(x2, adj2))
        x1,x2=self.crat(x1,adj1,x2,adj2)

        return x1,x2
