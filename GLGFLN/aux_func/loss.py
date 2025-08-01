import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import pairwise_distances

pairwise_distance = torch.nn.PairwiseDistance(p=2)

def cal_nonlocal_dist(vector, band_width):
    euc_dis = pairwise_distances(vector)
    gaus_dis = np.exp(- euc_dis * euc_dis / (band_width * band_width))
    return euc_dis

def change_consistent_loss(feat1,feat2,neigh_idx_t1,neigh_idx_t2,knn,band_width_t1,band_width_t2,obj_nums):
    #feat1 = feat1.data.cpu().numpy()
    #feat2 = feat2.data.cpu().numpy()
    new_adj1 = torch.zeros(obj_nums, obj_nums)
    new_adj2 = torch.zeros(obj_nums, obj_nums)
    for i in range(obj_nums):
        new_adj1[i] = pairwise_distance(feat1[i].unsqueeze(0),feat1)
        new_adj2[i] = pairwise_distance(feat2[i].unsqueeze(0), feat2)
    #fx_node_dist=torch.zeros(obj_nums)
    #fy_node_dist=torch.zeros(obj_nums)
    fx_node_dist = torch.zeros(obj_nums)
    fy_node_dist = torch.zeros(obj_nums)
    for i in range(obj_nums):
        fx_node_dist[i] = torch.mean(
            torch.abs(new_adj1[i, neigh_idx_t1[i, 1:knn]] - new_adj1[i, neigh_idx_t2[i, 1:knn]]))
        fy_node_dist[i] = torch.mean(
            torch.abs(new_adj2[i, neigh_idx_t2[i, 1:knn]] - new_adj2[i, neigh_idx_t1[i, 1:knn]]))
    #fx_node_dist=torch.from_numpy(fx_node_dist).cpu().float()
    #fy_node_dist=torch.from_numpy(fy_node_dist).cpu().float()
    fxy=torch.sum(torch.abs(fx_node_dist-fy_node_dist))
    return fxy,fx_node_dist,fy_node_dist



def change_consistent_loss_1(feat1,feat2,eud_t1,eud_t2,rec_adj_t1,rec_adj_t2):

    diff_forward  = torch.mean(torch.abs(eud_t1 * rec_adj_t1 -  eud_t1 *rec_adj_t2), dim=1)  # *obj_nums/k
    diff_backward = torch.mean(torch.abs(eud_t2 * rec_adj_t1 - eud_t2*rec_adj_t2), dim=1)
    sim=torch.sum(torch.abs(diff_forward-diff_backward))

    fb=1-(torch.sum(diff_forward*diff_backward)/(torch.sqrt(torch.sum(diff_forward*diff_forward))*torch.sqrt(torch.sum(diff_backward*diff_backward))))
    #fb=1-(torch.sum(diff_forward*diff_backward)/(torch.sqrt(torch.sum(diff_forward*diff_forward))*torch.sqrt(torch.sum(diff_backward*diff_backward))))
    #fb=torch.sum(torch.abs((diff_backward-torch.min(diff_backward))/torch.max(diff_backward)-(diff_forward-torch.min(diff_forward))/torch.max(diff_forward)))
    #fb=torch.sum(torch.abs(diff_backward-diff_forward))
    #fb=1-torch.cosine_similarity(diff_forward,diff_backward,dim=0)
    #fb=1/(torch.sum(sim))
    sparse_loss=torch.sum(torch.abs(diff_forward))+torch.sum(torch.abs(diff_backward))
    return fb,sparse_loss


