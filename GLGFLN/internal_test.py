import imageio
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from skimage.segmentation import slic
from aux_func.preprocess import preprocess_img
from aux_func.graph_func import construct_affinity_matrix
from aux_func.graph_func import construct_nolocal_matrix
from aux_func.graph_func import normalize_adj
from aux_func.acc_ass import assess_accuracy
from aux_func.clustering import otsu
from model.GCN import GraphFeatureLearning_internal
from model.GCN import GraphFeatureLearning_external

def load_checkpoint_for_evaluation(model, checkpoint):
    #saved_state_dict = torch.load(checkpoint, map_location='cuda:0')
    saved_state_dict = torch.load(checkpoint, map_location ='cpu')
    #model.load_state_dict(torch.load(args.pre_train, map_location={'0': 'CPU'}), strict=True)
    model.load_state_dict(saved_state_dict)
    model.cpu()
    model.eval()

def test(args):
    img_t1 = imageio.imread('./data/FR/T1.png')  # .astype(np.float32)
    img_t2 = imageio.imread('./data/FR/T2.png')  # .astype(np.float32)
    ground_truth_changed = imageio.imread('./data/FR/GT.png')
    ground_truth_changed=ground_truth_changed[:,:,0]
    ground_truth_unchanged = 255 - ground_truth_changed

    height, width, channel_t1 = img_t1.shape
    _, _, channel_t2 = img_t2.shape
    objects = slic(img_t1, n_segments=args.n_seg, compactness=args.cmp)

    img_t1 = preprocess_img(img_t1, d_type='opt', norm_type='1')
    img_t2 = preprocess_img(img_t2, d_type='opt', norm_type='1')
    # objects = np.load('./object_idx.npy')
    obj_nums = np.max(objects)
    min = np.min(objects)
    Internal_node_set_t1 = []
    Internal_node_set_t2 = []
    for idx in range(1,obj_nums+ 1):
        obj_idx = objects == idx
        Internal_node_set_t1.append(img_t1[obj_idx])
        Internal_node_set_t2.append(img_t2[obj_idx])
    am_set_t1 = construct_affinity_matrix(img_t1, objects, 0.5)
    am_set_t2 = construct_affinity_matrix(img_t2, objects, 0.5)
    print('internal graph construction')
    external_node_set_t1 = np.zeros((obj_nums, 3))
    external_node_set_t2 = np.zeros((obj_nums, 3))
    for idx in range(0,obj_nums):
        #a=np.mean(Internal_node_set_t1[0], axis=0)
        external_node_set_t1[idx] = np.mean(Internal_node_set_t1[idx], axis=0)
        external_node_set_t2[idx] = np.mean(Internal_node_set_t2[idx], axis=0)
    external_graph_t1 = construct_nolocal_matrix(external_node_set_t1, 0.5)
    external_graph_t2 = construct_nolocal_matrix(external_node_set_t2, 0.5)
    print('external graph construction')
    k_ratio=0.05
    neigh_idx_t1 = np.argsort(-external_graph_t1, axis=1)[:,0:int(150)]
    neigh_idx_t2 = np.argsort(-external_graph_t2, axis=1)[:,0:int(150)]
    exernal_adj_t1 = np.zeros((obj_nums, obj_nums))
    exernal_adj_t2 = np.zeros((obj_nums, obj_nums))
    for idx in range(0,obj_nums):
        exernal_adj_t1[idx, neigh_idx_t1[idx]] = 1
        exernal_adj_t2[idx, neigh_idx_t2[idx]] = 1
    exernal_nor_adj_t1 = normalize_adj(exernal_adj_t1)
    exernal_nor_adj_t2 = normalize_adj(exernal_adj_t2)
    print('internal graph learning')
    GFLI_model=GraphFeatureLearning_internal(nfeat=3,nhid=16,nclass=3,dropout=0.5)
    optimizer_GFLI_model=optim.AdamW(GFLI_model.parameters(), lr=1e-4, weight_decay=1e-6)
    GFLI_model.cpu()
    GFLI_model.train()

    # GFLE_model = GraphFeatureLearning_external(nfeat=3, nhid=16, nclass=3, dropout=0.5)
    # optimizer_GFLE_model = optim.AdamW(GFLE_model.parameters(), lr=1e-4, weight_decay=1e-6)
    # GFLE_model.cpu()
    # GFLE_model.train()
    if 1:
        for _epoch in range(args.epoch):
            for _iter in range(obj_nums):
                node_t1 = Internal_node_set_t1[_iter]  # np.expand_dims(node_set_t1[_iter], axis=0)
                adj_t1, norm_adj_t1 = am_set_t1[_iter]  # np.expand_dims(am_set_t1[_iter], axis=0)
                node_t1 = torch.from_numpy(node_t1).cpu().float()
                adj_t1 = torch.from_numpy(adj_t1).cpu().float()
                norm_adj_t1 = torch.from_numpy(norm_adj_t1).cpu().float()

                node_t2 = Internal_node_set_t2[_iter]  # np.expand_dims(node_set_t2[_iter], axis=0)
                adj_t2, norm_adj_t2 = am_set_t2[_iter]  # np.expand_dims(am_set_t2[_iter], axis=0)
                node_t2 = torch.from_numpy(node_t2).cpu().float()
                adj_t2 = torch.from_numpy(adj_t2).cpu().float()
                norm_adj_t2 = torch.from_numpy(norm_adj_t2).cpu().float()
                internal_feat1,internal_feat2=GFLI_model(node_t1,norm_adj_t1,node_t2,norm_adj_t2)
                recon_adj_t1 = (torch.matmul(internal_feat1, internal_feat1.T))
                recon_adj_t2 = (torch.matmul(internal_feat2, internal_feat2.T))

                cnstr_loss_t1 = F.mse_loss(input=recon_adj_t1, target=adj_t1)
                cnstr_loss_t2 = F.mse_loss(input=recon_adj_t2, target=adj_t2)

                internal_loss=cnstr_loss_t1+cnstr_loss_t2
                internal_loss.backward()
                optimizer_GFLI_model.step()
                if (_iter + 1) % 10 == 0:
                    print(f'Epoch is {_epoch + 1}, iter is {_iter}, mse loss is {internal_loss.item()}')
        torch.save(GFLI_model.state_dict(), './GFLI_model_weight/Internal.pth')
    restore_from = './GFLI_model_weight/Internal.pth'
    load_checkpoint_for_evaluation(GFLI_model, restore_from)
    GFLI_model.eval()
    diff_set_internal = []

    for _iter in range(obj_nums):
        node_t1 = Internal_node_set_t1[_iter]  # np.expand_dims(node_set_t1[_iter], axis=0)
        node_t2 = Internal_node_set_t1[_iter]  # np.expand_dims(node_set_t2[_iter], axis=0)
        adj_t1, norm_adj_t1 = am_set_t1[_iter]  # np.expand_dims(am_set_t1[_iter], axis=0)
        adj_t2, norm_adj_t2 = am_set_t2[_iter]  # np.expand_dims(am_set_t2[_iter], axis=0)

        node_t1 = torch.from_numpy(node_t1).cpu().float()
        node_t2 = torch.from_numpy(node_t2).cpu().float()
        norm_adj_t1 = torch.from_numpy(norm_adj_t1).cpu().float()
        norm_adj_t2 = torch.from_numpy(norm_adj_t2).cpu().float()

        feat_t1, feat_t2 = GFLI_model(node_t1, norm_adj_t1, node_t2, norm_adj_t2)
        diff_set_internal.append((torch.mean(torch.abs(feat_t1-feat_t2))).data.cpu().numpy())
    diff_map_internal = np.zeros((height, width))
    for i in range(0, obj_nums):
        diff_map_internal[objects == i + 1] = diff_set_internal[i]

    diff_map_internal = np.reshape(diff_map_internal, (height * width, 1))

    threshold = otsu(diff_map_internal)
    diff_map_internal = np.reshape(diff_map_internal, (height, width))

    bcm_internal = np.zeros((height, width)).astype(np.uint8)
    bcm_internal[diff_map_internal > threshold] = 255
    bcm_internal[diff_map_internal <= threshold] = 0

    conf_mat, oa, f1, kappa_co = assess_accuracy(ground_truth_changed, ground_truth_unchanged, bcm_internal)

    imageio.imsave('./result/Internal.png', bcm_internal)
    diff_map_internal = 255 * (diff_map_internal - np.min(diff_map_internal)) / (np.max(diff_map_internal) - np.min(diff_map_internal))
    imageio.imsave('./result/Inernal_DI.png', diff_map_internal.astype(np.uint8))

    print(conf_mat)
    print(oa)
    print(f1)
    print(kappa_co)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="XXX")
    parser.add_argument('--n_seg', type=int, default=300,
                        help='Approximate number of objects obtained by the segmentation algorithm')
    parser.add_argument('--cmp', type=int, default=30, help='Compectness of the obtained objects')
    parser.add_argument('--epoch', type=int, default=100, help='tarining epoch')
    args = parser.parse_args()
    test(args)
