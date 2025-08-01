import imageio
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from skimage.segmentation import slic
from aux_func.preprocess import preprocess_img
from aux_func.graph_func import construct_affinity_matrix

from aux_func.graph_func import normalize_adj
from aux_func.acc_ass import assess_accuracy
from aux_func.clustering import otsu
from model.GCN import GraphFeatureLearning_internal
from model.GCN import GraphFeatureLearning_external_1
from model.GCN import GraphFeatureLearning_external_2
from sklearn.metrics.pairwise import pairwise_distances
from aux_func.loss import change_consistent_loss_1


def construct_nolocal_matrix(vector, band_width):
    euc_dis = pairwise_distances(vector)
    gaus_dis = np.exp(- euc_dis * euc_dis / (band_width * band_width))
    return euc_dis,gaus_dis

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
    ground_truth_changed = imageio.imread('./data/FR/gt.png')
    #ground_truth_changed=ground_truth_changed[:,:,0]
    ground_truth_unchanged = 255 - ground_truth_changed


    height, width, channel_t1 = img_t1.shape
    _, _, channel_t2 = img_t2.shape
    image_t12= np.zeros((height, width,6))
    image_t12[:,:,0:3]=img_t1[:,:,:]
    image_t12[:,:,3:6]=img_t2[:,:,:]
    #image_t12=np.concatenate((img_t1_,img_t2),axis=2)
    objects = slic(img_t2, n_segments=args.n_seg, compactness=args.cmp)

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
    #am_set_t1 = construct_affinity_matrix(img_t1, objects, 0.5)
    #am_set_t2 = construct_affinity_matrix(img_t2, objects, 0.5)
    print('internal graph construction')
    external_node_set_t1 = np.zeros((obj_nums, 3))
    external_node_set_t2 = np.zeros((obj_nums, 3))
    for idx in range(0,obj_nums):
        #a=np.mean(Internal_node_set_t1[0], axis=0)
        external_node_set_t1[idx] = np.mean(Internal_node_set_t1[idx], axis=0)
        external_node_set_t2[idx] = np.mean(Internal_node_set_t2[idx], axis=0)
    external_eudis_t1,external_gaus_dis_t1 = construct_nolocal_matrix(external_node_set_t1, args.kernal)
    external_eudis_t2,external_gaus_dis_t2 = construct_nolocal_matrix(external_node_set_t2, args.kernal)
    print('external graph construction')

    # neigh_idx_t1 = np.argsort(-external_graph_t1, axis=1)[:,1:int(obj_nums*k_ratio)]
    # neigh_idx_t2 = np.argsort(-external_graph_t2, axis=1)[:,1:int(obj_nums*k_ratio)]

    # 对距离举证进行排序，去KNN
    neigh_idx_t1 = np.argsort(external_eudis_t1, axis=1)[:, 0:args.knn]#对距离举证进行排序，去KNN
    neigh_idx_t2 = np.argsort(external_eudis_t2, axis=1)[:, 0:args.knn]

    #构造邻接矩阵
    exernal_adj_t1_nolocal = np.zeros((obj_nums, obj_nums))
    exernal_adj_t2_nolocal = np.zeros((obj_nums, obj_nums))
    for idx in range(0,obj_nums):
        exernal_adj_t1_nolocal[idx, neigh_idx_t1[idx]] = 1
        exernal_adj_t2_nolocal[idx, neigh_idx_t2[idx]] = 1
    # exernal_nor_adj_t1 = normalize_adj(exernal_adj_t1_nolocal)
    # exernal_nor_adj_t2 = normalize_adj(exernal_adj_t2_nolocal)
    exernal_nor_adj_t1 = normalize_adj(exernal_adj_t1_nolocal)
    exernal_nor_adj_t2 = normalize_adj(exernal_adj_t2_nolocal)
    print('internal graph learning')
    # GFLI_model=GraphFeatureLearning_internal(nfeat=3,nhid=8,nclass=3,dropout=0.5)
    # optimizer_GFLI_model=optim.AdamW(GFLI_model.parameters(), lr=1e-4, weight_decay=1e-6)
    # GFLI_model.cpu()
    # GFLI_model.train()

    GFLE_model = GraphFeatureLearning_external_2(nfeat=3, nhid=16, nclass=3, dropout=0.1)
    optimizer_GFLE_model = optim.AdamW(GFLE_model.parameters(), lr=1e-2, weight_decay=1e-4)
    GFLE_model.cpu()
    GFLE_model.train()
    if args.train:
        with open('results_external.txt', 'w') as result_file:
            for _epoch in range(args.epoch):
                node_t1 = external_node_set_t1 # np.expand_dims(node_set_t1[_iter], axis=0)
                adj_t1 = exernal_adj_t1_nolocal
                norm_adj_t1 = exernal_nor_adj_t1
                node_t1 = torch.from_numpy(node_t1).cpu().float()
                adj_t1 = torch.from_numpy(adj_t1).cpu().float()
                norm_adj_t1 = torch.from_numpy(norm_adj_t1).cpu().float()

                node_t2 = external_node_set_t2  # np.expand_dims(node_set_t1[_iter], axis=0)
                adj_t2 = exernal_adj_t2_nolocal
                norm_adj_t2 = exernal_nor_adj_t2
                node_t2 = torch.from_numpy(node_t2).cpu().float()
                adj_t2 = torch.from_numpy(adj_t2).cpu().float()
                norm_adj_t2 = torch.from_numpy(norm_adj_t2).cpu().float()

                internal_feat1,internal_feat2=GFLE_model(node_t1,norm_adj_t1,node_t2,norm_adj_t2)
                recon_adj_t1 = (torch.matmul(internal_feat1, internal_feat1.T))
                recon_adj_t2 = (torch.matmul(internal_feat2, internal_feat2.T))

                sorted1, indices1 = torch.sort(recon_adj_t1)  # 在维度1上按照升序排列
                sorted2, indices2 = torch.sort(recon_adj_t2)  # 在维度1上按照升序排列

                neigh_idx_t1 = indices1[:, obj_nums - args.knn:obj_nums]  # 对距离举证进行排序，去KNN
                neigh_idx_t2 = indices2[:, obj_nums - args.knn:obj_nums]
                exernal_adj_t1 = torch.zeros(obj_nums, obj_nums)
                exernal_adj_t2 = torch.zeros(obj_nums, obj_nums)
                for idx in range(0, obj_nums):
                    exernal_adj_t1[idx, neigh_idx_t1[idx]] = 1
                    exernal_adj_t2[idx, neigh_idx_t2[idx]] = 1


                cnstr_loss_t1 = F.mse_loss(input=recon_adj_t1, target=adj_t1)
                cnstr_loss_t2 = F.mse_loss(input=recon_adj_t2, target=adj_t2)

                # eud_t1 = torch.from_numpy(external_gaus_dis_t1).cpu().float()
                # eud_t2 = torch.from_numpy(external_gaus_dis_t2).cpu().float()
                eud_t1 = torch.from_numpy(external_eudis_t1).cpu().float()
                eud_t2 = torch.from_numpy(external_eudis_t2).cpu().float()

                rec_loss=cnstr_loss_t1+cnstr_loss_t2
                total_loss=rec_loss
                total_loss.backward()
                optimizer_GFLE_model.step()
                print(f'Epoch is {_epoch + 1}, rec loss is {rec_loss.item()}')

                # external_graph_t1 = torch.from_numpy(external_gaus_dis_t1).cpu().float()
                # external_graph_t2 = torch.from_numpy(external_gaus_dis_t2).cpu().float()



                external_graph_t1 = torch.from_numpy(external_eudis_t1).cpu().float()
                external_graph_t2 = torch.from_numpy(external_eudis_t2).cpu().float()
                #exernal_adj_t1 = torch.from_numpy(exernal_adj_t1).cpu().float()
                #exernal_adj_t2 = torch.from_numpy(exernal_adj_t2).cpu().float()
                diff_forward = torch.mean(torch.abs(external_graph_t1 * (exernal_adj_t1 - exernal_adj_t2)), dim=1)
                diff_backward = torch.mean(torch.abs(external_graph_t2 * (exernal_adj_t1 - exernal_adj_t2)),dim=1)
                # #
                # diff_forward = torch.mean(torch.abs(external_graph_t1 * recon_adj_t1 - external_graph_t1*recon_adj_t2),dim = 1)
                # diff_backward = torch.mean(torch.abs(external_graph_t2 * recon_adj_t1 - external_graph_t2*recon_adj_t2),  dim=1)
                # diff_forward = torch.mean(torch.abs(external_graph_t1 * (recon_adj_t1 -  recon_adj_t2)), dim=1)
                # diff_backward = torch.mean(torch.abs(external_graph_t2 * (recon_adj_t1 - recon_adj_t2)), dim=1)
                diff_forward = diff_forward.data.cpu().numpy()
                diff_backward = diff_backward.data.cpu().numpy()
                #diff_total = (diff_forward / np.max(diff_forward) + diff_backward / np.max(diff_backward)) / 2
                diff_total = (diff_forward  + diff_backward ) / 2
                diff_map_external_forward = np.zeros((height, width))
                for i in range(0, obj_nums):
                    diff_map_external_forward[objects == i + 1] = diff_forward[i]

                diff_map_external_forward = np.reshape(diff_map_external_forward, (height * width, 1))

                threshold = otsu(diff_map_external_forward)
                diff_map_external_forward = np.reshape(diff_map_external_forward, (height, width))

                bcm_external_forward = np.zeros((height, width)).astype(np.uint8)
                bcm_external_forward[diff_map_external_forward > threshold] = 255
                bcm_external_forward[diff_map_external_forward <= threshold] = 0

                conf_mat, oa, f1, kappa_co = assess_accuracy(ground_truth_changed, ground_truth_unchanged,
                                                             bcm_external_forward)

                imageio.imsave(f'./result_external/External_forward{_epoch+1}.png', bcm_external_forward)
                diff_map_external_forward = 255 * (diff_map_external_forward - np.min(diff_map_external_forward)) / (
                            np.max(diff_map_external_forward) - np.min(diff_map_external_forward))
                imageio.imsave(f'./result_external/External_DI_forward{_epoch+1}.png', diff_map_external_forward.astype(np.uint8))

                print(conf_mat)
                print(oa)
                print(f1)
                print(kappa_co)
                #result_file.write(f'Epoch: {_epoch + 1}, External_forward OA: {oa}, F1: {f1}, Kappa: {kappa_co}\n')

                # backward------------------------------------------------------------------
                diff_map_external_backward = np.zeros((height, width))
                for i in range(0, obj_nums):
                    diff_map_external_backward[objects == i + 1] = diff_backward[i]

                diff_map_external_backward = np.reshape(diff_map_external_backward, (height * width, 1))

                threshold = otsu(diff_map_external_backward)
                diff_map_external_backward = np.reshape(diff_map_external_backward, (height, width))

                bcm_external_backward = np.zeros((height, width)).astype(np.uint8)
                bcm_external_backward[diff_map_external_backward > threshold] = 255
                bcm_external_backward[diff_map_external_backward <= threshold] = 0

                conf_mat, oa, f1, kappa_co = assess_accuracy(ground_truth_changed, ground_truth_unchanged,
                                                             bcm_external_backward)

                imageio.imsave(f'./result_external/External_backward{_epoch+1}.png', bcm_external_backward)
                diff_map_external_backward = 255 * (diff_map_external_backward - np.min(diff_map_external_backward)) / (
                        np.max(diff_map_external_backward) - np.min(diff_map_external_backward))
                imageio.imsave(f'./result_external/External_DI_backward{_epoch+1}.png', diff_map_external_backward.astype(np.uint8))

                print(conf_mat)
                print(oa)
                print(f1)
                print(kappa_co)
                #result_file.write(f'Epoch: {_epoch + 1}, External_backward OA: {oa}, F1: {f1}, Kappa: {kappa_co}\n')
                # total------------------------------------------------------------------
                diff_map_external_total = np.zeros((height, width))
                for i in range(0, obj_nums):
                    diff_map_external_total[objects == i + 1] = diff_total[i]

                diff_map_external_total = np.reshape(diff_map_external_total, (height * width, 1))

                threshold = otsu(diff_map_external_total)
                diff_map_external_total = np.reshape(diff_map_external_total, (height, width))

                bcm_external_total = np.zeros((height, width)).astype(np.uint8)
                bcm_external_total[diff_map_external_total > threshold] = 255
                bcm_external_total[diff_map_external_total <= threshold] = 0

                conf_mat, oa, f1, kappa_co = assess_accuracy(ground_truth_changed, ground_truth_unchanged,
                                                             bcm_external_total)

                imageio.imsave(f'./result_external/External_total{_epoch+1}.png', bcm_external_total)
                diff_map_external_total = 255 * (diff_map_external_total - np.min(diff_map_external_total)) / (
                        np.max(diff_map_external_total) - np.min(diff_map_external_total))
                imageio.imsave(f'./result_external/External_DI_total{_epoch+1}.png', diff_map_external_total.astype(np.uint8))

                print(conf_mat)
                print(oa)
                print(f1)
                print(kappa_co)
                result_file.write(f'Epoch: {_epoch + 1}, External_total OA: {oa}, F1: {f1}, Kappa: {kappa_co}\n')
                torch.save(GFLE_model.state_dict(), f'./model_weight_external/External{_epoch + 1}.pth')
        torch.save(GFLE_model.state_dict(), './GFLE_model_weight/External.pth')

    #restore_from = './model_weight_external/External50.pth'
    restore_from = f'D:/PycharmProjects/GiG-main/result_/ukmodel_external/External71.pth'
    load_checkpoint_for_evaluation(GFLE_model, restore_from)
    GFLE_model.eval()



    node_t1 = external_node_set_t1  # np.expand_dims(node_set_t1[_iter], axis=0)

    norm_adj_t1 = exernal_nor_adj_t1
    node_t1 = torch.from_numpy(node_t1).cpu().float()
    #adj_t1 = torch.from_numpy(adj_t1).cpu().float()
    norm_adj_t1 = torch.from_numpy(norm_adj_t1).cpu().float()

    node_t2 = external_node_set_t2  # np.expand_dims(node_set_t1[_iter], axis=0)
    #adj_t2 = exernal_adj_t2
    norm_adj_t2 = exernal_nor_adj_t2
    node_t2 = torch.from_numpy(node_t2).cpu().float()
    #adj_t2 = torch.from_numpy(adj_t2).cpu().float()
    norm_adj_t2 = torch.from_numpy(norm_adj_t2).cpu().float()

    feat_t1, feat_t2 = GFLE_model(node_t1, norm_adj_t1, node_t2, norm_adj_t2)

    recon_adj_t1 = (torch.matmul(feat_t1, feat_t1.T))
    recon_adj_t2 = (torch.matmul(feat_t2, feat_t2.T))

    sorted1, indices1 = torch.sort(recon_adj_t1)  # 在维度1上按照升序排列
    sorted2, indices2 = torch.sort(recon_adj_t2)  # 在维度1上按照升序排列

    neigh_idx_t1 = indices1[:, obj_nums - args.knn:obj_nums]  # 对距离举证进行排序，去KNN
    neigh_idx_t2 = indices2[:, obj_nums - args.knn:obj_nums]
    exernal_adj_t1 = torch.zeros(obj_nums, obj_nums)
    exernal_adj_t2 = torch.zeros(obj_nums, obj_nums)
    for idx in range(0, obj_nums):
        exernal_adj_t1[idx, neigh_idx_t1[idx]] = 1
        exernal_adj_t2[idx, neigh_idx_t2[idx]] = 1

    external_graph_t1 = torch.from_numpy(external_eudis_t1).cpu().float()
    external_graph_t2 = torch.from_numpy(external_eudis_t2).cpu().float()

    external_graph_t1 = torch.from_numpy(external_gaus_dis_t1).cpu().float()
    external_graph_t2 = torch.from_numpy(external_gaus_dis_t2).cpu().float()

    exernal_adj_t1_nolocal = torch.from_numpy(exernal_adj_t1_nolocal).cpu().float()
    exernal_adj_t2_nolocal = torch.from_numpy(exernal_adj_t2_nolocal).cpu().float()
    diff_forward  = torch.mean(torch.abs(external_graph_t1 * recon_adj_t1-external_graph_t1 *recon_adj_t2), dim=1)
    diff_backward = torch.mean(torch.abs(external_graph_t2 * recon_adj_t1-external_graph_t2 *recon_adj_t2), dim=1)
    # diff_forward = torch.mean(torch.abs(external_graph_t1 * exernal_adj_t1_nolocal - external_graph_t1 * exernal_adj_t2_nolocal), dim=1)
    # diff_backward = torch.mean(torch.abs(external_graph_t2 * exernal_adj_t1_nolocal - external_graph_t2 * exernal_adj_t2_nolocal), dim=1)
    # diff_forward = torch.mean(torch.abs(external_graph_t1 * exernal_adj_t1 - external_graph_t1 * exernal_adj_t2), dim=1)
    # diff_backward = torch.mean(torch.abs(external_graph_t2 * exernal_adj_t1 - external_graph_t2 * exernal_adj_t2), dim=1)
    # diff_forward = torch.mean(torch.abs(external_graph_t1 * exernal_adj_t1 - external_graph_t1 * exernal_adj_t2), dim=1)
    # diff_backward = torch.mean(torch.abs(external_graph_t2 * exernal_adj_t1 - external_graph_t2 * exernal_adj_t2), dim=1)
    diff_forward=diff_forward.data.cpu().numpy()
    diff_backward=diff_backward.data.cpu().numpy()
    diff_total=(diff_forward+diff_backward)/2
    #diff_set_external=(torch.mean(torch.abs(feat_t1-feat_t2),1)).data.cpu().numpy()

#forward------------------------------------------------------------------
    diff_map_external_forward = np.zeros((height, width))
    for i in range(0, obj_nums ):
        diff_map_external_forward[objects == i + 1] = diff_forward[i]

    diff_map_external_forward = np.reshape(diff_map_external_forward, (height * width, 1))

    threshold = otsu(diff_map_external_forward)
    diff_map_external_forward = np.reshape(diff_map_external_forward, (height, width))

    bcm_external_forward = np.zeros((height, width)).astype(np.uint8)
    bcm_external_forward[diff_map_external_forward > threshold] = 255
    bcm_external_forward[diff_map_external_forward <= threshold] = 0

    conf_mat, oa, f1, kappa_co = assess_accuracy(ground_truth_changed, ground_truth_unchanged, bcm_external_forward)

    imageio.imsave('./result/External_forward.png', bcm_external_forward)
    diff_map_external_forward = 255 * (diff_map_external_forward - np.min(diff_map_external_forward)) / (np.max(diff_map_external_forward) - np.min(diff_map_external_forward))
    imageio.imsave('./result/External_DI_forward.png', diff_map_external_forward.astype(np.uint8))

    print(conf_mat)
    print(oa)
    print(f1)
    print(kappa_co)

# backward------------------------------------------------------------------
    diff_map_external_backward = np.zeros((height, width))
    for i in range(0, obj_nums):
        diff_map_external_backward[objects == i + 1] = diff_backward[i]

    diff_map_external_backward = np.reshape(diff_map_external_backward, (height * width, 1))

    threshold = otsu(diff_map_external_backward)
    diff_map_external_backward = np.reshape(diff_map_external_backward, (height, width))

    bcm_external_backward = np.zeros((height, width)).astype(np.uint8)
    bcm_external_backward[diff_map_external_backward > threshold] = 255
    bcm_external_backward[diff_map_external_backward <= threshold] = 0

    conf_mat, oa, f1, kappa_co = assess_accuracy(ground_truth_changed, ground_truth_unchanged, bcm_external_backward)

    imageio.imsave('./result/External_backward.png', bcm_external_backward)
    diff_map_external_backward = 255 * (diff_map_external_backward - np.min(diff_map_external_backward)) / (
                np.max(diff_map_external_backward) - np.min(diff_map_external_backward))
    imageio.imsave('./result/External_DI_backward.png', diff_map_external_backward.astype(np.uint8))

    print(conf_mat)
    print(oa)
    print(f1)
    print(kappa_co)
# total------------------------------------------------------------------
    diff_map_external_total = np.zeros((height, width))
    for i in range(0, obj_nums):
        diff_map_external_total[objects == i + 1] = diff_total[i]

    diff_map_external_total = np.reshape(diff_map_external_total, (height * width, 1))

    threshold = otsu(diff_map_external_total)
    diff_map_external_total = np.reshape(diff_map_external_total, (height, width))

    bcm_external_total = np.zeros((height, width)).astype(np.uint8)
    bcm_external_total[diff_map_external_total > threshold] = 255
    bcm_external_total[diff_map_external_total <= threshold] = 0

    conf_mat, oa, f1, kappa_co = assess_accuracy(ground_truth_changed, ground_truth_unchanged, bcm_external_total)

    imageio.imsave('./result/External_total.png', bcm_external_total)
    diff_map_external_total = 255 * (diff_map_external_total - np.min(diff_map_external_total)) / (
                np.max(diff_map_external_total) - np.min(diff_map_external_total))
    imageio.imsave('./result/External_DI_total.png', diff_map_external_total.astype(np.uint8))

    print(conf_mat)
    print(oa)
    print(f1)
    print(kappa_co)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="XXX")
    parser.add_argument('--n_seg', type=int, default=900,
                        help='Approximate number of objects obtained by the segmentation algorithm')
    parser.add_argument('--cmp', type=int, default=15, help='Compectness of the obtained objects')
    parser.add_argument('--knn', type=int, default=50, help='KNN')
    parser.add_argument('--train', type=int, default=1, help='train')
    parser.add_argument('--epoch', type=int, default=100, help='epoch')
    parser.add_argument('--kernal', type=int, default=1, help='kernal')
    args = parser.parse_args()
    test(args)
