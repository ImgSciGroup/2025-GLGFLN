import imageio
import numpy as np
from aux_func.acc_ass import assess_accuracy
from aux_func.clustering import otsu


def fourier_transform(image):
    # 对图像进行傅里叶变换
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)  # 移动频谱到中心
    magnitude_spectrum = np.abs(fshift)  # 幅度谱
    return fshift, magnitude_spectrum


def inverse_fourier_transform(fshift):
    # 对频域图像进行反傅里叶变换
    f_ishift = np.fft.ifftshift(fshift)  # 逆移频谱
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)  # 获取实部作为最终图像
    return img_back


def frequency_split(fshift, cutoff=30):
    # 将频域图像分为低频和高频
    rows, cols = fshift.shape
    crow, ccol = rows // 2, cols // 2

    # 创建一个遮罩
    mask = np.zeros((rows, cols))

    # 设置低频区域的遮罩
    mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 1
    low_f = fshift * mask  # 提取低频信息
    high_f = fshift * (1 - mask)  # 提取高频信息
    return low_f, high_f


def frequency_fusion(f1, f2, alpha=0.5,lam=0.5):
    # 融合两幅图像的低频和高频分量
    low_f1, high_f1 = frequency_split(f1)
    low_f2, high_f2 = frequency_split(f2)

    # 融合低频和高频部分
    low_f_fused = alpha * low_f1 + (1 - alpha) * low_f2
    high_f_fused = alpha * high_f1 + (1 - alpha) * high_f2

    # 合成频域图像
    f_fused = lam*low_f_fused + (1-lam)*high_f_fused
    #f_fused = low_f_fused + high_f_fused
    return f_fused

#
# def plot_image(image, title):
#     plt.imshow(image, cmap='gray')
#     plt.title(title)
#     plt.axis('off')
#     plt.show()


# 载入图像
LocalGCN_DI = imageio.imread(f'E:/sy/UK2/Internal_DI.png').astype(np.float32)
NLocalGCN_DI = imageio.imread(f'E:/sy/UK2/cs/External_DI_total50.png').astype(np.float32)
height, width = LocalGCN_DI.shape
# 确保两幅图像的尺寸一致
assert LocalGCN_DI.shape == NLocalGCN_DI.shape, "两幅图像的尺寸不一致"

# 获取傅里叶变换结果
fshift1, _ = fourier_transform(LocalGCN_DI)
fshift2, _ = fourier_transform(NLocalGCN_DI)

# 图像频率融合
f_fused = frequency_fusion(fshift1, fshift2, alpha=0.80,lam=0.70)

# 反傅里叶变换得到融合图像
fuse_DI = inverse_fourier_transform(f_fused)

fuse_DI = np.reshape(fuse_DI, (height, width))
threshold = otsu(fuse_DI)

bcm = np.zeros((height, width)).astype(np.uint8)
bcm[fuse_DI > threshold] = 255
bcm[fuse_DI <= threshold] = 0
imageio.imsave('./result/FFT_fuse.png', bcm)
fuse_DI = 255 * (fuse_DI - np.min(fuse_DI)) / (
                np.max(fuse_DI) - np.min(fuse_DI))
imageio.imsave('./result/FFT_fuse_DI.png', fuse_DI.astype(np.uint8))
ground_truth_changed = imageio.imread('./data/uk2/gt.png')
#ground_truth_changed = ground_truth_changed[:, :, 0]
ground_truth_unchanged = 255 - ground_truth_changed
conf_mat, oa, f1, kappa_co = assess_accuracy(ground_truth_changed, ground_truth_unchanged, bcm)
print(conf_mat)
print(oa)
print(f1)
print(kappa_co)

# def fuse_DI():
#     # LocalGCN_DI = imageio.imread(f'D:/实验2.0/uk/参数讨论/Local_Sigma/uk/0.1/12,CMI.png').astype(np.float32)
#     # NLocalGCN_DI = imageio.imread(f'D:/实验2.0/uk/参数讨论/Nolocal_Sigma/K=100,Sigma=0.2,CMI.png').astype(np.float32)
#     # height, width = LocalGCN_DI.shape
#     # alpha = np.var(LocalGCN_DI.reshape(-1))
#     # beta = np.var(NLocalGCN_DI.reshape(-1))
#     # fuse_DI = (alpha * LocalGCN_DI + beta * NLocalGCN_DI) / (alpha + beta)
#     # fuse_DI = np.reshape(fuse_DI, (height * width, 1))
#     # threshold = otsu(fuse_DI)
#     # fuse_DI = np.reshape(fuse_DI, (height, width))
#     # bcm = np.zeros((height, width)).astype(np.uint8)
#     # bcm[fuse_DI > threshold] = 255
#     # bcm[fuse_DI <= threshold] = 0
#     # imageio.imsave('./result/Adaptive_Fuse.png', bcm)
#     #
#     # fuse_DI = 255 * ((fuse_DI - np.min(fuse_DI)) / (np.max(fuse_DI) - np.min(fuse_DI)))
#     #
#     # imageio.imsave('./result/Adaptive_Fuse_DI.png', fuse_DI.astype(np.uint8))
#     # #
#     # # LocalGCN_DI=LocalGCN_DI/255
#     # # NLocalGCN_DI=NLocalGCN_DI/255
#     # # fuse_DI=LocalGCN_DI*NLocalGCN_DI*255
#     fuse_DI = imageio.imread(f'D:/NSST/csz111.png').astype(np.float32)
#     height, width = fuse_DI.shape
#     fuse_DI = np.reshape(fuse_DI, (height, width))
#     threshold = otsu(fuse_DI)
#
#     bcm = np.zeros((height, width)).astype(np.uint8)
#     bcm[fuse_DI > threshold] = 255
#     bcm[fuse_DI <= threshold] = 0
#     imageio.imsave('./result/Adaptive_Fuse.png', bcm)
#     imageio.imsave('./result/Adaptive_Fuse_DI.png', fuse_DI.astype(np.uint8))
#     ground_truth_changed = imageio.imread('./data/uk/gt.png')
#     ground_truth_changed = ground_truth_changed[:, :, 0]
#     ground_truth_unchanged = 255 - ground_truth_changed
#     conf_mat, oa, f1, kappa_co = assess_accuracy(ground_truth_changed, ground_truth_unchanged, bcm)
#     print(conf_mat)
#     print(oa)
#     print(f1)
#     print(kappa_co)
#
#
# if __name__ == '__main__':
#     fuse_DI()
