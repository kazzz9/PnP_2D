import os
import math
import numpy as np
import cv2
from torchvision.utils import make_grid
import torch
import matplotlib.pyplot as plt
# from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / \
        (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(
            math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(img_path, img)


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')




#----------------------------- usage of nmr -----------------------------
def visualize_spectrum(lr_img, sr_img, hr_img, save_dir, step):
    os.makedirs(save_dir, exist_ok=True)

    # ========================
    # 1) 从模型输出恢复单通道幅度谱
    # ========================
    # 模型输入是 3 通道，但 3 个通道相同 → 取其中一个即可
    # if lr_img.ndim == 3:
    #     lr = lr_img[0]  
    # else:
    #     lr = lr_img

    # if sr_img.ndim == 3:
    #     sr = sr_img[0]
    # else:
    #     sr = sr_img

    # if hr_img.ndim == 3:
    #     hr = hr_img[0]
    # else:
    #     hr = hr_img

    # # 取幅度
    # lr = np.abs(lr)
    # sr = np.abs(sr)
    # hr = np.abs(hr)
    def prepare_img(img):
        if img.ndim == 3:  # C,H,W
            img = img[0]
        img = np.abs(img)    # 幅度谱非负
        # [-1,1] -> [0,1] 映射
        img = (img + 1.0) / 2.0
        img = np.clip(img, 0, 1)
        return img

    lr = prepare_img(lr_img)
    sr = prepare_img(sr_img)
    hr = prepare_img(hr_img)
    # ========================
    # 2) 可视化等高线图
    # ========================
    fig = plt.figure(figsize=(12, 4))

    ax = fig.add_subplot(131)
    c1 = ax.contour(lr, levels=20, cmap='viridis')
    ax.set_title("Input (NUS spectrum)")
    ax.set_xlabel('f2 (Frequency)')
    ax.set_ylabel('f1 (Frequency)')
    plt.colorbar(c1, ax=ax)

    ax = fig.add_subplot(132)
    c2 = ax.contour(sr, levels=20, cmap='viridis')
    ax.set_title("SR3 Reconstruction")
    ax.set_xlabel('f2 (Frequency)')
    ax.set_ylabel('f1 (Frequency)')
    plt.colorbar(c2, ax=ax)

    ax = fig.add_subplot(133)
    c3 = ax.contour(hr, levels=20, cmap='viridis')
    ax.set_title("Ground Truth")
    ax.set_xlabel('f2 (Frequency)')
    ax.set_ylabel('f1 (Frequency)')
    plt.colorbar(c3, ax=ax)

    plt.tight_layout()

    save_path = os.path.join(save_dir, f"spec_step_{step}.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved spectrum visualization to {save_path}")
    

# def calculate_psnr_nmr(sr_img, hr_img):
#     # 去掉 batch 维度
#     sr = sr_img[0]
#     hr = hr_img[0]

#     # 如果是多通道，如 (C, H, W)，取第一个通道 或者取幅度 (NMR 建议取幅度)
#     if sr.ndim == 3:
#         sr = sr[0]
#         hr = hr[0]

#     # 取幅度（NMR 必须）
#     sr = np.abs(sr)
#     hr = np.abs(hr)

#     # 转 float64
#     sr = sr.astype(np.float64)
#     hr = hr.astype(np.float64)

#     # 归一化到 0~255
#     sr = (sr - sr.min()) / (sr.max() - sr.min()) * 255
#     hr = (hr - hr.min()) / (hr.max() - hr.min()) * 255

#     mse = np.mean((sr - hr) ** 2)
#     if mse == 0:
#         return float('inf')
#     return 20 * math.log10(255.0 / math.sqrt(mse))


# def calculate_ssim_nmr(sr_img, hr_img):
#     # 1) remove batch
#     sr = sr_img[0]
#     hr = hr_img[0]

#     # 2) remove channel if exists
#     if sr.ndim == 3:
#         sr = sr[0]
#         hr = hr[0]

#     # 3) normalize to 0~255 for SSIM compatibility
#     sr = (sr - sr.min()) / (sr.max() - sr.min() + 1e-12) * 255
#     hr = (hr - hr.min()) / (hr.max() - hr.min() + 1e-12) * 255

#     sr = sr.astype(np.uint8)
#     hr = hr.astype(np.uint8)

#     return ssim(sr, hr)   # 使用你原来的 ssim()

# def calculate_psnr_nmr(sr_img, hr_img):
#     def prepare_img(img):
#         if img.ndim == 3:
#             img = img[0]
#         img = np.abs(img)
#         img = (img + 1.0) / 2.0   # [-1,1] -> [0,1]
#         img = img / (img.max() + 1e-12) * 255
#         return img.astype(np.float64)

#     sr = prepare_img(sr_img)
#     hr = prepare_img(hr_img)

#     mse = np.mean((sr - hr) ** 2)
#     if mse == 0:
#         return float('inf')
#     return 20 * math.log10(255.0 / math.sqrt(mse))


# def calculate_ssim_nmr(sr_img, hr_img):
#     def prepare_img(img):
#         if img.ndim == 3:
#             img = img[0]
#         img = np.abs(img)
#         img = (img + 1.0) / 2.0   # [-1,1] -> [0,1]
#         img = img / (img.max() + 1e-12) * 255
#         return img.astype(np.uint8)

#     sr = prepare_img(sr_img)
#     hr = prepare_img(hr_img)
#     return ssim(sr, hr)

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# def calculate_psnr_nmr(sr_img, hr_img):
#     """
#     使用 scikit-image 的标准 PSNR
#     """
#     def prepare_img(img):
#         while img.ndim > 2:
#             img = img[0] if img.shape[0] == 1 else img.squeeze()
#         img = np.abs(img)
#         # img = (img + 1.0) / 2.0  # [-1, 1] -> [0, 1]
#         if img.max() > 1.0 or img.min() < -1.0:
#             # 如果不在 [-1, 1]，先做 min-max
#             img = (img - img.min()) / (img.max() - img.min() + 1e-10)
#         else:
#             # 假设在 [-1, 1]
#             img = (img + 1.0) / 2.0
#         img = np.clip(img, 0.0, 1.0)
#         return img.astype(np.float64)
    
#     sr = prepare_img(sr_img)
#     hr = prepare_img(hr_img)
    
#     # data_range=1.0 表示数据范围是 [0, 1]
#     return peak_signal_noise_ratio(hr, sr, data_range=1.0)

def calculate_psnr_nmr(sr_img, hr_img):
    def prepare_img(img):
        if img.ndim == 3:
            img = img[0]
        img = (img + 1.0) / 2.0
        img = np.clip(img, 0.0, 1.0)
        return img.astype(np.float64)
    
    sr = prepare_img(sr_img)
    hr = prepare_img(hr_img)
    
    mse = np.mean((sr - hr) ** 2)
    if mse < 1e-10:
        return 100.0
    
    return 20 * np.log10(1.0 / np.sqrt(mse))

def calculate_psnr_nmr_01(sr_img, hr_img):
    
    if sr_img.ndim == 3:
        sr_img = sr_img[0]
    if hr_img.ndim == 3:
        hr_img = hr_img[0]

    sr = np.clip(sr_img, 0.0, 1.0).astype(np.float64)
    hr = np.clip(hr_img, 0.0, 1.0).astype(np.float64)
    
    mse = np.mean((sr - hr) ** 2)
    if mse < 1e-10:
        return 100.0
    
    return 20 * np.log10(1.0 / np.sqrt(mse))

def calculate_ssim_nmr(sr_img, hr_img):
    """
    使用 scikit-image 的标准 SSIM
    """
    def prepare_img(img):
        while img.ndim > 2:
            img = img[0] if img.shape[0] == 1 else img.squeeze()
        
        img = np.abs(img)
        # img = (img + 1.0) / 2.0
        if img.max() > 1.0 or img.min() < -1.0:
            img = (img - img.min()) / (img.max() - img.min() + 1e-10)
        else:
            img = (img + 1.0) / 2.0
        img = np.clip(img, 0.0, 1.0)
        return img.astype(np.float64)
    
    sr = prepare_img(sr_img)
    hr = prepare_img(hr_img)

    min_dim = min(sr.shape)
    win_size = min(11, min_dim if min_dim % 2 == 1 else min_dim - 1)
    
    # data_range=1.0, win_size 必须是奇数且 <= 图像最小维度
    return structural_similarity(hr, sr, data_range=1.0, win_size=win_size)




import os
import math
import numpy as np
import cv2
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / \
        (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(
            math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

def save_contour(nus_spec, recon_spec, label_spec, save_path):
    fig = plt.figure(figsize=(15, 4))
    contour_levels = 50
    ax = fig.add_subplot(131)
    contour_input = ax.contour(np.abs(nus_spec), levels=contour_levels, cmap='viridis')
    ax.set_title('Contour Plot of NOISE_input')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(contour_input, ax=ax, label='Amplitude')
    ax = fig.add_subplot(132)
    contour_input = ax.contour(np.abs(recon_spec), levels=contour_levels, cmap='viridis')
    ax.set_title('Contour Plot of Recon_result')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(contour_input, ax=ax, label='Amplitude')
    ax = fig.add_subplot(133)
    contour_input = ax.contour(np.abs(label_spec), levels=contour_levels, cmap='viridis')
    ax.set_title('Contour Plot of GT_label')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(contour_input, ax=ax, label='Amplitude')
    plt.savefig(save_path)

def save_plot(nus_spec, recon_spec, label_spec, sr_spec, save_path):
    fig = plt.figure(figsize=(15, 4))
    ax = fig.add_subplot(141)
    ax.plot(np.abs(nus_spec))
    ax.set_title('Contour Plot of NOISE_input')
    ax = fig.add_subplot(142)
    ax.plot(np.abs(recon_spec))
    ax.set_title('Contour Plot of Recon_result')
    ax = fig.add_subplot(143)
    ax.plot(np.abs(label_spec))
    ax = fig.add_subplot(144)
    ax.plot(np.abs(sr_spec))
    ax.set_title('Contour Plot of GT_label')
    plt.savefig(save_path)

def plot_contour(nus_spec, recon_spec, label_spec, fig_num):
    fig = plt.figure(fig_num, figsize=(15, 4))
    contour_levels = 50
    ax = fig.add_subplot(131)
    contour_input = ax.contour(np.abs(nus_spec), levels=contour_levels, cmap='viridis')
    ax.set_title('Contour Plot of NOISE_input')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(contour_input, ax=ax, label='Amplitude')
    ax = fig.add_subplot(132)
    contour_input = ax.contour(np.abs(recon_spec), levels=contour_levels, cmap='viridis')
    ax.set_title('Contour Plot of Recon_result')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(contour_input, ax=ax, label='Amplitude')
    ax = fig.add_subplot(133)
    contour_input = ax.contour(np.abs(label_spec), levels=contour_levels, cmap='viridis')
    ax.set_title('Contour Plot of GT_label')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(contour_input, ax=ax, label='Amplitude')

def save_img(img, img_path, mode='RGB'):
    # cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    cv2.imwrite(img_path, img)


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_mse(hr, sr):
    return np.mean((hr - sr)**2)

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
