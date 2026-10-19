import os
import torch
import logging
import random
import ctypes
import numpy as np
import scipy.io as scio
from tqdm import tqdm
from torch.utils.data import Dataset

# # -------------------1D数据-----------

class NUSNMRDataset(Dataset):
    def __init__(self, num_samples, split='train', need_LR=False):
        super(NUSNMRDataset,self).__init__()
        self.num_samples = num_samples
        self.split = split
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        NUS_FID, NOISE_input, GT_label, factor_matrix = self.__gen_signal__(idx)
        max_amp = np.max(np.abs(NOISE_input))

        NUS_FID = torch.complex(torch.from_numpy(NUS_FID.real).float(), torch.from_numpy(NUS_FID.imag).float()) / (max_amp * 6.67)
        NOISE_input = torch.complex(torch.from_numpy(NOISE_input.real).float(), torch.from_numpy(NOISE_input.imag).float()) / (max_amp)
        GT_label = torch.complex(torch.from_numpy(GT_label.real).float(), torch.from_numpy(GT_label.imag).float()) / (max_amp * 6.67)
        # NUS_FID = torch.complex(torch.from_numpy(NUS_FID.real).float(), torch.from_numpy(NUS_FID.imag).float()) / (max_amp * 1)
        # NOISE_input = torch.complex(torch.from_numpy(NOISE_input.real).float(), torch.from_numpy(NOISE_input.imag).float()) / (max_amp)
        # GT_label = torch.complex(torch.from_numpy(GT_label.real).float(), torch.from_numpy(GT_label.imag).float()) / (max_amp * 1)
        factor_matrix = torch.from_numpy(factor_matrix).float()

        LR = NUS_FID.unsqueeze(0)
        SR = torch.cat((NOISE_input.real.unsqueeze(0), NOISE_input.imag.unsqueeze(0)), dim=0)
        HR = torch.cat((GT_label.real.unsqueeze(0), GT_label.imag.unsqueeze(0)), dim=0)
        WR = factor_matrix.unsqueeze(0)

        # return {'LR': LR, 'HR': HR, 'SR': SR, 'Index': idx}
        return {'HR': HR, 'Index': idx}
    
    def __gen_signal__(self, idx):
        if self.split == 'train':
            np.random.seed(idx)
        elif self.split == 'valid':
            np.random.seed(idx+self.num_samples)
        
        dim = 1
        N1 = 256  # 第一个维度大小
        max_J = 20  

        # Generate FID signals
        J = np.random.randint(1, 12, size=(dim, 1))  # Random number of harmonics
        mask = np.zeros((dim, max_J))
        mask[np.arange(dim), J.ravel() - 1] = 1  # TODO
        mask = np.cumsum(mask, axis=1)

        ph = np.random.uniform(-np.pi, np.pi, size=(dim, max_J))  # Random phase  # TODO
        A = np.random.uniform(0.05, 1.0, size=(dim, max_J))  # Random amplitude
        w = np.random.uniform(0.01, 0.99, size=(dim, max_J))  # Random frequency
        sgm = np.random.uniform(10, 500, size=(dim, max_J))  # Random relaxation time

        t1 = np.arange(N1)  # Time axis

        A = np.multiply(A, mask)
        x1 = A[..., None] * np.exp(1j * ph[..., None]) * np.exp(-t1 / sgm[..., None]) * np.exp(
        1j * 2 * np.pi * w[..., None] * t1)
        clean_xn = np.sum(x1, axis=1)

        # Add noise to FID signals
        noise_scale = 1e-4
        noise = np.random.normal(loc=0.0, scale=noise_scale, size=(1, N1))
        xx = noise + clean_xn

        def generate_noise(snr, max_val, N1, N2):
            std_noise = max_val / (2 * snr)
            noise_real = np.random.normal(loc=0.0, scale=std_noise, size=(N1, N2))
            noise_imag = np.random.normal(loc=0.0, scale=std_noise, size=(N1, N2))
            noise_out = noise_real + 1j * noise_imag
            return noise_out

        xx_noise = xx + generate_noise(np.random.uniform(15, 80), np.max(np.abs(xx)), 1, N1)

        # Control the range of peak, can be adjusted,change of CB
        threshold = 3
        # Create binary masks for the two dimensions
        temp1 = np.zeros((dim, max_J, N1))

        for i in range(dim):
            for j in range(max_J):
                # For temp1 (Dimension 1)
                freq_indices1 = (np.abs(2 * np.pi * w[i, j] - np.linspace(0, 1, N1) * 2 * np.pi)
                                    <= threshold / 50)
                                    # <= threshold / sgm[i, j])
                temp1[i, j, freq_indices1] = 1

        # Combine binary peak presence masks across dimensions
        A1 = np.multiply(np.ones((dim, max_J)), mask)
        # temp_l1l2 = np.matmul(A1[0][:, np.newaxis, np.newaxis] * temp1[0][:, :, np.newaxis],
        #                 A1[1][:, np.newaxis, np.newaxis] * temp2[1][:, np.newaxis])
        factor_matrix = np.sum(A1[0][:, np.newaxis] * temp1[0], axis=0)
        factor_matrix[factor_matrix > 1] = 1

        # 采样率固定
        result_formatted = np.random.randint(1, 100000)  # 生成1到150000之间的随机整数
        file_name = f"mask_{result_formatted:05d}.mat"
        path = '/mnt/DATA1/shaojie/code/data/0.15_stack_mask-1-256_100000/' + file_name

        Mask = scio.loadmat(path)
        U = Mask['mask'].squeeze()

        NUS_FID = np.multiply(U[np.newaxis, :], xx)
        NOISE_FID = np.multiply(U[np.newaxis, :], xx_noise)
        NOISE_input = np.fft.fft(NOISE_FID)
        GT_label = np.fft.fft(xx)

        return NUS_FID, NOISE_input, GT_label, factor_matrix
            
    def __gen_signal__(self, idx):
        if self.split == 'train':
            np.random.seed(idx)
        elif self.split == 'valid':
            np.random.seed(idx+self.num_samples)

        dim = 1
        N1 = 256
        max_J = 20

        # ============================================================
        # Generate FID signals
        # ============================================================

        J = np.random.randint(1, 12, size=(dim, 1))

        mask = np.zeros((dim, max_J))
        mask[np.arange(dim), J.ravel() - 1] = 1
        mask = np.cumsum(mask, axis=1)


    
def create_dataset(dataset_opt, phase):
    '''create dataset'''
    mode = dataset_opt['mode']
    num_samples = dataset_opt['data_len']
    dataset = NUSNMRDataset(
                num_samples=num_samples,
                split=phase,
                need_LR=(mode == 'LRHR')
                )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset

