import os
import torch
import logging
import random
import numpy as np
import scipy.io as scio
from tqdm import tqdm
from torch.utils.data import Dataset
from data import data_api_1 as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
import os
from data import dataset
import matplotlib.pyplot as plt
import ctypes
import os
import csv
import json
import time
from datetime import datetime

def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def pad_data(data, block_size):
    height, width = data.shape
    pad_height = (block_size - height % block_size) % block_size
    pad_width = (block_size - width % block_size) % block_size
    padded_data = np.pad(data, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)
    return padded_data, pad_height, pad_width

def block_data(data, block_size):
    padded_data, pad_height, pad_width = pad_data(data, block_size)
    height, width = padded_data.shape
    num_blocks_vertical = height // block_size
    num_blocks_horizontal = width // block_size
    blocks = np.empty((num_blocks_vertical * num_blocks_horizontal, block_size, block_size), dtype=padded_data.dtype)
    idx = 0
    for i in range(num_blocks_vertical):
        for j in range(num_blocks_horizontal):
            block = padded_data[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            blocks[idx] = block
            idx += 1
    return blocks, padded_data.shape, pad_height, pad_width

def reconstruct_data(blocks, pad_shape, pad_height, pad_width):
    height, width = pad_shape
    num_blocks, block_height, block_width = blocks.shape
    num_blocks_vertical = height // block_height
    num_blocks_horizontal = width // block_width
    reconstructed_data = np.empty((height, width), dtype=blocks.dtype)
    idx = 0
    for i in range(num_blocks_vertical):
        for j in range(num_blocks_horizontal):
            block = blocks[idx]
            reconstructed_data[i*block_height:(i+1)*block_height, j*block_width:(j+1)*block_width] = block
            idx += 1
    final_data = reconstructed_data[:height if not pad_height else -pad_height, :width if not pad_width else -pad_width]
    return final_data

def reconstruct_3D_slice(ori_fid, mask, diffusion, branch_name, c_batch=8):
    U = mask
    NUS_FID_all = np.multiply(U[:, :, :, np.newaxis], ori_fid)
    NOISE_input_all = np.fft.fftn(NUS_FID_all, axes=(0, 1, 2))
    GT_label_all = np.fft.fftn(ori_fid, axes=(0, 1, 2))

    Nx, Ny, Nz, Nc = ori_fid.shape
    recon_spec = np.zeros((Nx, Ny, Nz, Nc), dtype=np.complex64)
    timing_records = []
    input_size = 64
    sync_cuda()
    branch_start = time.perf_counter()

    for c0 in range(0, Nc, c_batch):
        c1 = min(c0 + c_batch, Nc)
        print("################################iter= {}:{}".format(c0, c1))
        sync_cuda()
        batch_start = time.perf_counter()

        lr_batches = []
        sr_batches = []
        hr_batches = []
        max_amps = []
        block_num = None
        pad_shape = None
        pad_height = None
        pad_width = None

        for c in range(c0, c1):
            NOISE_input = NOISE_input_all[:, :, :, c]
            GT_label = GT_label_all[:, :, :, c]
            NUS_FID = NUS_FID_all[:, :, :, c]

            NUS_FID, cur_pad_shape, cur_pad_height, cur_pad_width = block_data(NUS_FID.reshape(-1, Nz), input_size)
            NOISE_input, _, _, _ = block_data(NOISE_input.reshape(-1, Nz), input_size)
            GT_label, _, _, _ = block_data(GT_label.reshape(-1, Nz), input_size)

            if block_num is None:
                block_num = GT_label.shape[0]
                pad_shape = cur_pad_shape
                pad_height = cur_pad_height
                pad_width = cur_pad_width
            elif block_num != GT_label.shape[0] or pad_shape != cur_pad_shape:
                raise ValueError("Inconsistent block shape while batching c slices.")

            max_amp = np.max(np.abs(NOISE_input))
            max_amps.append(max_amp)
            NUS_FID = torch.complex(torch.from_numpy(NUS_FID.real).float(), torch.from_numpy(NUS_FID.imag).float()) / (max_amp * 4)
            NOISE_input = torch.complex(torch.from_numpy(NOISE_input.real).float(), torch.from_numpy(NOISE_input.imag).float()) / max_amp
            GT_label = torch.complex(torch.from_numpy(GT_label.real).float(), torch.from_numpy(GT_label.imag).float()) / (max_amp * 4)

            lr_batches.append(NUS_FID.unsqueeze(1))
            sr_batches.append(torch.cat((NOISE_input.real.unsqueeze(1), NOISE_input.imag.unsqueeze(1)), dim=1))
            hr_batches.append(torch.cat((GT_label.real.unsqueeze(1), GT_label.imag.unsqueeze(1)), dim=1))

        LR = torch.cat(lr_batches, dim=0)
        SR = torch.cat(sr_batches, dim=0)
        HR = torch.cat(hr_batches, dim=0)
        meta = torch.tensor([
            block_num,
            c1 - c0,
            Nx,
            Ny,
            Nz,
            pad_shape[0],
            pad_shape[1],
            pad_height,
            pad_width,
        ], dtype=torch.long)
        val_data = {'LR': LR, 'HR': HR, 'SR': SR, 'meta': meta}
        print(HR.shape)

        val_data = diffusion.set_device(val_data)
        diffusion.feed_data(val_data)
        diffusion.test(continous=False, pnp_mode='nus3d')
        visuals = diffusion.get_current_visuals()
        total_blocks = block_num * (c1 - c0)
        sr_blocks = torch.complex(
            visuals['SR'][-total_blocks:, 0],
            visuals['SR'][-total_blocks:, 1]).cpu().numpy()

        for local_idx, c in enumerate(range(c0, c1)):
            start = local_idx * block_num
            end = start + block_num
            sr_img = reconstruct_data(sr_blocks[start:end], pad_shape, pad_height, pad_width)
            recon_spec[:, :, :, c] = sr_img.reshape(Nx, Ny, Nz) * (max_amps[local_idx] * 4)

        sync_cuda()
        batch_elapsed = time.perf_counter() - batch_start
        per_slice_elapsed = batch_elapsed / (c1 - c0)
        for c in range(c0, c1):
            timing_records.append({
                'branch': branch_name,
                'slice': int(c),
                'batch_start_slice': int(c0),
                'batch_end_slice_exclusive': int(c1),
                'batch_size': int(c1 - c0),
                'batch_time_sec': float(batch_elapsed),
                'slice_time_sec': float(per_slice_elapsed),
            })
        print("{} slices {}:{} batch_time={:.3f}s avg_slice_time={:.3f}s".format(
            branch_name, c0, c1, batch_elapsed, per_slice_elapsed))

    sync_cuda()
    branch_elapsed = time.perf_counter() - branch_start
    return recon_spec, {
        'branch': branch_name,
        'total_time_sec': float(branch_elapsed),
        'num_slices': int(Nc),
        'c_batch': int(c_batch),
        'slice_records': timing_records,
    }

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_64_64.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'], 
                        help='Run either train(training) or val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    parser.add_argument('--c_batch', type=int, default=8,
                        help='Number of c slices reconstructed in one diffusion call')

    args =parser.parse_known_args()[0]
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))

    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    np.random.seed(60)

    data = scio.loadmat('data/Hnoesy_4D.mat')['data']
    mask = scio.loadmat('data/Hnoesy_4D_Mask_0.1.mat')['mask']
    FID_Ideal = data

    N1, N2, N3, num_block = data.shape
    N1 = N1 // 2  
    N2 = N2 // 2
    N3 = N3 // 2
    N4 = num_block

    X_temp = np.zeros((N1, N2, N3, N4))
    X1_temp = np.zeros((N1, N2, N3, num_block))
    X2_temp = np.zeros((N1, N2, N3, num_block))
    X3_temp = np.zeros((N1, N2, N3, num_block))
    X4_temp = np.zeros((N1, N2, N3, num_block))
    X5_temp = np.zeros((N1, N2, N3, num_block))
    X6_temp = np.zeros((N1, N2, N3, num_block))
    X7_temp = np.zeros((N1, N2, N3, num_block))
    X8_temp = np.zeros((N1, N2, N3, num_block))

    F1 = np.squeeze(FID_Ideal[0, 0, 0, :])  
    F1 = np.abs(F1 / np.max(F1))

    plt.figure(1)
    plt.plot(F1)

    RRR = FID_Ideal[0:2*N1:2, 0:2*N2:2, 0:2*N3:2, F1>0.01]  
    RIR = FID_Ideal[0:2*N1:2, 1:2*N2:2, 0:2*N3:2, F1>0.01]  
    IRR = FID_Ideal[1:2*N1:2, 0:2*N2:2, 0:2*N3:2, F1>0.01]
    IIR = FID_Ideal[1:2*N1:2, 1:2*N2:2, 0:2*N3:2, F1>0.01]
    RRI = FID_Ideal[0:2*N1:2, 0:2*N2:2, 1:2*N3:2, F1>0.01]
    RII = FID_Ideal[0:2*N1:2, 1:2*N2:2, 1:2*N3:2, F1>0.01]
    IRI = FID_Ideal[1:2*N1:2, 0:2*N2:2, 1:2*N3:2, F1>0.01]
    III = FID_Ideal[1:2*N1:2, 1:2*N2:2, 1:2*N3:2, F1>0.01]

    fid1 = RRR + 1j*RRI
    fid2 = RIR + 1j*RII
    fid3 = IRR + 1j*IRI
    fid4 = IIR + 1j*III

    os.makedirs('results', exist_ok=True)
    run_tag = '0.1_cbatch{}_{}'.format(
        args.c_batch,
        datetime.now().strftime('%Y%m%d_%H%M%S'))
    all_timing = {
        'c_batch': int(args.c_batch),
        'run_tag': run_tag,
        'branches': [],
        'slice_records': [],
    }

    recon_spec1, timing = reconstruct_3D_slice(fid1, mask, diffusion, 'fid1', c_batch=args.c_batch)
    all_timing['branches'].append(timing)
    all_timing['slice_records'].extend(timing['slice_records'])
    np.save('results/recon_spec1_{}.npy'.format(run_tag), recon_spec1)

    recon_spec2, timing = reconstruct_3D_slice(fid2, mask, diffusion, 'fid2', c_batch=args.c_batch)
    all_timing['branches'].append(timing)
    all_timing['slice_records'].extend(timing['slice_records'])
    np.save('results/recon_spec2_{}.npy'.format(run_tag), recon_spec2)

    recon_spec3, timing = reconstruct_3D_slice(fid3, mask, diffusion, 'fid3', c_batch=args.c_batch)
    all_timing['branches'].append(timing)
    all_timing['slice_records'].extend(timing['slice_records'])
    np.save('results/recon_spec3_{}.npy'.format(run_tag), recon_spec3)

    recon_spec4, timing = reconstruct_3D_slice(fid4, mask, diffusion, 'fid4', c_batch=args.c_batch)
    all_timing['branches'].append(timing)
    all_timing['slice_records'].extend(timing['slice_records'])
    np.save('results/recon_spec4_{}.npy'.format(run_tag), recon_spec4)

    sync_cuda()
    all_timing['total_reconstruction_time_sec'] = float(
        sum(item['total_time_sec'] for item in all_timing['branches']))
    timing_json_path = 'results/reconstruction_timing_{}.json'.format(run_tag)
    timing_csv_path = 'results/reconstruction_timing_{}.csv'.format(run_tag)
    with open(timing_json_path, 'w') as f:
        json.dump(all_timing, f, indent=2)
    with open(timing_csv_path, 'w', newline='') as f:
        fieldnames = [
            'branch',
            'slice',
            'batch_start_slice',
            'batch_end_slice_exclusive',
            'batch_size',
            'batch_time_sec',
            'slice_time_sec',
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_timing['slice_records'])
    print("Total reconstruction time: {:.3f}s".format(all_timing['total_reconstruction_time_sec']))
    print("Run tag: {}".format(run_tag))
    print("Timing saved to {} and {}".format(timing_json_path, timing_csv_path))



   
