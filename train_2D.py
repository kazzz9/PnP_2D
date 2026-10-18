import torch
from data import data_api_1 as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
import matplotlib.pyplot as plt
from data.dataset_2D import * 


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_64_64.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'], 
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = create_dataset(dataset_opt, 'valid')
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

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
    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                if current_step % opt['train']['val_freq'] == 0:
                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    sample_sum = opt['datasets']['val']['data_len']
                    for idx in range(sample_sum):
                        diffusion.sample(continous=False)
                        visuals = diffusion.get_current_visuals(sample=True)
                        sample_img = to_numpy(visuals['SAM'])

                        if sample_img.ndim == 4:
                            sample_img = sample_img[0]
                        if sample_img.ndim == 3 and sample_img.shape[0] >= 2:
                            real_part = sample_img[0]
                            imag_part = sample_img[1]
                        elif sample_img.ndim == 2:
                            real_part = sample_img
                            imag_part = np.zeros_like(sample_img)
                        else:
                            raise ValueError(f'Unexpected generated sample shape: {sample_img.shape}')

                        magnitude = np.sqrt(real_part ** 2 + imag_part ** 2)

                        fig = plt.figure(figsize=(15, 4))
                        ax = fig.add_subplot(131)
                        im = ax.contour(np.abs(real_part), levels=50, cmap='viridis')
                        ax.set_title('Generated SR Real')
                        ax.set_xlabel('X')
                        ax.set_ylabel('Y')
                        plt.colorbar(im, ax=ax, label='Amplitude')

                        ax = fig.add_subplot(132)
                        im = ax.contour(np.abs(imag_part), levels=50, cmap='viridis')
                        ax.set_title('Generated SR Imag')
                        ax.set_xlabel('X')
                        ax.set_ylabel('Y')
                        plt.colorbar(im, ax=ax, label='Amplitude')

                        ax = fig.add_subplot(133)
                        im = ax.contour(magnitude, levels=50, cmap='viridis')
                        ax.set_title('Generated SR Magnitude')
                        ax.set_xlabel('X')
                        ax.set_ylabel('Y')
                        plt.colorbar(im, ax=ax, label='Amplitude')

                        plt.tight_layout()
                        save_path = os.path.join(
                            result_path, f'sample_step{current_step}_{idx}_sr.png')
                        plt.savefig(save_path, dpi=150, bbox_inches='tight')
                        plt.close(fig)
                        logger.info('Saved generated SR sample: {}'.format(save_path))

                        if wandb_logger:
                            wandb_logger.log_image(f'validation_{idx}', magnitude)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    logger.info('# Validation # generated {} unconditional samples'.format(sample_sum))
                    logger_val = logging.getLogger('val')
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> generated samples: {:d}'.format(
                        current_epoch, current_step, sample_sum))

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')
        avg_psnr = 0.0
        avg_ssim = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for _,  val_data in enumerate(val_loader):
            idx += 1
            diffusion.feed_data(val_data)
            diffusion.test(continous=True)
            visuals = diffusion.get_current_visuals()

            bsz = visuals['HR'].shape[0]
            for i in range(bsz):
                hr_img = Metrics.tensor2img(visuals['HR'][i])
                lr_img = Metrics.tensor2img(visuals['LR'][i])
                fake_img = Metrics.tensor2img(visuals['INF'][i])
                sr_img = Metrics.tensor2img(visuals['SR'][-bsz+i])

                Metrics.save_img(
                    hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, i+1))
                Metrics.save_img(
                    lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, i+1))
                Metrics.save_img(
                    sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, i+1))

                eval_psnr = Metrics.calculate_psnr(sr_img, hr_img)
                eval_ssim = Metrics.calculate_ssim(sr_img, hr_img)

                avg_psnr += eval_psnr
                avg_ssim += eval_ssim

        avg_psnr = avg_psnr / (idx*bsz)
        avg_ssim = avg_ssim / (idx*bsz)

        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        logger_val = logging.getLogger('val')
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim: {:.4e}'.format(
            current_epoch, current_step, avg_psnr, avg_ssim))
