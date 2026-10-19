import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
import os
from scipy.signal import find_peaks


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size_1,
        image_size_2,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size_1 = image_size_1
        self.image_size_2 = image_size_2
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        if schedule_opt is not None:
            pass



    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)


    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))


    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped


    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level))
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance, x_recon

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        model_mean, model_log_variance, _ = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()


    @torch.no_grad()
    def _p_sample_loop_dosy_lowrank(self, x_in, nus_fid, continous=False,
                                    lam_l1=None, lam_lr=None, norm_spec=None,
                                    cs1=6500, cs2=7700):
        """DOSY low-rank PnP-ADMM reconstruction implementation."""
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps // 10))

        if (
            torch.is_tensor(x_in)
            and torch.is_tensor(nus_fid)
            and x_in.ndim == 2
            and torch.is_complex(x_in)
        ):
            lam_l1 = 0.03  if lam_l1 is None else (lam_l1.item() if torch.is_tensor(lam_l1) else float(lam_l1))
            lam_lr = 0.048 if lam_lr is None else (lam_lr.item() if torch.is_tensor(lam_lr) else float(lam_lr))

            X_obs_crop = x_in.to(device)               
            X_obs_full = nus_fid.to(device)          
            norm_spec  = norm_spec.to(device).squeeze() 
            num_iter      = 100
            rho           = 0.1
            admm_iters    = 5
            sparse_domain = 'freq'

            if (
                not hasattr(self, 'admm_z1_dosy')
                or self.admm_z1_dosy.shape != X_obs_full.shape
            ):
                self.admm_z1_dosy = X_obs_full.clone().detach()   # (32, 16384)
                self.admm_z2_dosy = X_obs_full.clone().detach()   # (32, 16384)
                self.admm_u1_dosy = torch.zeros_like(X_obs_full)
                self.admm_u2_dosy = torch.zeros_like(X_obs_full)

            z1 = self.admm_z1_dosy
            z2 = self.admm_z2_dosy
            u1 = self.admm_u1_dosy
            u2 = self.admm_u2_dosy

            ret_img = X_obs_crop.clone()

            for i in tqdm(reversed(range(0, num_iter)),
                        desc='whole DOSY sampling', total=num_iter):

                for _ in range(admm_iters):

                    # (A) x-update
                    x_admm = (
                        X_obs_full + rho * (z1 - u1) + rho * (z2 - u2)
                    ) / (1.0 + 2.0 * rho)           
                    # (B1) L1 稀疏
                    if sparse_domain == 'freq':
                        mag    = torch.abs(x_admm + u1)
                        phase  = torch.angle(x_admm + u1)
                        thresh = lam_l1 / rho
                        x_sparse = torch.clamp(mag - thresh, min=0.0) * torch.exp(1j * phase)
                    else:
                        T      = x_admm + u1
                        mag    = torch.abs(T)
                        phase  = torch.angle(T)
                        thresh = lam_l1 / rho
                        x_sparse = torch.clamp(mag - thresh, min=0.0) * torch.exp(1j * phase)
                    # (B2) z1-update：仅在窗口内做 SVT
                    z1 = x_sparse.clone()
                    z1[:, cs1:cs2] = low_rank_svt(x_sparse[:, cs1:cs2], lam_lr / rho)
                    # (B3) z2-update

                    x_for_diff_full = x_admm + u2               
                    x_for_diff_win  = x_for_diff_full[:, cs1:cs2]  

                    z_rows = []
                    for k in range(x_for_diff_win.shape[0]):
                        row_1d = x_for_diff_win[k]               
                        row_2d = row_1d.unsqueeze(-1) * norm_spec.unsqueeze(0) 
                        row_blocks, pad_shape, pad_height, pad_width = \
                            block_data_ddpm(row_2d, 256)
                        row_blocks = row_blocks.unsqueeze(1)
                        tmp_2ch    = torch.cat(
                            (row_blocks.real, row_blocks.imag), dim=1
                        )

                        z_denoised = self.p_sample(tmp_2ch, i)
                        z_freq_c   = torch.complex(z_denoised[:, 0], z_denoised[:, 1])
                        z_freq_2d  = reconstruct_data_ddpm(
                            z_freq_c, pad_shape, pad_height, pad_width
                        )
                        row_rec_np = pseudo2d_to_1d(
                            z_freq_2d.detach().cpu().numpy(),
                            norm_spec.detach().cpu().numpy()
                        )
                        row_rec = torch.from_numpy(np.asarray(row_rec_np)).to(device)
                        if not torch.is_complex(row_rec):
                            row_rec = torch.complex(
                                row_rec.float(), torch.zeros_like(row_rec.float())
                            )
                        z_rows.append(row_rec)

                    z_win = torch.stack(z_rows, dim=0)           

                    z2 = x_for_diff_full.clone()
                    z2[:, cs1:cs2] = z_win                     

                    u1 = u1 + x_admm - z1
                    u2 = u2 + x_admm - z2

                if i % sample_inter == 0:
                    ret_img = x_admm[:, cs1:cs2].clone()         

            self.admm_z1_dosy = z1.detach()
            self.admm_z2_dosy = z2.detach()
            self.admm_u1_dosy = u1.detach()
            self.admm_u2_dosy = u2.detach()

            self.x_admm_full = x_admm.detach()               

            if continous:
                return ret_img
            else:
                return x_admm[:, cs1:cs2]


    @torch.no_grad()
    def _p_sample_loop_conditional(self, x_in, fid):
        """Conditional diffusion reconstruction."""
        device = self.betas.device
        sample_inter = 1 | self.num_timesteps // 10

        x = x_in
        img = torch.randn(x.shape, device=device)
        ret_img = x

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            gamma = 0.1
            lam_l1 = 0.002
            sparse_domain = 'freq'

            # Data consistency in the time domain.
            img_c = torch.complex(img[:, 0], img[:, 1]).unsqueeze(1)
            fid_est = torch.fft.ifft(img_c, dim=-2)
            mask = (fid != 0).to(fid_est.dtype)
            inv_gamma = 1.0 / gamma
            fid_dc = (fid_est + inv_gamma * (mask * fid)) / (1.0 + inv_gamma * mask)

            # Sparse prior.
            if sparse_domain == 'freq':
                freq_c = torch.fft.fft(fid_dc, dim=-2)
                mag = torch.abs(freq_c)
                phase = torch.angle(freq_c)
                mag_thr = torch.clamp(mag - lam_l1, min=0.0)
                freq_sparse = mag_thr * torch.exp(1j * phase)
                fid_dc = torch.fft.ifft(freq_sparse, dim=-2)
            else:
                mag = torch.abs(fid_dc)
                phase = torch.angle(fid_dc)
                mag_thr = torch.clamp(mag - lam_l1, min=0.0)
                fid_dc = mag_thr * torch.exp(1j * phase)

            # Diffusion denoising.
            img_freq = torch.fft.fft(fid_dc, dim=-2)
            img = torch.concat((img_freq.real, img_freq.imag), dim=1)
            img = self.p_sample(img, i, condition_x=x)

            if i % sample_inter == 0:
                ret_img = torch.cat([ret_img, img], dim=0)

        return ret_img


    @torch.no_grad()
    def _p_sample_loop_ca(self, x_in, fid):
        """CA reconstruction with independent in-phase and anti-phase states."""
        device = self.betas.device
        sample_inter = 1 | self.num_timesteps // 10
        img = torch.randn(x_in.shape, device=device)
        ret_img = x_in
        num_iter = 100

        for i in tqdm(reversed(range(num_iter)), desc='ca 2D PnP sampling', total=num_iter):

            rho = 0.8
            lam_l1 = 0.03
            admm_iters = 3
            J = 53.0
            dwell_time = 1 / 3000

            pad_shape = [256, 512]
            pad_height = 0
            pad_width = 69

            img = torch.complex(img[:, 0], img[:, 1])
            img = reconstruct_data_ddpm(img, pad_shape, pad_height, pad_width)

            fid_est = torch.fft.ifft(img, dim=-1).unsqueeze(0)
            noise_fid = reconstruct_data_ddpm(fid.squeeze(), pad_shape, pad_height, pad_width)
            
            B, N1, N2 = fid_est.shape
            t2 = torch.arange(N2, device=fid_est.device) * dwell_time
            c = torch.cos(np.pi * J * t2).view(1, 1, N2)
            s = torch.sin(np.pi * J * t2).view(1, 1, N2)

            y_in = noise_fid.real
            y_anti = noise_fid.imag

            if not hasattr(self, 'admm_z_in') or self.admm_z_in.shape != y_in.shape:
                self.admm_z_in = y_in.clone().detach()
                self.admm_z_anti = y_anti.clone().detach()
                self.admm_u_in = torch.zeros_like(y_in)
                self.admm_u_anti = torch.zeros_like(y_anti)
            z_in, z_anti = (self.admm_z_in, self.admm_z_anti)
            u_in, u_anti = (self.admm_u_in, self.admm_u_anti)


            for _ in range(admm_iters):
                x_in = (c * y_in + rho * (z_in - u_in)) / (c * c + rho)
                x_anti = (s * y_anti + rho * (z_anti - u_anti)) / (s * s + rho)

                Xin_f = torch.fft.fft(x_in + u_in, dim=-1)
                Xanti_f = torch.fft.fft(x_anti + u_anti, dim=-1)

                mag_in, ph_in = (torch.abs(Xin_f), torch.angle(Xin_f))
                mag_anti, ph_anti = (torch.abs(Xanti_f), torch.angle(Xanti_f))

                thresh = lam_l1 / rho
                Z_in_f = torch.clamp(mag_in - thresh, min=0.0) * torch.exp(1j * ph_in)
                Z_anti_f = torch.clamp(mag_anti - thresh, min=0.0) * torch.exp(1j * ph_anti)
                z_in_sparse = torch.fft.ifft(Z_in_f, dim=-1)
                z_anti_sparse = torch.fft.ifft(Z_anti_f, dim=-1)

                z_complex = torch.complex(z_in_sparse, z_anti_sparse)
                z_freq = torch.fft.fft(z_complex, dim=-1).squeeze()

                z_freq, _, _, _ = block_data_ddpm(z_freq, 256)
                z_freq = z_freq.unsqueeze(1)
                z_2ch = torch.cat((z_freq.real, z_freq.imag), dim=1)
                z_denoised = self.p_sample(z_2ch, i)
                z_freq_c = torch.complex(z_denoised[:, 0], z_denoised[:, 1])
                z_freq_c = reconstruct_data_ddpm(z_freq_c, pad_shape, pad_height, pad_width)
                z_time = torch.fft.ifft(z_freq_c, dim=-1)
                z_in = z_time.real
                z_anti = z_time.imag

                u_in = u_in + x_in - z_in
                u_anti = u_anti + x_anti - z_anti

            self.admm_z_in = z_in.detach()
            self.admm_z_anti = z_anti.detach()
            self.admm_u_in = u_in.detach()
            self.admm_u_anti = u_anti.detach()

            x_complex = torch.complex(x_in, x_anti)
            img_freq = torch.fft.fft(x_complex, dim=-1).squeeze()
            img_freq, _, _, _ = block_data_ddpm(img_freq, 256)
            img_freq = img_freq.unsqueeze(1)
            img = torch.cat((img_freq.real, img_freq.imag), dim=1)

            if i % sample_inter == 0:
                ret_img = torch.cat([ret_img, img], dim=0)

        save_dir = './convergence_logs'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'admm_convergence.npz')
        np.savez(save_path, delta=np.array([]), primal=np.array([]), dual=np.array([]))
        print('Convergence data saved to:', save_path)
        return ret_img

    @torch.no_grad()
    def _p_sample_loop_nus(self, x_in, fid, pad_shape=None,
                           pad_height=None, pad_width=None):
        device = self.betas.device
        sample_inter = 1 | self.num_timesteps // 10
        img = torch.randn(x_in.shape, device=device)
        ret_img = x_in
        num_iter = 100
        x_nus = fid

        # Explicit notebook geometry takes precedence over legacy environment defaults.
        if pad_shape is None:
            pad_shape = (
                int(os.environ.get('NUS_PAD_HEIGHT', '256')),
                int(os.environ.get('NUS_PAD_WIDTH', '768')),
            )
        if pad_height is None:
            pad_height = int(os.environ.get('NUS_CROP_HEIGHT', '0'))
        if pad_width is None:
            pad_width = int(os.environ.get('NUS_CROP_WIDTH', '30'))
        if (len(pad_shape) != 2 or any(v <= 0 for v in pad_shape)
                or not 0 <= pad_height < pad_shape[0]
                or not 0 <= pad_width < pad_shape[1]):
            raise ValueError('Invalid padding geometry')
        block_h, block_w = x_in.shape[-2:]
        if (pad_shape[0] % block_h or pad_shape[1] % block_w
                or (pad_shape[0] // block_h) * (pad_shape[1] // block_w) != x_in.shape[0]):
            raise ValueError('Padding geometry does not match the input blocks')

        for i in tqdm(reversed(range(num_iter)), desc='nus 2D PnP sampling', total=num_iter):

            rho = float(os.environ.get('NUS_RHO', '0.01'))
            lam_l1 = float(os.environ.get('NUS_LAM_L1', '0.0002'))
            admm_iters = int(os.environ.get('NUS_ADMM_ITERS', '3'))
            sparse_domain = 'freq'


            img = torch.complex(img[:, 0], img[:, 1])
            img = reconstruct_data_ddpm(
                img, pad_shape, pad_height, pad_width
            )
            fid_est = torch.fft.ifft(img, dim=-2)
            nus_fid = reconstruct_data_ddpm(
                x_nus.squeeze(1), pad_shape, pad_height, pad_width
            )

            mask = (nus_fid != 0).to(fid_est.dtype)
            HTy = mask * nus_fid
            denom = mask + 2 * rho

            if (
                not hasattr(self, 'admm_z1')
                or not hasattr(self, 'admm_z2')
                or self.admm_z1.shape != fid_est.shape
                or self.admm_z2.shape != fid_est.shape
            ):
                self.admm_z1 = fid_est.clone().detach()
                self.admm_z2 = fid_est.clone().detach()
                self.admm_u1 = torch.zeros_like(fid_est)
                self.admm_u2 = torch.zeros_like(fid_est)
            z1 = self.admm_z1
            z2 = self.admm_z2
            u1 = self.admm_u1
            u2 = self.admm_u2
            x_admm = None

            for _ in range(admm_iters):
                if x_admm is None:
                    x_prev = fid_est.clone().detach()
                    z1_prev = z1.clone().detach()
                    z2_prev = z2.clone().detach()
                else:
                    x_prev = x_admm.clone().detach()
                    z1_prev = z1.clone().detach()
                    z2_prev = z2.clone().detach()

                x_admm = (HTy + rho * (z1 - u1) + rho * (z2 - u2)) / denom

                rank = int(os.environ.get('NUS_RANK', '10'))
                fid_temp = (x_admm + u1).unsqueeze(0)
                B, H, W = fid_temp.shape

                fid_temp = fid_temp.permute(0, 2, 1).reshape(-1, H)
                fid_temp = hankel_lowrank_batch(
                    fid_temp,
                    rank=rank,
                    L=int(os.environ.get('NUS_HANKEL_L', '128'))
                )
                fid_temp = fid_temp.reshape(B, W, H).permute(0, 2, 1)
                z_lowrank = fid_temp.squeeze(0)

                if sparse_domain == 'freq':
                    Xf = torch.fft.fft(z_lowrank, dim=-2)
                    mag, phase = (torch.abs(Xf), torch.angle(Xf))
                    thresh = lam_l1 / rho
                    mag_thr = torch.clamp(mag - thresh, min=0.0)
                    Zf = mag_thr * torch.exp(1j * phase)
                    z1 = torch.fft.ifft(Zf, dim=-2)
                else:
                    mag, phase = (torch.abs(z_lowrank), torch.angle(z_lowrank))
                    thresh = lam_l1 / rho
                    mag_thr = torch.clamp(mag - thresh, min=0.0)
                    z1 = mag_thr * torch.exp(1j * phase)

                tmp_freq = torch.fft.fft(x_admm + u2, dim=-2)
                tmp_freq, _, _, _ = block_data_ddpm(tmp_freq, 256)
                tmp_freq = tmp_freq.unsqueeze(1)
                tmp_2ch = torch.concat((tmp_freq.real, tmp_freq.imag), dim=1)
                z_denoised = self.p_sample(tmp_2ch, i)
                z_freq_c = torch.complex(z_denoised[:, 0], z_denoised[:, 1])
                z_freq_c = reconstruct_data_ddpm(
                    z_freq_c, pad_shape, pad_height, pad_width
                )
                z2 = torch.fft.ifft(z_freq_c, dim=-2)

                u1 = u1 + x_admm - z1
                u2 = u2 + x_admm - z2

            self.admm_z1 = z1.detach()
            self.admm_z2 = z2.detach()
            self.admm_u1 = u1.detach()
            self.admm_u2 = u2.detach()

            img_freq = torch.fft.fft(x_admm, dim=-2)
            img_freq, _, _, _ = block_data_ddpm(img_freq, 256)
            img_freq = img_freq.unsqueeze(1)
            img = torch.concat((img_freq.real, img_freq.imag), dim=1)

            if i % sample_inter == 0:
                ret_img = torch.cat([ret_img, img], dim=0)

        return ret_img


    @torch.no_grad()
    def _p_sample_loop_single_prior(
        self, x_in, fid, continous=False, label=None, meta=None,
        pnp_mode=None, lam_l1=None, lam_lr=None, norm_spec=None,
        cs1=6500, cs2=7700,
    ):

        device = self.betas.device
        sample_inter = 1 | self.num_timesteps // 10
        nus_fid = fid
        x_nus = nus_fid
        x = x_in
        shape = x.shape
        img = torch.randn(shape, device=device)
        ret_img = x
        num_iter = 100

        batch_meta = None
        if meta is not None:
            meta_values = meta.detach().long().cpu().tolist()
            batch_meta = {'block_num': meta_values[0], 'num_c': meta_values[1], 'N1': meta_values[2], 'N2': meta_values[3], 'N3': meta_values[4], 'pad_shape': [meta_values[5], meta_values[6]], 'pad_height': meta_values[7], 'pad_width': meta_values[8]}

        conv_delta = []
        primal_residual = []
        dual_residual = []

        for i in tqdm(reversed(range(num_iter)), desc=f'{pnp_mode} 2D PnP sampling', total=num_iter):
 

            # ---- vd_1d -------------------------------------------------
            if pnp_mode == 'vd_1d':

                rho = 0.5
                lam_l1 = 0.0001
                admm_iters = 10
                sparse_domain = 'freq'

                J = 9.45
                dwell_time = 1 / 50000
                cs1 = 48500
                cs2 = 49800

                if i == num_iter - 1:
                    img = x_in
                pad_shape = [256, 1536]
                pad_height = 0
                pad_width = 236
                img = torch.complex(img[:, 0], img[:, 1])
                img = reconstruct_data_ddpm(img, pad_shape, pad_height, pad_width)
                X_full = torch.fft.fft(nus_fid, dim=-1)
                X_full[:, :, cs1:cs2] = img
                fid_est = torch.fft.ifft(X_full, dim=-1)
                noise_fid = x_nus
                B, N1, N2 = fid_est.shape
                t2 = torch.arange(N2).to(fid_est.device) * dwell_time
                c = torch.cos(np.pi * J * t2)
                c = c.reshape(1, 1, N2)
                HTy = c * noise_fid
                denom = c * c + rho
                fft_dims = (-1,)
                block_layout = 'window'

            # ---- vd restruction----------------------------------------------------
            elif pnp_mode == 'vd':

                rho = 0.8
                lam_l1 = 0.03
                admm_iters = 3
                sparse_domain = 'freq'
                J = 53
                dwell_time = 1 / 3000
                pad_shape = [256, 512]
                pad_height = 0
                pad_width = 69
                img = torch.complex(img[:, 0], img[:, 1])
                img = reconstruct_data_ddpm(img, pad_shape, pad_height, pad_width)
                fid_est = torch.fft.ifft(img, dim=-1).unsqueeze(0)
                noise_fid = reconstruct_data_ddpm(x_nus.squeeze(), pad_shape, pad_height, pad_width)
                B, N1, N2 = fid_est.shape
                t2 = torch.arange(N2).to(fid_est.device) * dwell_time
                c = torch.cos(np.pi * J * t2)
                c = c.reshape(1, 1, N2)
                HTy = c * noise_fid
                denom = c * c + rho
                fft_dims = (-1,)
                block_layout = 'blocks'

            # ---- ps ----------------------------------------------------
            # PS 沿 -2 维执行一维 FFT，与其 FID 观测方向保持一致。
            elif pnp_mode == 'ps':
                rho = 0.07
                lam_l1 = 0.009
                admm_iters = 3
                sparse_domain = 'freq'
                img = torch.complex(img[:, 0], img[:, 1])
                pad_shape = [
                    img.shape[0] * img.shape[-2], img.shape[-1]
                ]
                pad_height = 0
                pad_width = 0
                img = reconstruct_data_ddpm(
                    img, pad_shape, pad_height, pad_width
                )
                fid_est = torch.fft.ifft(img, dim=-2)
                nus_fid = reconstruct_data_ddpm(
                    x_nus.squeeze(1), pad_shape, pad_height, pad_width
                )
                mask = (nus_fid != 0).to(fid_est.dtype)
                HTy = mask * nus_fid
                denom = mask + rho
                fft_dims = (-2,)
                block_layout = 'blocks'

            # ---- nus3d -------------------------------------------------
            # 完整三维体模式：还原体数据后，对三个空间维度做 FFT。
            elif pnp_mode == 'nus3d':
                rho = 0.02
                lam_l1 = 0.0001
                admm_iters = 5
                sparse_domain = 'freq'
                if batch_meta is None:
                    batch_meta = {'block_num': img.shape[0], 'num_c': 1, 'N1': 15, 'N2': 18, 'N3': 58, 'pad_shape': [320, 64], 'pad_height': 50, 'pad_width': 6}
                N1 = batch_meta['N1']
                N2 = batch_meta['N2']
                N3 = batch_meta['N3']
                img = torch.complex(img[:, 0], img[:, 1])
                img = reconstruct_3d_batches(img, batch_meta)
                fid_est = torch.fft.ifftn(img, dim=(-1, -2, -3))
                nus_fid = reconstruct_3d_batches(x_nus.squeeze(1), batch_meta)
                mask = (nus_fid != 0).to(fid_est.dtype)
                HTy = mask * nus_fid
                denom = mask + rho
                fft_dims = (-1, -2, -3)
                block_layout = 'volume'


            if pnp_mode in ('vd_1d', 'vd', 'ps', 'nus3d'):
                if pnp_mode == 'nus3d' and i == num_iter - 1 or not hasattr(self, 'admm_z') or self.admm_z.shape != fid_est.shape:
                    self.admm_z = fid_est.clone().detach()
                    self.admm_u = torch.zeros_like(fid_est)
                z = self.admm_z
                u = self.admm_u

            for _ in range(admm_iters):

                x_admm = (HTy + rho * (z - u)) / denom

                if sparse_domain == 'freq':
                    if block_layout == 'volume':
                        Xf = torch.fft.fftn(x_admm + u, dim=fft_dims)
                    else:
                        Xf = x_admm + u
                        for dim in fft_dims:
                            Xf = torch.fft.fft(Xf, dim=dim)
                    mag, phase = (torch.abs(Xf), torch.angle(Xf))
                    thresh = lam_l1 / rho
                    mag_thr = torch.clamp(mag - thresh, min=0.0)
                    Zf = mag_thr * torch.exp(1j * phase)
                    if block_layout == 'volume':
                        z_sparse = torch.fft.ifftn(Zf, dim=fft_dims)
                    else:
                        z_sparse = Zf
                        for dim in reversed(fft_dims):
                            z_sparse = torch.fft.ifft(z_sparse, dim=dim)
                else:
                    T = x_admm + u
                    mag, phase = (torch.abs(T), torch.angle(T))
                    thresh = lam_l1 / rho
                    mag_thr = torch.clamp(mag - thresh, min=0.0)
                    z_sparse = mag_thr * torch.exp(1j * phase)

  
                if block_layout == 'volume':
                    tmp_freq = torch.fft.fftn(z_sparse, dim=fft_dims)
                else:
                    tmp_freq = z_sparse
                    for dim in fft_dims:
                        tmp_freq = torch.fft.fft(tmp_freq, dim=dim)
                if block_layout == 'window':
                    tmp_freq, _, _, _ = block_data_ddpm(
                        tmp_freq.squeeze()[:, cs1:cs2], 256
                    )
                elif block_layout == 'blocks':
                    tmp_freq, _, _, _ = block_data_ddpm(
                        tmp_freq.squeeze(), 256
                    )
                elif block_layout == 'volume':
                    tmp_freq = block_3d_batches(tmp_freq, batch_meta)
                tmp_freq = tmp_freq.unsqueeze(1)
                tmp_2ch = torch.concat(
                    (tmp_freq.real, tmp_freq.imag), dim=1
                )


                z_denoised = self.p_sample(tmp_2ch, i)
                z_freq_c = torch.complex(
                    z_denoised[:, 0], z_denoised[:, 1]
                )

                if block_layout in ('window', 'blocks'):
                    z_freq_c = reconstruct_data_ddpm(
                        z_freq_c, pad_shape, pad_height, pad_width
                    )
                elif block_layout == 'volume':
                    z_freq_c = reconstruct_3d_batches(
                        z_freq_c, batch_meta
                    )
                if block_layout == 'window':
                    X_full[:, :, cs1:cs2] = z_freq_c
                    z_freq_c = X_full
                if block_layout == 'volume':
                    z = torch.fft.ifftn(z_freq_c, dim=fft_dims)
                else:
                    z = z_freq_c
                    for dim in reversed(fft_dims):
                        z = torch.fft.ifft(z, dim=dim)

                u = u + x_admm - z


            self.admm_z = z.detach()
            self.admm_u = u.detach()

            img_freq = x_admm
            if block_layout == 'volume':
                img_freq = torch.fft.fftn(img_freq, dim=fft_dims)
            else:
                for dim in fft_dims:
                    img_freq = torch.fft.fft(img_freq, dim=dim)
            if block_layout == 'window':
                img_freq, _, _, _ = block_data_ddpm(
                    img_freq.squeeze()[:, cs1:cs2], 256
                )
            elif block_layout == 'blocks':
                img_freq, _, _, _ = block_data_ddpm(
                    img_freq.squeeze(), 256
                )
            elif block_layout == 'volume':
                img_freq = block_3d_batches(img_freq, batch_meta)
            img_freq = img_freq.unsqueeze(1)
            img = torch.concat((img_freq.real, img_freq.imag), dim=1)

            if i % sample_inter == 0:
                ret_img = torch.cat([ret_img, img], dim=0)

        save_dir = './convergence_logs'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'admm_convergence.npz')
        np.savez(save_path, delta=np.array(conv_delta), primal=np.array(primal_residual), dual=np.array(dual_residual))
        print('Convergence data saved to:', save_path)
        return ret_img


    # ===================== 64x64 slice reconstruction: Hankel + L1 and diffusion =====================
    @torch.no_grad()
    def _p_sample_loop_nus_slice_64(self, x_in, fid, continous=False):
  
        height = 40
        width = 64
        pad_shape = [64, 64]
        pad_height = 24
        pad_width = 0

        rho = 0.1
        lam_l1 = 1e-3
        admm_iters = 5

        use_lowrank = True
        hankel_rank = 10
        hankel_L = 20
        num_iter = 100
  
        img = torch.randn_like(x_in)
        img_c = torch.complex(img[:, 0], img[:, 1])
        img_c = reconstruct_batch_ddpm(img_c, pad_shape, pad_height, pad_width)
        observed = reconstruct_batch_ddpm(fid[:, 0], pad_shape, pad_height, pad_width)


        mask = (observed != 0).to(observed.dtype)
        hty = mask * observed

        denom = mask + 2.0 * rho

        z1 = torch.fft.ifft2(img_c, dim=(-2, -1))
        z2 = z1.clone()
        u1, u2 = torch.zeros_like(z1), torch.zeros_like(z1)
        sample_inter = 1 | (self.num_timesteps // 10)
        history = [x_in] if continous else None

        for i in tqdm(reversed(range(num_iter)), desc='nus_slice_64 PnP sampling', total=num_iter):
            for _ in range(admm_iters):

                x_admm = (hty + rho * (z1 - u1) + rho * (z2 - u2)) / denom
                lowrank = x_admm + u1

                if use_lowrank:
                    batch_size = lowrank.shape[0]
                    rows = lowrank.permute(0, 2, 1).reshape(-1, height)
                    rows = hankel_lowrank_batch(rows, rank=hankel_rank, L=hankel_L)
                    lowrank = rows.reshape(batch_size, width, height).permute(0, 2, 1).contiguous()

                spectrum = torch.fft.fft2(lowrank, dim=(-2, -1))
                magnitude = spectrum.abs()
                sparse = torch.clamp(magnitude - lam_l1 / rho, min=0.0) * torch.exp(1j * torch.angle(spectrum))
                z1 = torch.fft.ifft2(sparse, dim=(-2, -1))

                # -------- (B2) Diffusion prior --------

                spectrum = torch.fft.fft2(x_admm + u2, dim=(-2, -1))
                blocks = block_batch_ddpm(spectrum, pad_shape, pad_height, pad_width, 64)
                prediction = self.p_sample(torch.stack((blocks.real, blocks.imag), dim=1), i)
                spectrum = torch.complex(prediction[:, 0], prediction[:, 1])
                spectrum = reconstruct_batch_ddpm(spectrum, pad_shape, pad_height, pad_width)
                z2 = torch.fft.ifft2(spectrum, dim=(-2, -1))

                # -------- (C) Dual update --------
                u1 = u1 + x_admm - z1
                u2 = u2 + x_admm - z2

            if continous and i % sample_inter == 0:
                spectrum = torch.fft.fft2(x_admm, dim=(-2, -1))
                blocks = block_batch_ddpm(spectrum, pad_shape, pad_height, pad_width, 64)
                history.append(torch.stack((blocks.real, blocks.imag), dim=1))

        spectrum = torch.fft.fft2(x_admm, dim=(-2, -1))
        blocks = block_batch_ddpm(spectrum, pad_shape, pad_height, pad_width, 64)
        result = torch.stack((blocks.real, blocks.imag), dim=1)
        return torch.cat(history, dim=0) if continous else result


    @torch.no_grad()
    def p_sample_loop(
        self, x_in, fid, continous=False, label=None, meta=None,
        pnp_mode=None, lam_l1=None, lam_lr=None, norm_spec=None,
        cs1=6500, cs2=7700,
        pad_shape=None, pad_height=None, pad_width=None,
    ):
     
        if self.conditional:
            return self._p_sample_loop_conditional(x_in, fid)

        pnp_mode = str(pnp_mode).lower()

        if pnp_mode == 'nus_slice_64':
            return self._p_sample_loop_nus_slice_64(
                x_in, fid, continous=continous
            )
        if pnp_mode == 'dosy':
            return self._p_sample_loop_dosy_lowrank(
                x_in, fid, continous=continous,
                lam_l1=lam_l1, lam_lr=lam_lr, norm_spec=norm_spec,
                cs1=cs1, cs2=cs2,
            )
        if pnp_mode == 'ca':
            return self._p_sample_loop_ca(x_in, fid)
        if pnp_mode == 'nus':
            return self._p_sample_loop_nus(
                x_in, fid, pad_shape=pad_shape,
                pad_height=pad_height, pad_width=pad_width,
            )

        return self._p_sample_loop_single_prior(
            x_in, fid, continous=continous, label=label, meta=meta,
            pnp_mode=pnp_mode, lam_l1=lam_l1, lam_lr=lam_lr,
            norm_spec=norm_spec, cs1=cs1, cs2=cs2,
        )


    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size_1 = self.image_size_1
        image_size_2 = self.image_size_2
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size_1, image_size_2), continous)

    @torch.no_grad()
    def super_resolution(
        self, x_in, fid, continous=False, label=None, meta=None,
        pnp_mode=None, lam_l1=None, lam_lr=None, norm_spec=None,
        cs1=6500, cs2=7700,
        pad_shape=None, pad_height=None, pad_width=None,
    ):
        return self.p_sample_loop(
            x_in, fid, continous, label=label, meta=meta,
            pnp_mode=pnp_mode, lam_l1=lam_l1, lam_lr=lam_lr,
            norm_spec=norm_spec, cs1=cs1, cs2=cs2,
            pad_shape=pad_shape, pad_height=pad_height, pad_width=pad_width,
        )


    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

  
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )


    def p_losses(self, x_in, noise=None):

        x_start = x_in['HR']
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        nus_fid = x_in['LR']
        img = torch.complex(x_noisy[:, 0], x_noisy[:, 1]).unsqueeze(1)
        img = torch.fft.ifft(img, dim=-2)
        img[torch.nonzero(nus_fid, as_tuple=True)] = (img[torch.nonzero(nus_fid, as_tuple=True)]
                                                    + 1e6 * nus_fid[torch.nonzero(nus_fid, as_tuple=True)])/(1 + 1e6)
        x_noisy =  torch.concat((torch.fft.fft(img, dim=-2).real, torch.fft.fft(img, dim=-2).imag), dim=1)


        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon = self.denoise_fn(
                torch.cat([x_in['SR'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod)

        loss = self.loss_func(noise, x_recon)
        return loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)


    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            self.sqrt_recip_alphas_cumprod[t] * x_t - pred_xstart) / self.sqrt_recipm1_alphas_cumprod[t]

    @torch.no_grad()
    def ddim_sample(
        self,
        x,
        t,
        clip_denoised=True,
        condition_x=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        model_mean, model_log_variance, x_recon = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, x_recon)
        # alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        # alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        alpha_bar = self.alphas_cumprod[t]
        alpha_bar_prev = self.alphas_cumprod_prev[t]

        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )


        noise = torch.randn_like(x)
        mean_pred = (
            x_recon * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
   
        if t !=0:
            nonzero_mask=1
        else:
            nonzero_mask=0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": x_recon}


    def ddim_sample_loop_progressive(
        self,
        x_in,
        noise=None,
        clip_denoised=True,
        device=None,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(self.denoise_fn.parameters()).device
        shape = x_in.shape
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]
        sample_inter = (1 | (self.num_timesteps//10))

        ret_img = x_in
        for i in tqdm(indices):
            # t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.ddim_sample(
                    x=img,
                    t=i,
                    clip_denoised=clip_denoised,
                    condition_x=x_in,
                    eta=eta,
                )
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, out["sample"]], dim=0)
                yield ret_img
                img = out["sample"]


    def ddim_sample_loop(
        self,
        x_in,
        noise=None,
        clip_denoised=True,
        device=None,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            x_in=x_in,
            noise=noise,
            clip_denoised=clip_denoised,
            device=device,
            eta=eta,
        ):
            final = sample
        return final

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def pad_data_ddpm(data, block_size):
    height, width = data.shape
    pad_height = (block_size - height % block_size) % block_size
    pad_width = (block_size - width % block_size) % block_size
    if isinstance(data, torch.Tensor):
        padded_data = F.pad(data, (0, pad_width, 0, pad_height), mode='constant', value=0)
    elif isinstance(data, np.ndarray):
        padded_data = np.pad(data, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)
    return padded_data, pad_height, pad_width


def block_data_ddpm(data, block_size):
    padded_data, pad_height, pad_width = pad_data_ddpm(data, block_size)
    height, width = padded_data.shape
    num_blocks_vertical = height // block_size
    num_blocks_horizontal = width // block_size
    if isinstance(data, torch.Tensor):
        blocks = torch.empty((num_blocks_vertical * num_blocks_horizontal, block_size, block_size),
                             dtype=padded_data.dtype).to(data.device)
    elif isinstance(data, np.ndarray):
        blocks = np.empty((num_blocks_vertical * num_blocks_horizontal, block_size, block_size),
                             dtype=padded_data.dtype)
    idx = 0
    for i in range(num_blocks_vertical):
        for j in range(num_blocks_horizontal):
            block = padded_data[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            blocks[idx] = block
            idx += 1

    return blocks, padded_data.shape, pad_height, pad_width

def reconstruct_data_ddpm(blocks, pad_shape, pad_height, pad_width):
    height, width = pad_shape
    if len(blocks.shape) == 4 and blocks.shape[0] == 1:
        blocks = blocks.squeeze(0)
    num_blocks, block_height, block_width = blocks.shape
    num_blocks_vertical = height // block_height
    num_blocks_horizontal = width // block_width
    reconstructed_data = torch.empty((height, width), dtype=blocks.dtype).to(blocks.device)
    idx = 0
    for i in range(num_blocks_vertical):
        for j in range(num_blocks_horizontal):
            block = blocks[idx]
            reconstructed_data[i * block_height:(i + 1) * block_height,j * block_width:(j + 1) * block_width] = block
            idx += 1
    final_data = reconstructed_data[:height if pad_height == 0 else -pad_height, :width if pad_width == 0 else -pad_width]
    return final_data


# ===================== Batched Hankel low-rank projection =====================
def hankel_lowrank_batch(
        fid,
        rank=30,
        L=None
):
    """
    Batch Hankel low-rank constraint for NMR FID

    Parameters
    ----------
    fid : torch.Tensor
        shape: (B, N)
        complex FID signals

    rank : int
        retained singular values

    L : int
        Hankel rows

    Returns
    -------
    fid_lr : torch.Tensor
        low-rank reconstructed FID
    """

    B, N = fid.shape

    if L is None:
        L = N // 2

    K = N - L + 1


    H = fid.unfold(
        dimension=-1,
        size=K,
        step=1
    )


    U, S, V = torch.svd_lowrank(
        H,
        q=rank + 5,
        niter=1
    )

    U = U[:, :, :rank]
    S = S[:, :rank]
    V = V[:, :, :rank]

    H_lr = (U * S.unsqueeze(1)) @ V.conj().transpose(-2, -1)

    fid_lr = torch.zeros(
        (B,N),
        dtype=fid.dtype,
        device=fid.device
    )

    count = torch.zeros(
        N,
        device=fid.device
    )


    for i in range(L):
        fid_lr[:,i:i+K] += H_lr[:,i,:]
        count[i:i+K] += 1


    fid_lr = fid_lr / count


    return fid_lr


def reconstruct_3d_batches(blocks, meta_info):
    specs = []
    block_num = meta_info['block_num']
    for b in range(meta_info['num_c']):
        start = b * block_num
        end = start + block_num
        spec_2d = reconstruct_data_ddpm(
            blocks[start:end],
            meta_info['pad_shape'],
            meta_info['pad_height'],
            meta_info['pad_width'])
        specs.append(spec_2d.reshape(meta_info['N1'], meta_info['N2'], meta_info['N3']))
    return torch.stack(specs, dim=0)

def block_3d_batches(specs, meta_info):
    blocks = []
    for b in range(specs.shape[0]):
        spec_2d = specs[b].reshape(-1, meta_info['N3'])
        spec_blocks, _, _, _ = block_data_ddpm(spec_2d, 64)
        blocks.append(spec_blocks)
    return torch.cat(blocks, dim=0)


def reconstruct_batch_ddpm(batch_blocks, pad_shape, pad_height, pad_width):

    recon_list = []
    for b in range(batch_blocks.shape[0]):
        recon_list.append(
            reconstruct_data_ddpm(
                batch_blocks[b:b + 1],
                pad_shape,
                pad_height,
                pad_width
            )
        )
    return torch.stack(recon_list, dim=0)


def block_batch_ddpm(batch_data, pad_shape, pad_height, pad_width, block_size):
    block_list = []
    for b in range(batch_data.shape[0]):
        blocks, _, _, _ = block_data_ddpm(batch_data[b], block_size)
        block_list.append(blocks[0])
    return torch.stack(block_list, dim=0)

def low_rank_svt(X, tau):
    """
    X: complex torch tensor, shape (g, w)
       g = DOSY条数, w = 谱点数
    tau: SVT threshold
    """
    X_real = torch.cat([X.real, X.imag], dim=-1)   # (g, 2w)
    U, S, Vh = torch.linalg.svd(X_real, full_matrices=False)
    S_thr = torch.clamp(S - tau, min=0.0)
    X_lr = (U * S_thr.unsqueeze(0)) @ Vh
    half = X.shape[-1]
    X_rec = torch.complex(X_lr[:, :half], X_lr[:, half:])

    return X_rec

def pseudo2d_to_1d(spec2d, norm_spec, peak_threshold=0.05, peak_distance=20):

    template = np.asarray(norm_spec).squeeze()
    template_real = np.real(template)

    peaks, props = find_peaks(template_real, height=peak_threshold, distance=peak_distance)

    if len(peaks) == 0:
        return np.mean(spec2d, axis=1)

    template_vals = template[peaks]

    valid = np.abs(template_vals) > 1e-12
    peaks = peaks[valid]
    template_vals = template_vals[valid]

    if len(peaks) == 0:
        return np.mean(spec2d, axis=1)

    spec_1d = np.mean(spec2d[:, peaks] / template_vals[None, :], axis=1)
    return spec_1d
