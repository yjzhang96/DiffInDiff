import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm


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
    else:
        raise NotImplementedError(schedule)
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
        model_d,
        model_D,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        # self.denoise_fn = denoise_fn
        self.model_d = model_d
        self.model_D = model_D
        self.loss_type = loss_type
        self.conditional = conditional
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

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
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in
            shape = x.shape
            img = torch.randn(shape, device=device)
            ret_img = x
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i, condition_x=x)
                if i % sample_inter == 0:
                    # img = img
                    ret_img = torch.cat([ret_img, x_in + img], dim=0)
        if continous:
            return ret_img
        else:
            return ret_img[-1]


    @torch.no_grad()
    def p_sample_skip(self, denoise_fn, xt, cond, t, next_t, eta=0):
        n = t.size(0)
        at = self.alphas_cumprod[(t+1).long()]
        at_next = self.alphas_cumprod[(next_t+1).long()]
        noise_level = torch.FloatTensor(
                [self.sqrt_alphas_cumprod_prev[t.long()+1]]).repeat(n, 1).to(t.device)
        et = denoise_fn(torch.cat([cond, xt],dim=1),noise_level)
            
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
        c2 = ((1 - at_next) - c1 ** 2).sqrt()
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(cond) + c2 * et
        return xt_next

    
    @torch.no_grad()
    def p_sample_skiploop(self, x_in, seq, eta = 0, continous=False):
        device = self.betas.device
        sample_steps = len(seq)
        sample_inter = (1 | (sample_steps//10))
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, sample_steps)), desc='sampling loop time step', total=sample_steps):
                img = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            shape = x_in.shape
            Nt = torch.randn(shape, device=device)
            yt = x_in
            ret_img = x_in
            seq_next = [-1] + list(seq[:-1])
            seq_rev = seq[::-1]
            seq_next_rev = seq_next[::-1]
            n = x_in.size(0)
            
            for i in tqdm(range(len(seq)), desc='sampling loop time step', total=sample_steps):
                t = (torch.ones(n) * seq_rev[i]).to(device)
                next_t = (torch.ones(n) * seq_next_rev[i]).to(device)
                Nt = self.p_sample_skip(self.model_D, Nt, yt, t, next_t) 
                
                ## small diffusion to generate condition image
                if i == (len(seq) -1):
                    ## the last loop, break
                    ret_img = torch.cat([ret_img, Nt], dim=0)
                else:
                    seq_inner = seq_rev[i+1:]
                    seq_next_inner = seq_next_rev[i+1:]  
                    for j in tqdm(range(len(seq_inner)), desc='sampling inner loop', total=len(seq_inner)):
                        inner_t = (torch.ones(n) * seq_inner[j]).to(device)
                        inner_next_t = (torch.ones(n) * seq_next_inner[j]).to(device)
                        yt = self.p_sample_skip(self.model_d, Nt, x_in, inner_t, inner_next_t)

                if i % sample_inter == 0:
                    # img = img
                    ret_img = torch.cat([ret_img, Nt], dim=0)
        if continous:
            return ret_img
        else:
            return ret_img[-1]

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def restore(self, x_in, continous=False):
        return self.p_sample_loop(x_in, continous)

    @torch.no_grad()
    def skip_restore(self, x_in, seq, continous=False):
        return self.p_sample_skiploop(x_in,seq,continous=continous) 

    def q_sample(self, x_start, sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
            sqrt_alpha_cumprod * x_start +
            (1 - sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def q_sample_reverse(self, x_noisy, sqrt_alpha_cumprod, noise_recon=None):
        return (x_noisy - (1 - sqrt_alpha_cumprod**2).sqrt() * noise_recon) * (1/sqrt_alpha_cumprod) 

    def p_losses(self, x_in, noise=None):
        y_start = x_in['HQ'] 
        [b, c, h, w] = y_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        # sqrt_alpha_cumprod = torch.FloatTensor(
        #     np.random.uniform(
        #         self.sqrt_alphas_cumprod_prev[t-1],
        #         self.sqrt_alphas_cumprod_prev[t],
        #         size=b
        #     )
        sqrt_alpha_cumprod = (torch.ones(b) * self.sqrt_alphas_cumprod_prev[t]).to(y_start.device) ## here should we use t+1?
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.view(
            b, -1)

        noise = default(noise, lambda: torch.randn_like(y_start))
        y_t = t/self.num_timesteps * x_in['LQ'] + (1 - t/self.num_timesteps) * x_in['HQ']
        x_noisy = self.q_sample(
            x_start=y_t, sqrt_alpha_cumprod=sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        ## small diffusion process         
        noise_recon = self.model_d(
            torch.cat([x_in['LQ'], x_noisy], dim=1), sqrt_alpha_cumprod)
        loss_d = self.loss_func(noise, noise_recon)

        ## large diffusion process
        y_t_recon = self.q_sample_reverse(x_noisy, sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise_recon) 
        residual_recon = self.model_D(
            torch.cat([y_t_recon.detach(),x_noisy], dim=1), sqrt_alpha_cumprod
        )
        residual_gt = x_noisy - y_start
        loss_D = self.loss_func(residual_recon, residual_gt.detach())
        return [loss_d, loss_D]

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
