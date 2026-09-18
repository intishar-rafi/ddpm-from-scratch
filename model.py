"""
Denoising Diffusion (DDPM) from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - linear_beta_schedule
import torch
import torch.nn.functional as F

def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    # TODO: return a linear beta schedule of length T
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)

# Step 2 - alphas_from_betas
import torch
import torch.nn.functional as F

def alphas_from_betas(betas):
    # TODO: return 1 - betas
    return 1 - betas

# Step 3 - cumprod_alphas
import torch
import torch.nn.functional as F

def cumprod_alphas(alphas):
    # TODO: cumulative product of alphas
    return torch.cumprod(alphas, dim=0)

# Step 4 - extract_into_batch
import torch
import torch.nn.functional as F

def extract_into_batch(a, t, x):
    # TODO: gather a[t] and reshape to (B, 1, 1, 1) for broadcasting with x
    out = a.gather(0, t)
    return out.reshape(t.shape[0], *((1,) * (x.dim() - 1)))

# Step 5 - q_sample
import torch
import torch.nn.functional as F

def q_sample(x0, t, noise, alphas_cumprod):
    # TODO: x_t = sqrt(bar_alpha_t) * x0 + sqrt(1 - bar_alpha_t) * noise
    sqrt_ac = extract_into_batch(torch.sqrt(alphas_cumprod), t, x0)
    sqrt_one_minus_ac = extract_into_batch(torch.sqrt(1 - alphas_cumprod), t, x0)
    return sqrt_ac * x0 + sqrt_one_minus_ac * noise

# Step 6 - build_diffusion_schedule
import torch
import torch.nn.functional as F

def build_diffusion_schedule(T: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02) -> dict:
    # TODO: build betas, alphas, alphas_cumprod and useful sqrts
    betas = linear_beta_schedule(T, beta_start, beta_end)
    alphas = alphas_from_betas(betas)
    alphas_cumprod = cumprod_alphas(alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

    return {
        'T': T,
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
    }

# Step 7 - noise_prediction_loss
import torch
import torch.nn.functional as F

def noise_prediction_loss(noise_pred, noise):
    # TODO: MSE between predicted and true noise
    return F.mse_loss(noise_pred, noise)

# Step 8 - diffusion_training_loss
import torch
import torch.nn.functional as F

def diffusion_training_loss(model, x0, t, noise, alphas_cumprod):
    # TODO: q_sample -> model -> MSE(noise_pred, noise)
    x_t = q_sample(x0, t, noise, alphas_cumprod)
    noise_pred = model(x_t, t)
    return noise_prediction_loss(noise_pred, noise)

# Step 9 - timestep_embedding
import torch
import torch.nn.functional as F

def timestep_embedding(t, dim: int):
    # TODO: sinusoidal timestep embedding of shape (B, dim)
    half = dim // 2
    if half == 1:
        exponents = torch.zeros(1, device=t.device, dtype=torch.float32)
    else:
        exponents = torch.arange(half, device=t.device, dtype=torch.float32) / (half - 1)
    freqs = 1.0 / (10000 ** exponents)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    return emb

# Step 10 - init_tiny_unet
import torch
import torch.nn.functional as F

def init_tiny_unet(in_ch: int = 1, hidden: int = 16, time_dim: int = 16, seed: int = 0) -> dict:
    # TODO: initialize tiny residual denoiser parameters
    torch.manual_seed(seed)

    def randn_param(*shape):
        return (torch.randn(*shape) * 0.02).requires_grad_(True)

    def zeros_param(*shape):
        return torch.zeros(*shape).requires_grad_(True)

    params = {
        'conv_in_w': randn_param(hidden, in_ch, 3, 3),
        'conv_in_b': zeros_param(hidden),
        'time_mlp_w': randn_param(hidden, time_dim),
        'time_mlp_b': zeros_param(hidden),
        'conv_mid_w': randn_param(hidden, hidden, 3, 3),
        'conv_mid_b': zeros_param(hidden),
        'conv_out_w': randn_param(in_ch, hidden, 3, 3),
        'conv_out_b': zeros_param(in_ch),
    }
    return params

# Step 11 - tiny_unet_forward
import torch
import torch.nn.functional as F

def tiny_unet_forward(x, t, params: dict):
    # TODO: time-conditioned tiny CNN predicting noise
    h = F.conv2d(x, params['conv_in_w'], params['conv_in_b'], padding=1)

    temb = timestep_embedding(t, params['time_mlp_w'].shape[1])
    temb = F.relu(F.linear(temb, params['time_mlp_w'], params['time_mlp_b']))
    h = h + temb[:, :, None, None]

    h = F.relu(h)
    h = F.relu(F.conv2d(h, params['conv_mid_w'], params['conv_mid_b'], padding=1))

    return F.conv2d(h, params['conv_out_w'], params['conv_out_b'], padding=1)

# Step 12 - make_blob_dataset
import torch
import torch.nn.functional as F

def make_blob_dataset(n: int = 128, size: int = 8, seed: int = 0):
    # TODO: n images with a random bright disk on a black background
    torch.manual_seed(seed)
    radius = size // 4

    rows = torch.arange(size).view(size, 1).float()
    cols = torch.arange(size).view(1, size).float()

    images = torch.zeros(n, 1, size, size)
    for i in range(n):
        center = torch.randint(radius, size - radius, (2,))
        cy, cx = center[0].item(), center[1].item()
        dist = torch.sqrt((rows - cy) ** 2 + (cols - cx) ** 2)
        disk = (dist <= radius).float()
        images[i, 0] = disk

    return images

# Step 13 - ddpm_train_step
import torch
import torch.nn.functional as F

def ddpm_train_step(params: dict, x0, schedule: dict, lr: float = 1e-2, seed: int = 0) -> tuple[dict, float]:
    # TODO: sample t,noise -> loss -> SGD on params
    torch.manual_seed(seed)

    B = x0.shape[0]
    T = schedule['T']
    alphas_cumprod = schedule['alphas_cumprod']

    t = torch.randint(0, T, (B,))
    noise = torch.randn_like(x0)

    loss = diffusion_training_loss(
        lambda x, t: tiny_unet_forward(x, t, params),
        x0, t, noise, alphas_cumprod
    )
    loss.backward()

    new_params = {}
    for name, p in params.items():
        if p.grad is not None:
            new_params[name] = (p - lr * p.grad).detach().requires_grad_(True)
        else:
            new_params[name] = p.detach().clone().requires_grad_(True)

    return new_params, float(loss)

# Step 14 - train_ddpm
import torch
import torch.nn.functional as F

def train_ddpm(dataset, params: dict, schedule: dict, num_steps: int = 50, batch_size: int = 16, lr: float = 1e-2, seed: int = 0) -> tuple[dict, list]:
    # TODO: minibatch SGD training loop
    n = dataset.shape[0]
    history = []

    for step in range(num_steps):
        torch.manual_seed(seed + step)
        idx = torch.randint(0, n, (batch_size,))
        x0 = dataset[idx]

        params, loss = ddpm_train_step(params, x0, schedule, lr=lr, seed=seed + step)
        history.append(loss)

    return params, history

# Step 15 - predict_x0_from_eps
import torch
import torch.nn.functional as F

def predict_x0_from_eps(x_t, t, eps, alphas_cumprod):
    # TODO: invert the q_sample equation for x0
    sqrt_ac = extract_into_batch(torch.sqrt(alphas_cumprod), t, x_t)
    sqrt_one_minus_ac = extract_into_batch(torch.sqrt(1 - alphas_cumprod), t, x_t)
    return (x_t - sqrt_one_minus_ac * eps) / sqrt_ac

# Step 16 - ddpm_p_mean_variance
import torch
import torch.nn.functional as F

def ddpm_p_mean_variance(x_t, t, eps, schedule: dict):
    # TODO: return (posterior_mean, variance, x0_hat)
    alphas = schedule['alphas']
    alphas_cumprod = schedule['alphas_cumprod']
    betas = schedule['betas']

    x0_hat = predict_x0_from_eps(x_t, t, eps, alphas_cumprod)
    x0_hat = torch.clamp(x0_hat, -1.0, 1.0)

    # build alphas_cumprod_prev with the convention abar_{-1} = 1
    alphas_cumprod_prev = torch.cat([torch.ones(1, device=alphas_cumprod.device), alphas_cumprod[:-1]])

    ac_t = extract_into_batch(alphas_cumprod, t, x_t)
    ac_prev = extract_into_batch(alphas_cumprod_prev, t, x_t)
    alpha_t = extract_into_batch(alphas, t, x_t)
    beta_t = extract_into_batch(betas, t, x_t)

    coef_x0 = torch.sqrt(ac_prev) * beta_t / (1 - ac_t)
    coef_xt = torch.sqrt(alpha_t) * (1 - ac_prev) / (1 - ac_t)

    mean = coef_x0 * x0_hat + coef_xt * x_t
    variance = beta_t

    return mean, variance, x0_hat

# Step 17 - ddpm_p_sample
import torch
import torch.nn.functional as F

def ddpm_p_sample(x_t, t, params: dict, schedule: dict, noise=None):
    # TODO: one reverse step x_t -> x_{t-1}
    eps = tiny_unet_forward(x_t, t, params)
    mean, var, _ = ddpm_p_mean_variance(x_t, t, eps, schedule)

    if noise is None:
        noise = torch.randn_like(x_t)

    # zero out noise where t == 0 (deterministic final step)
    mask = (t != 0).float()
    mask = extract_into_batch(mask, torch.arange(t.shape[0], device=t.device), x_t)
    noise = noise * mask

    x_prev = mean + torch.sqrt(var) * noise
    return x_prev

# Step 18 - ddpm_sample_loop
import torch
import torch.nn.functional as F

def ddpm_sample_loop(params: dict, schedule: dict, shape: tuple, seed: int = 0):
    # TODO: ancestral sampling from pure noise to x0
    torch.manual_seed(seed)
    x = torch.randn(shape)

    T = schedule['T']
    B = shape[0]

    for t in range(T - 1, -1, -1):
        t_batch = torch.full((B,), t, dtype=torch.long)
        x = ddpm_p_sample(x, t_batch, params, schedule)

    return x

# Step 19 - sample_quality_mse
import torch
import torch.nn.functional as F

def sample_quality_mse(samples, dataset) -> float:
    # TODO: mean over samples of min MSE to any dataset image
    N = samples.shape[0]
    M = dataset.shape[0]

    samples_flat = samples.reshape(N, -1)
    dataset_flat = dataset.reshape(M, -1)

    # pairwise squared diffs -> mean over pixel dim -> (N, M) MSE matrix
    diffs = samples_flat.unsqueeze(1) - dataset_flat.unsqueeze(0)  # (N, M, D)
    mse_matrix = (diffs ** 2).mean(dim=2)  # (N, M)

    min_mse, _ = mse_matrix.min(dim=1)  # (N,)
    return float(min_mse.mean())

# Step 20 - ddpm_experiment
import torch
import torch.nn.functional as F

def ddpm_experiment(n_data: int = 64, size: int = 8, T: int = 20, hidden: int = 16, num_steps: int = 40, batch_size: int = 16, lr: float = 5e-2, n_samples: int = 8, seed: int = 0) -> dict:
    # TODO: data -> train -> sample -> metrics
    dataset = make_blob_dataset(n_data, size, seed)
    schedule = build_diffusion_schedule(T)
    params = init_tiny_unet(1, hidden, time_dim=hidden, seed=seed)

    params, history = train_ddpm(dataset, params, schedule, num_steps, batch_size, lr, seed)

    samples = ddpm_sample_loop(params, schedule, (n_samples, 1, size, size), seed=seed + 1)

    torch.manual_seed(seed + 2)
    noise_samples = torch.randn(n_samples, 1, size, size)

    sample_mse = sample_quality_mse(samples, dataset)
    noise_mse = sample_quality_mse(noise_samples, dataset)

    return {
        'train_losses': history,
        'final_loss': history[-1],
        'sample_mse': sample_mse,
        'noise_mse': noise_mse,
        'improvement': noise_mse - sample_mse,
    }
