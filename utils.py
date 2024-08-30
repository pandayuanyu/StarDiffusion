import random
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torchvision.utils import save_image

def seed_torch(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# forward diffusion (using the nice property)
def q_sample(x_start, t, sqrt_one_minus_alphas_cumprod, sqrt_alphas_cumprod, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def p_losses(denoise_model, x_start, x_Guide, t,
             sqrt_one_minus_alphas_cumprod, sqrt_alphas_cumprod,
             noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)     # channel = 3, x0

    x_noisy = q_sample(
        x_start=x_start, t=t, noise=noise,
        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
        sqrt_alphas_cumprod=sqrt_alphas_cumprod
    )
    x_to_model = torch.cat([x_Guide, x_noisy], dim=1)  #     channel = 6
    predicted_noise = denoise_model(x_to_model, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


def p_sample(model, X_Guide,
             x, t, t_index,
             betas,
             sqrt_one_minus_alphas_cumprod,
             sqrt_recip_alphas,
             posterior_variance):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    with torch.no_grad():
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * model(torch.cat([X_Guide, x], dim=1), t) / sqrt_one_minus_alphas_cumprod_t
        )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

    # Algorithm 2 (including returning all images)


def p_sample_loop(
        result_path,
        model, X_Guide,
                  shape, timesteps,
                  betas,
                  sqrt_one_minus_alphas_cumprod,
                  sqrt_recip_alphas,
                  posterior_variance):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        with torch.no_grad():
            img = p_sample(
                model,
                X_Guide,
                img,
                torch.full((b,), i, device=device, dtype=torch.long),
                i,
                betas,
                sqrt_one_minus_alphas_cumprod,
                sqrt_recip_alphas,
                posterior_variance
            )
        save_path = os.path.join(result_path, 'process{}.png'.format(i))
        save_image(img, save_path, normalize=True )

        #print(i, '--', img.max(), img.min())


    return img
