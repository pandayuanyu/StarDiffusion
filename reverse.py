import torch
from network import Unet
import utils
from diffusion_process import make_beta_schedule
import torch.nn.functional as F

def rev():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = './checkpoints/09-13_01-09_epoch500/epoch500_13.pt'
    timesteps = 1000

    model = Unet(
        dim=64,
        init_dim=64,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=6,
        with_time_emb=True,
        resnet_block_groups=8,
        use_convnext=False,
        convnext_mult=2,
    ).to(device)

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['encoder_state_dict'])

    betas = make_beta_schedule(
        timesteps=timesteps,
        schedule_type='quadratic'
    )

    # define alphas
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


    result = utils.p_sample_loop(
        model=model,
        shape=[64, 3, 64, 64],
        timesteps=timesteps,
        betas=betas,
        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
        sqrt_recip_alphas=sqrt_recip_alphas,
        posterior_variance=posterior_variance)


if __name__ == '__main__':
    rev()
