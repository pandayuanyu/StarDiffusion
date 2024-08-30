import torch
from network import Unet
import utils
from diffusion_process import make_beta_schedule
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import os
from torchvision.utils import save_image
from tqdm import tqdm

def rev(checkpoints_path = './weights/noise_random2/finnal_weights.pt',
        conditional_image_path = './test_images/a/input.png',
        result_path = './fig/wu_a'):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    transform_tensor = A.Compose(
        [

         # A.CenterCrop(height=2880, width=4320),
         #  A.Resize(height=480, width=720),
            # A.GaussNoise(var_limit=(10.0, 50.0), mean=0.0, always_apply=False, p=0.5),
          #  A.GaussianBlur(blur_limit=37, always_apply=False, p=0.5),
            # A.Cutout(num_holes=20, max_h_size=5, max_w_size=5, fill_value=0, p=0.5),
            # A.Cutout(num_holes=20, max_h_size=5, max_w_size=5, fill_value=255, p=0.5),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0),
            ToTensorV2()
        ]
    )
    conditional_image = np.array(Image.open(conditional_image_path).convert('RGB'))
    conditional_image = transform_tensor(image=conditional_image)['image'].to(device)
    conditional_image = conditional_image.unsqueeze(0)

   # print(conditional_image.shape)

    save_image(conditional_image, os.path.join(result_path, 'guide.png'), normalize=True)


    timesteps = 1000

    # 与训练保持一致
    model = Unet(
        dim=64,
        init_dim=64,
        out_dim=None,
        dim_mults=(1, 2, 4),
        channels=6,
        with_time_emb=True,
        resnet_block_groups=8,
        use_convnext=False,
        convnext_mult=2,
    ).to(device)

    checkpoint = torch.load(checkpoints_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    betas = make_beta_schedule(
        timesteps=timesteps,
        schedule_type='linear'
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

    with torch.no_grad():
        result = utils.p_sample_loop(
        result_path=result_path,
        model=model,
        X_Guide = conditional_image,
        shape=conditional_image.shape,
        timesteps=timesteps,
        betas=betas,
        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
        sqrt_recip_alphas=sqrt_recip_alphas,
        posterior_variance=posterior_variance)

if __name__ == '__main__':
    utils.seed_torch(2022)
    weights_path = './weights/noise_random2/finnal_weights.pt'
    input_folder_path = './lol_15/low'
    result_imgs_path = './test_results'

    for filename in os.listdir(input_folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):

            conditional_image_path = os.path.join(input_folder_path, filename)
            tests_name = os.path.splitext(filename)[0]
            result_save_path = result_imgs_path + '/' + tests_name

            rev(checkpoints_path=weights_path, conditional_image_path=conditional_image_path, result_path=result_save_path)

