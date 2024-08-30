import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from datetime import datetime
import os

from args_file import set_args
from diffusion_process import make_beta_schedule
from network import Unet
from Dataset import Starry_Sky
import utils

def main(args):
    utils.seed_torch(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Starry_Sky(dataset_path=args.dataset_path, patch_size=args.patch_size)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )
    for _, batch in enumerate(data_loader):
        test_batch = batch
        break
   # save_image(test_batch, 'test_batch.png', normalize=True)

    model = Unet(
        dim=64,
        init_dim=64,
        out_dim=None,
        dim_mults=(1, 2, 4),
        channels=3+3,
        with_time_emb=True,
        resnet_block_groups=8,
        use_convnext=False,
        convnext_mult=2,
    ).to(device)
    print("-Train_network para: {} ".format(sum(x.numel() for x in model.parameters())))

    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    start_epoch = 0
    if args.warm_start_path is not None:
        checkpoints = torch.load(args.warm_start_path)
        model.load_state_dict(checkpoints['model_state_dict'])
        optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
        start_epoch = checkpoints['epoch']
        optimizer.param_groups[0]['lr'] = args.learning_rate
        print('--weight loaded...')

    train_time = datetime.now().strftime("%m-%d_%H-%M")
    logs_name = train_time + '_epoch{}'.format(args.epochs + start_epoch)
    logs_dir = os.path.join('./logs/', logs_name)
    writer = SummaryWriter(logs_dir)

    betas = make_beta_schedule(
        timesteps=args.timesteps,
        schedule_type=args.beta_schedule_type
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

    step = 0
    for epoch in range(start_epoch, start_epoch + args.epochs):
        loop = tqdm(data_loader)
        for _, sample_batch in enumerate(loop):
            optimizer.zero_grad()

            GT_batch = sample_batch['GT'].to(device)
            input_batch = sample_batch['input'].to(device)

            batch_size = GT_batch.shape[0]

            t = torch.randint(0, args.timesteps, (batch_size,), device=device).long()
            loss = utils.p_losses(
                denoise_model=model,
                x_start=GT_batch,
                x_Guide=input_batch,
                t=t,
                sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                sqrt_alphas_cumprod=sqrt_alphas_cumprod
            )

            loss.backward()
            optimizer.step()
            step += 1

            writer.add_scalar('loss', loss.item(), global_step=step)
            loop.set_description(f"Epoch [{epoch + 1}/{args.epochs}]")
            loop.set_postfix(
                loss=loss.item(),
            )

        save_path = os.path.join('./checkpoints/', logs_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if epoch % args.save_frequency == 0 :
            save_name = os.path.join(save_path + '/',
                                     'epoch{}_{}.pt'.format(args.epochs, epoch + 1))
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch
                }, save_name
            )
            print('weight saved at:' + save_name)



if __name__ == '__main__':
    args = set_args()
    main(args)

