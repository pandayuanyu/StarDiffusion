import torch


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def make_beta_schedule(timesteps, schedule_type='cosine'):
    if schedule_type == 'cosine':
        return cosine_beta_schedule(timesteps, s=0.008)
    if schedule_type == 'linear':
        return linear_beta_schedule(timesteps)
    if schedule_type == 'quadratic':
        return quadratic_beta_schedule(timesteps)
    if schedule_type == 'sigmoid':
        return sigmoid_beta_schedule(timesteps)


from torch.utils.tensorboard import SummaryWriter
if __name__ == '__main__':
    timesteps = 1000
    cbs = cosine_beta_schedule(timesteps)
    lbs = linear_beta_schedule(timesteps)
    qbs = quadratic_beta_schedule(timesteps)
    sbs = sigmoid_beta_schedule(timesteps)

    mkb = make_beta_schedule(timesteps)
    print(mkb == cbs)


