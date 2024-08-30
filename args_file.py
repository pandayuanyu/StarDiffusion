import argparse

def set_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default='./starfield_data', help='main path to the dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='number of CPU cores to load the dataset')
    parser.add_argument('--patch_size', type=int, default=128, help='patch size for training')

    # 训练相关参数
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='LR')
    parser.add_argument('--epochs', type=int, default=500, help='training epochs')
    parser.add_argument('--timesteps', type=int, default=1000, help='diffusion time steps')
    parser.add_argument('--beta_schedule_type', type=str, default='linear',
                        help='optional choice：cosine、linear、quadratic、sigmoid')

    parser.add_argument('--warm_start_path', type=str, default=None, help='Continue training the previously trained weight')
    parser.add_argument('--save_frequency', type=int, default=5, help='frequency of saving checkpoints')

    args = parser.parse_args()
    print('=-' * 30)
    for arg in vars(args):
        print('--', arg, ':', getattr(args, arg))
    print('=-' * 30)

    return args


if __name__ == '__main__':
    set_args()
