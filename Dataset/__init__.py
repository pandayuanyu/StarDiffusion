import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class Starry_Sky(Dataset):
    def __init__(self, dataset_path, patch_size=128):
        self.dataset_path = dataset_path
        self.input_path = os.path.join(self.dataset_path, 'input/')
        self.GT_path = os.path.join(self.dataset_path, 'GT/')

        self.data_list = os.listdir(self.input_path)
        self.transform = A.Compose(
            [
                A.RandomCrop(height=patch_size, width=patch_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
                ToTensorV2()
            ],
            additional_targets={"image1": "image"}
        )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image_name = self.data_list[idx]
        input_image_path = os.path.join(self.input_path, image_name)
        GT_image_path = os.path.join(self.GT_path, image_name)

        input_image = np.array(Image.open(input_image_path).convert('RGB'))
        GT_image = np.array(Image.open(GT_image_path).convert('RGB'))

        augmentations = self.transform(
            image=input_image,
            image1=GT_image
        )

        input_image = augmentations['image']
        GT_image = augmentations['image1']

        sample = {
            'input': input_image,
            'GT': GT_image
        }

        return sample


from args_file import set_args
from torch.utils.data import DataLoader
from torchvision.utils import save_image
if __name__ == '__main__':
    args = set_args()
    dataset = Starry_Sky(args.dataset_path)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )

    for i, sample_batch in enumerate(dataloader):
        GT_batch = sample_batch['GT']
        input_batch = sample_batch['input']

        print('=-'*30)
        print(GT_batch.shape)
        print(GT_batch.min(), GT_batch.max())
        print(type(GT_batch))
        print(GT_batch.dtype)

        save_image(GT_batch, 'GT_batch.png', nrow=8, normalize=True)
        save_image(input_batch, 'input_batch.png', nrow=8, normalize=True)

        break








