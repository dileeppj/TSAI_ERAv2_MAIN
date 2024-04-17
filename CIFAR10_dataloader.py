import albumentations as A
from albumentations.pytorch import ToTensorV2


from torch.utils.data import Dataset
from torchvision import datasets


mean_val = [0.4915, 0.4823, 0.4468]
std_val = [0.2470, 0.2435, 0.2616]

# cutout needs to be half of the image size
cutout_size = 8

class CIFAR10_Transforms(Dataset):
    def __init__(self, dataset, transforms):
        self.transforms = transforms
        self.dataset = dataset

    def __getitem__(self, idx):
        # Get the image and label from the dataset
        image, label = self.dataset[idx]

        # Apply transformations on the image
        image = self.transforms(image=np.array(image))["image"]

        return image, label

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        return (f"CIFAR10_Transforms(dataset={self.dataset}, transforms={self.transforms})")

    def __str__(self):
        return (f"CIFAR10_Transforms(dataset={self.dataset}, transforms={self.transforms})")

def get_CFAR10_data_loaders(train_data, test_data):
    # Train Phase Transformations
    train_transforms = A.Compose(
        A.Compose([
            A.PadIfNeeded(min_height=40, min_width=40, border_mode=0, value=mean_val, p=1.0),
            A.RandomCrop(height=32, width=32, p=1.0),
            A.HorizontalFlip(),
            # A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=12, p=0.5),
            A.CoarseDropout( max_holes=1, max_height=cutout_size, max_width=cutout_size,
                            min_holes=1, min_height=cutout_size, min_width=cutout_size,
                            fill_value=mean_val, p=0.5, mask_fill_value=None),
            A.Normalize(mean=mean_val, std=std_val),
            ToTensorV2(),
        ])
    )

    # Test Phase Transformations
    test_transforms = A.Compose(A.Compose([
        A.Normalize(mean=mean_val, std=std_val),
        ToTensorV2(),
    ]))

    train_data = CIFAR10_Transforms(train_data, train_transforms)
    test_data = CIFAR10_Transforms(test_data, test_transforms)

    # Dataloader arguments
    dataloader_args = dict(shuffle=True, batch_size=512, num_workers=2, pin_memory=True) if cuda_availabilty() else dict(shuffle=True, batch_size=64)

    # Train DataLoader
    train_loader = torch.utils.data.DataLoader(train_data, **dataloader_args)

    # Test DataLoader
    test_loader = torch.utils.data.DataLoader(test_data, **dataloader_args)

    return train_loader, test_loader

def get_CIFAR10_dataset():
    train_data = datasets.CIFAR10('../data', train=True, download=True)
    test_data = datasets.CIFAR10('../data', train=False, download=True)
    return train_data, test_data
