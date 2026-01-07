from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

# Wraps any dataset to also return indices (for train + score loaders)
class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)

def load_cifar10_data_agat(
    batch_size=128,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    **kwargs
):
    """
    Returns:
      - train_loader: shuffle=True (for training)
      - score_loader: shuffle=False (for stable full-pass scoring, if you need it)
      - test_loader : shuffle=False
    Notes:
      - pass worker_init_fn=... and generator=... via **kwargs for reproducibility
    """

    # base transform only (AGAT augmentations happen in training code)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914008984375, 0.482159140625, 0.446531015625), # mean
            (0.24703278185799551, 0.24348423011049403, 0.26158752307127053) # std
        ),
    ])

    raw_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_dataset = IndexedDataset(raw_train)

    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    # only keep persistent_workers if num_workers > 0
    if num_workers == 0:
        persistent_workers = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        **kwargs
    )

    # stable pass over the same train dataset
    score_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        **kwargs
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        **kwargs
    )

    return train_loader, score_loader, test_loader
