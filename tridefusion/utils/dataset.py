import json
import os
import torch
from torchvision.datasets.folder import has_file_allowed_extension

IMG_EXTENSIONS = ['.png', 'tif', 'tiff']
ALL_NOISE_LEVELS = [1, 2, 4, 8, 16]


def is_image_file(filename):
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, noise_levels):
        super().__init__()
        assert all([level in ALL_NOISE_LEVELS for level in noise_levels])
        self.noise_levels = noise_levels
        self.samples = self._gather_files()
        dataset_info = {'Dataset': 'train' if train else 'test',
                        'Noise levels': self.noise_levels,
                        f'{len(self.types)} Types': self.types,
                        'Fovs': self.fovs,
                        '# samples': len(self.samples)
                        }
        print(json.dumps(dataset_info, indent=4))

    def _gather_files(self):
        samples = []
        root_dir = os.path.expanduser(self.root)
        test_mix_dir = os.path.join(root_dir, 'test_mix')
        gt_dir = os.path.join(test_mix_dir, 'gt')
        for noise_level in self.noise_levels:
            if noise_level == 1:
                noise_dir = os.path.join(test_mix_dir, 'raw')
            elif noise_level in [2, 4, 8, 16]:
                noise_dir = os.path.join(test_mix_dir, f'avg{noise_level}')
            for fname in sorted(os.listdir(noise_dir)):
                if is_image_file(fname):
                    noisy_file = os.path.join(noise_dir, fname)
                    clean_file = os.path.join(gt_dir, fname)
                    samples.append((noisy_file, clean_file))
        return samples
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (noisy, clean)
        """
        noisy_file, clean_file = self.samples[index]
        noisy, clean = self.loader(noisy_file), self.loader(clean_file)
        if self.transform is not None:
            noisy = self.transform(noisy)
        if self.target_transform is not None:
            clean = self.target_transform(clean)

    def __len__(self):
        return len(self.samples)