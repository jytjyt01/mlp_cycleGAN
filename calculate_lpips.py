from tqdm import tqdm
import torch
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import cv2
import lpips
import argparse

parser = argparse.ArgumentParser(description='PSNR SSIM script', add_help=False)
parser.add_argument('--real_path', default='outputlanting/B')
parser.add_argument('--fake_path', default='outputlanting/A')
parser.add_argument('-v', '--version', type=str, default='0.1')
args = parser.parse_args()


def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg", ".png", ".jpeg"])


def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img / 255.
    return img


class DataLoaderVal(Dataset):
    def __init__(self, target_transform=None):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform

        gt_dir = args.real_path
        input_dir = args.fake_path

        clean_files = sorted(os.listdir(os.path.join(gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(input_dir)))

        self.clean_filenames = [os.path.join(gt_dir, x) for x in clean_files if is_png_file(x)]
        self.noisy_filenames = [os.path.join(input_dir, x) for x in noisy_files if is_png_file(x)]

        self.tar_size = len(self.clean_filenames)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size

        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        clean = clean.permute(2, 0, 1)
        noisy = noisy.permute(2, 0, 1)

        return clean, noisy, clean_filename, noisy_filename


def get_validation_data():
    return DataLoaderVal(None)


test_dataset = get_validation_data()
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

## Initializing the model
loss_fn = lpips.LPIPS(net='alex', version=args.version)

if __name__ == '__main__':
    # ---------------------- LPIPS ----------------------
    files = os.listdir(args.real_path)
    i = 0
    total_lpips_distance = 0
    average_lpips_distance = 0
    for file in files:

        try:
            # Load images
            img0 = lpips.im2tensor(lpips.load_image(os.path.join(args.real_path, file)))
            img1 = lpips.im2tensor(lpips.load_image(os.path.join(args.fake_path, file)))

            if (os.path.exists(os.path.join(args.real_path, file)),
                os.path.exists(os.path.join(args.fake_path, file))):
                i = i + 1

            # Compute distance
            current_lpips_distance = loss_fn.forward(img0, img1)
            total_lpips_distance = total_lpips_distance + current_lpips_distance

        except Exception as e:
            print(e)

    average_lpips_distance = float(total_lpips_distance) / i

    print("The processed iamges is ", i)
    print("LPIPS: ", average_lpips_distance)