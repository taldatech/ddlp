"""
functions and classes to process the CLEVRER dataset
"""

import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import cv2
# import utils.tps as tps
import glob

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


# --- old preprocessing functions for the single image setting --- #
def list_images_in_dir(path):
    valid_images = [".jpg", ".gif", ".png"]
    img_list = []
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        img_list.append(os.path.join(path, f))
    return img_list


def prepare_numpy_file(path_to_image_dir, image_size=128, frameskip=1, start_frame=1):
    # path_to_image_dir = '/media/newhd/data/traffic_data/rimon_frames/'
    img_list = list_images_in_dir(path_to_image_dir)
    img_list = sorted(img_list, key=lambda x: int(x.split('/')[-1].split('_')[-1].split('.')[0]))
    img_list = [img_list[i] for i in range(len(img_list)) if
                abs(int(img_list[i].split('/')[-1].split('_')[-1].split('.')[0])) % 1000 > start_frame]
    print(f'img_list: {len(img_list)}, 0: {img_list[0]}, -1: {img_list[-1]}')
    img_np_list = []
    for i in tqdm(range(len(img_list))):
        if i % frameskip != 0:
            continue
        img = Image.open(img_list[i])
        img = img.convert('RGB')
        #         img = img.crop((60, 0, 480, 420))
        img = img.resize((image_size, image_size), Image.BICUBIC)
        img_np = np.asarray(img)
        img_np_list.append(img_np)
    img_np_array = np.stack(img_np_list, axis=0)
    print(f'img_np_array: {img_np_array.shape}')
    save_path = os.path.join(path_to_image_dir, f'clevrer_img{image_size}np_fs{frameskip}.npy')
    np.save(save_path, img_np_array)
    print(f'file save at @ {save_path}')


# --- end old preprocessing functions for the single image setting --- #

# --- new preprocessing functions for the episodic setting --- #
"""
Instructions:
1. Download the CLEVRER dataset from here: http://clevrer.csail.mit.edu/
2. Extract the directories 'train' and 'valid', they should contain directories named like: 'video_10000-11000'
3. In the directory containing the 'train' and 'valid', run `preprocess_clevrer(mode='train', ep_len=100, start_frame=18)`
"""


def extract_frames(path_to_video, path_to_save_frames, start_frame, end_frame, image_size=128):
    vidcap = cv2.VideoCapture(path_to_video)
    success, image = vidcap.read()
    count = 0
    curr_frame = 0
    while success:
        if count >= start_frame:
            path = os.path.join(path_to_save_frames, "%d.png" % curr_frame)
            resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
            cv2.imwrite(path, resized)
            curr_frame += 1
        success, image = vidcap.read()
        #         print('Read a new frame: ', success)
        count += 1
        if count == end_frame:
            break
    vidcap.release()


def preprocess_clevrer(mode='train', ep_len=100, start_frame=18):
    assert mode in ['train', 'valid']
    end_frame = start_frame + ep_len
    path_to_dir = f'./ep_{mode}'
    os.makedirs(path_to_dir, exist_ok=True)
    path_to_video_dir = f'./{mode}'
    video_dirs = [d for d in os.listdir(path_to_video_dir)
                  if os.path.isdir(os.path.join(path_to_video_dir, d)) and 'video' in d]
    video_dirs = sorted(video_dirs)
    episode = 0
    for i in range(len(video_dirs)):
        curr_dir = os.path.join(path_to_video_dir, video_dirs[i])
        print(f'current dir: {curr_dir}')
        videos_curr_dir = sorted([v for v in os.listdir(curr_dir) if 'video' in v])
        for j in range(len(videos_curr_dir)):
            curr_video = os.path.join(curr_dir, videos_curr_dir[j])
            target_dir = os.path.join(path_to_dir, f'{episode}')
            os.makedirs(target_dir, exist_ok=True)
            # extract frames
            extract_frames(curr_video, target_dir, start_frame, end_frame)
            episode += 1


# --- end new preprocessing functions for the episodic setting --- #


class CLEVREREpDataset(Dataset):
    def __init__(self, root, mode, ep_len=100, sample_length=20, image_size=128):
        # path = os.path.join(root, mode)
        assert mode in ['train', 'val', 'valid', 'test']
        if mode == 'val':
            mode = 'valid'
        self.root = os.path.join(root, mode)
        self.image_size = image_size

        self.mode = mode
        self.sample_length = sample_length

        # Get all numbers
        self.folders = []
        for file in os.listdir(self.root):
            try:
                self.folders.append(int(file))
            except ValueError:
                continue
        self.folders.sort()

        self.epsisodes = []
        self.EP_LEN = ep_len
        self.seq_per_episode = self.EP_LEN - self.sample_length + 1

        for f in self.folders:
            dir_name = os.path.join(self.root, str(f))
            paths = list(glob.glob(osp.join(dir_name, '*.png')))
            # if len(paths) != self.EP_LEN:
            #     continue
            # assert len(paths) == self.EP_LEN, 'len(paths): {}'.format(len(paths))
            get_num = lambda x: int(osp.splitext(osp.basename(x))[0])
            paths.sort(key=get_num)
            self.epsisodes.append(paths)

    def __getitem__(self, index):

        imgs = []
        if self.mode == 'train':
            # Implement continuous indexing
            ep = index // self.seq_per_episode
            offset = index % self.seq_per_episode
            end = offset + self.sample_length

            e = self.epsisodes[ep]
            for image_index in range(offset, end):
                img = Image.open(osp.join(e[image_index]))
                img = img.resize((self.image_size, self.image_size))
                img = transforms.ToTensor()(img)[:3]
                imgs.append(img)
        else:
            for path in self.epsisodes[index]:
                img = Image.open(path)
                img = img.resize((self.image_size, self.image_size))
                img = transforms.ToTensor()(img)[:3]
                imgs.append(img)

        img = torch.stack(imgs, dim=0).float()
        pos = torch.zeros(0)
        size = torch.zeros(0)
        id = torch.zeros(0)
        in_camera = torch.zeros(0)

        return img, pos, size, id, in_camera

    def __len__(self):
        length = len(self.epsisodes)
        if self.mode == 'train':
            return length * self.seq_per_episode
        else:
            return length


class CLEVREREpDatasetImage(Dataset):
    def __init__(self, root, mode, ep_len=100, sample_length=20, image_size=128):
        # path = os.path.join(root, mode)
        assert mode in ['train', 'val', 'valid', 'test']
        if mode == 'val':
            mode = 'valid'
        self.root = os.path.join(root, mode)
        self.image_size = image_size

        self.mode = mode
        self.sample_length = sample_length

        # Get all numbers
        self.folders = []
        for file in os.listdir(self.root):
            try:
                self.folders.append(int(file))
            except ValueError:
                continue
        self.folders.sort()

        self.epsisodes = []
        self.EP_LEN = ep_len
        self.seq_per_episode = self.EP_LEN - self.sample_length + 1

        for f in self.folders:
            dir_name = os.path.join(self.root, str(f))
            paths = list(glob.glob(osp.join(dir_name, '*.png')))
            # if len(paths) != self.EP_LEN:
            #     continue
            # assert len(paths) == self.EP_LEN, 'len(paths): {}'.format(len(paths))
            get_num = lambda x: int(osp.splitext(osp.basename(x))[0])
            paths.sort(key=get_num)
            self.epsisodes.append(paths)

    def __getitem__(self, index):

        imgs = []
        # Implement continuous indexing
        ep = index // self.seq_per_episode
        offset = index % self.seq_per_episode
        end = offset + self.sample_length

        e = self.epsisodes[ep]
        for image_index in range(offset, end):
            img = Image.open(osp.join(e[image_index]))
            img = img.resize((self.image_size, self.image_size))
            img = transforms.ToTensor()(img)[:3]
            imgs.append(img)

        img = torch.stack(imgs, dim=0).float()
        pos = torch.zeros(0)
        size = torch.zeros(0)
        id = torch.zeros(0)
        in_camera = torch.zeros(0)

        return img, pos, size, id, in_camera

    def __len__(self):
        length = len(self.epsisodes)
        return length * self.seq_per_episode


class CLEVRERDataset(Dataset):
    def __init__(self, path_to_npy, image_size=128, transform=None, mode='single', train=True, horizon=3,
                 frames_per_video=34, video_as_index=False):
        super(CLEVRERDataset, self).__init__()
        assert mode in ['single', 'frames', 'tps', 'horizon']
        self.mode = mode
        self.frames_per_video = frames_per_video
        self.horizon = horizon if (horizon > 0 and self.mode == 'horizon') else self.frames_per_video
        self.train_mode = train
        if train:
            print(f'clevrer dataset mode: {self.mode}')
            if self.mode == 'horizon':
                print(f'time steps horizon: {self.horizon}')
        if self.mode == 'tps':
            # self.warper = tps.Warper(H=image_size, W=image_size, warpsd_all=0.00001,
            #                          warpsd_subset=0.001, transsd=0.1, scalesd=0.1,
            #                          rotsd=2, im1_multiplier=0.1, im1_multiplier_aff=0.1)
            pass
        else:
            self.warper = None
        data = np.load(path_to_npy)
        # train_size = int(0.9 * data.shape[0])
        # valid_size = data.shape[0] - train_size
        if train:
            self.data = data
            # self.data = data[:self.frames_per_video * 200]
            print(f'loaded data with shape: {self.data.shape}, size: {self.data.shape[0]}')
        else:
            self.data = data[:5000]
        self.image_size = image_size
        self.num_videos = self.data.shape[0] // self.frames_per_video
        self.video_as_index = video_as_index
        if transform is None:
            self.input_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(image_size),
                transforms.ToTensor()
            ])
        else:
            self.input_transform = transform

    def __getitem__(self, index):
        if not self.video_as_index:
            video_num = int(index / self.frames_per_video)
            video_start_idx = video_num * self.frames_per_video
            curr_idx = index % self.frames_per_video
            max_idx = min(video_start_idx + self.frames_per_video - 1, self.data.shape[0] - 1)
            global_idx = video_start_idx + curr_idx
            if self.mode == 'single':
                return self.input_transform(self.data[index])
            elif self.mode == 'frames':
                min_idx = min(video_start_idx, index - 1)
                if min_idx == video_start_idx:
                    im1 = self.input_transform(self.data[min_idx + 1])
                    im2 = self.input_transform(self.data[min_idx])
                else:
                    im1 = self.input_transform(self.data[min_idx])
                    im2 = self.input_transform(self.data[min_idx - 1])
                return im1, im2
            elif self.mode == 'horizon':
                images = []
                length = max_idx
                if (index + self.horizon) >= length:
                    slack = index + self.horizon - length
                    index = index - slack
                for i in range(self.horizon):
                    t = index + i
                    images.append(self.input_transform(self.data[t]))
                images = torch.stack(images, dim=0)
                return images
            elif self.mode == 'tps':
                im = self.input_transform(self.data[index])
                im = im * 255
                im2, im1, _, _, _, _ = self.warper(im)
                return im1 / 255, im2 / 255
            else:
                raise NotImplementedError
        else:
            video_num = index
            video_start_idx = video_num * self.frames_per_video
            max_idx = video_start_idx + self.frames_per_video - 1
            images = []
            length = max_idx
            frame_idx = video_start_idx
            actual_horizon = self.frames_per_video if ((frame_idx + self.horizon) >= length) else self.horizon
            for i in range(actual_horizon):
                t = frame_idx + i
                images.append(self.input_transform(self.data[t]))
            images = torch.stack(images, dim=0)
            return images

    def __len__(self):
        if not self.video_as_index:
            return self.data.shape[0]
        else:
            return self.num_videos


if __name__ == '__main__':
    # -- single image setting --- #
    path_to_img = '/media/newhd/data/clevrer/train/frames/'
    # prepare_numpy_file(path_to_img, image_size=128, frameskip=3, start_frame=26)
    test_epochs = True
    # load data
    # path_to_npy = '/media/newhd/data/clevrer/valid/clevrer_img128np_fs3_valid.npy'
    # mode = 'frames'
    # horizon = 4
    # train = True
    # clevrer_ds = CLEVRERDataset(path_to_npy, mode=mode, train=train, horizon=horizon)
    # clevrer_dl = DataLoader(clevrer_ds, shuffle=True, pin_memory=True, batch_size=5)
    # batch = next(iter(clevrer_dl))
    # if mode == 'single':
    #     im1 = batch[0]
    # elif mode == 'frames' or mode == 'tps':
    #     im1 = batch[0][0]
    #     im2 = batch[1][0]
    #
    # if mode == 'single':
    #     print(im1.shape)
    #     img_np = im1.permute(1, 2, 0).data.cpu().numpy()
    #     fig = plt.figure(figsize=(5, 5))
    #     ax = fig.add_subplot(111)
    #     ax.imshow(img_np)
    # elif mode == 'horizon':
    #     print(f'batch shape: {batch.shape}')
    #     images = batch[0]
    #     print(f'images shape: {images.shape}')
    #     fig = plt.figure(figsize=(8, 8))
    #     for i in range(images.shape[0]):
    #         ax = fig.add_subplot(1, horizon, i + 1)
    #         im = images[i]
    #         im_np = im.permute(1, 2, 0).data.cpu().numpy()
    #         ax.imshow(im_np)
    #         ax.set_title(f'im {i + 1}')
    # else:
    #     print(f'im1: {im1.shape}, im2: {im2.shape}')
    #     im1_np = im1.permute(1, 2, 0).data.cpu().numpy()
    #     im2_np = im2.permute(1, 2, 0).data.cpu().numpy()
    #     fig = plt.figure(figsize=(8, 8))
    #     ax = fig.add_subplot(1, 2, 1)
    #     ax.imshow(im1_np)
    #     ax.set_title('im1')
    #
    #     ax = fig.add_subplot(1, 2, 2)
    #     ax.imshow(im2_np)
    #     ax.set_title('im2 [t-1] or [tps]')
    # plt.show()
    # --- end single image --- #

    # --- episodic setting --- #
    root = '/media/newhd/data/clevrer/episodes'
    clevrer_ds = CLEVREREpDataset(root=root, ep_len=100, sample_length=30, mode='train')
    clevrer_dl = DataLoader(clevrer_ds, shuffle=True, pin_memory=True, batch_size=5)
    batch = next(iter(clevrer_dl))
    im = batch[0][0][0]
    print(im.shape)
    img_np = im.permute(1, 2, 0).data.cpu().numpy()
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.imshow(img_np)
    plt.show()

    if test_epochs:
        from tqdm import tqdm

        pbar = tqdm(iterable=clevrer_dl)
        for batch in pbar:
            pass
        pbar.close()
