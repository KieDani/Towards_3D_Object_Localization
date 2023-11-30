import random
import numpy as np
import os
from torch.utils.data.dataset import Dataset
import pandas as pd
import cv2
from tqdm import tqdm
from helper import create_heatmaps, get_data_path
import einops as eo
import torch

data_set_size = 90000
video_length = 30
#num_cameras = 9


class ParcourDataset(Dataset):
    def __init__(self, mode='train', get_heatmap=True, multi_camera=False, num_env=1, transform=None):
        self.num_cameras = 9 if multi_camera else 1
        assert num_env in [1, 2, 3]
        self.num_environments = num_env
        self.get_heatmap = get_heatmap
        self.video_length = video_length
        self.original_size = (224, 224)
        self.FPS = 15
        self.transform = transform

        if mode == 'train':
            self.path = os.path.join(get_data_path(), 'synthetic_dataset', 'train')
        elif mode == 'val':
            self.path = os.path.join(get_data_path(), 'synthetic_dataset', 'val')
        elif mode == 'test':
            self.path = os.path.join(get_data_path(), 'synthetic_dataset', 'test')
        else:
            raise ValueError('mode must be one of train, val or test')

        videos = {}
        if multi_camera == False:
            cameras = ['camera0']
            for xml_num in range(self.num_environments):
                vids = [(cameras[0], v) for v in sorted(os.listdir(os.path.join(self.path, f'xml{0}', cameras[0])))]
                rand = random.Random(42)
                rand.shuffle(vids)
                if xml_num not in videos.keys(): videos[xml_num] = []
                videos[xml_num] += vids
        else:
            cameras = [f'camera{i}' for i in range(self.num_cameras - 3)] if mode=='train' else [f'camera{i}' for i in range(self.num_cameras)]
            for xml_num in range(self.num_environments):
                for cam in cameras:
                    vids = [(cam, v) for v in sorted(os.listdir(os.path.join(self.path, f'xml{xml_num}', cam)))]
                    rand = random.Random(42)
                    rand.shuffle(vids)
                    if xml_num not in videos.keys(): videos[xml_num] = []
                    videos[xml_num] += vids
        if mode=='train':
            for xml_num in range(self.num_environments):
                rand = random.Random(42)
                rand.shuffle(videos[xml_num])
                if data_set_size // self.num_environments < len(videos) and data_set_size > 0: videos[xml_num] = videos[xml_num][:data_set_size]
        self.videos = videos
        self.cameras = cameras

        annotations = {}
        for xml_num, video_list in self.videos.items():
            for cam_num, video in tqdm(video_list):
                frame_list = list()
                df = pd.read_csv(os.path.join(self.path, f'xml{xml_num}', cam_num, video, 'label.csv'))
                keys = list(df.keys())
                for i, row in df.iterrows():
                    if i > self.video_length: break
                    frame_dict = dict()
                    frame_dict['frame'] = i
                    frame_dict['path'] = os.path.join(self.path, f'xml{xml_num}', cam_num, video)
                    for key in keys:
                        frame_dict[key] = row[key]
                    frame_list.append(frame_dict)
                extM, intM = self.create_matrix(cam_num, video, xml_num)
                matrix_dict = {}
                matrix_dict['extM'] = extM
                matrix_dict['intM'] = intM
                annots = (video, frame_list, matrix_dict, cam_num)
                if xml_num not in annotations.keys(): annotations[xml_num] = []
                annotations[xml_num].append(annots)
        # structure of annotations: {xml_num: [(video, frame_list, matrix_dict, cam_num), ...], ...}
        self.annotations = annotations

    def create_matrix(self, camera, video, xml_num):
        with open(os.path.join(self.path, f'xml{xml_num}', camera, video, 'info.txt')) as f:
            info_file = f.readlines()
        e_l1 = info_file[1].split(',')
        e_l2 = info_file[2].split(',')
        e_l3 = info_file[3].split(',')
        e_l4 = info_file[4].split(',')

        i_l1 = info_file[6].split(',')
        i_l2 = info_file[7].split(',')
        i_l3 = info_file[8].split(',')
        i_l4 = info_file[9].split(',')

        extM = np.array([[float(e_l1[0]), float(e_l1[1]), float(e_l1[2]), float(e_l1[3])],
                              [float(e_l2[0]), float(e_l2[1]), float(e_l2[2]), float(e_l2[3])],
                              [float(e_l3[0]), float(e_l3[1]), float(e_l3[2]), float(e_l3[3])],
                              [float(e_l4[0]), float(e_l4[1]), float(e_l4[2]), float(e_l4[3])]], dtype=np.float32)
        intM = np.array([[float(i_l1[0]), float(i_l1[1]), float(i_l1[2]), float(i_l1[3])],
                              [float(i_l2[0]), float(i_l2[1]), float(i_l2[2]), float(i_l2[3])],
                              [float(i_l3[0]), float(i_l3[1]), float(i_l3[2]), float(i_l3[3])],
                              [float(i_l4[0]), float(i_l4[1]), float(i_l4[2]), float(i_l4[3])]], dtype=np.float32)
        return extM, intM

    def __len__(self):
        min_length = 999_999_999
        for annot_list in self.annotations.values():
            min_length = min(len(annot_list), min_length)
        return min_length

    def __getitem__(self, idx):
        data_dict = {}
        for xml_num, annot_list in self.annotations.items():
            vid, frame_list, matrix_dict, cam = annot_list[idx]
            cam_num = int(cam.strip('camera'))
            video = list()
            label = list()
            camlabel = list()
            timestamps = list()
            for frame_dict in frame_list[:self.video_length]:
                frame = 'image-' + str(frame_dict['frame']) + '.png'
                frame_path = os.path.join(frame_dict['path'], frame)
                image = cv2.imread(frame_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = eo.rearrange(image, 'h w c -> c h w')
                image = image.astype(np.float32) / 255.
                image = torch.from_numpy(image)
                if self.transform is not None:
                    image = self.transform({'image': image})['image']
                C, H, W = image.shape

                video.append(image)
                label.append([H - frame_dict['ball imgy'], frame_dict['ball imgx']])
                camlabel.append([frame_dict['ball camy'], frame_dict['ball camx'], frame_dict['ball camz']])
                timestamps.append(frame_dict['frame'] / self.FPS)
            video = torch.stack(video)
            label = np.asarray(label, dtype=np.float32)
            timestamps = np.asarray(timestamps, dtype=np.float32)

            heatmaps = None
            if self.get_heatmap:
                heatmaps = list()
                for i in range(label.shape[0]):
                    #x, y, visibility
                    heatmap = create_heatmaps((label[i, 0], label[i, 1], 1.), (H, W), sigma=2)
                    heatmaps.append(heatmap)
                heatmaps = np.array(heatmaps, dtype=np.float32)

            Mint, Mext = matrix_dict['intM'], matrix_dict['extM']
            Mint[1, 1] *= -1 # because label = [H - frame_dict['ball imgy'], frame_dict['ball imgx']]
            data_dict[xml_num] = (video, label, heatmaps, Mext, Mint, np.asarray(camlabel, dtype=np.float32), cam_num, timestamps)
        # structure of data_dict: {xml_num: (video, label, heatmaps, extM, intM, camlabel, cam_num), ...}
        return data_dict


class CarouselDataset(Dataset):
    def __init__(self, mode='train', get_heatmap=True, transform=None):
        self.num_cameras = 1
        self.num_environments = 1
        self.get_heatmap = get_heatmap
        self.video_length = video_length
        self.original_size = (224, 224)
        self.FPS = 15
        self.transform = transform

        if mode == 'train':
            self.path = os.path.join(get_data_path(), 'ball_carousel', 'train')
        elif mode == 'val':
            self.path = os.path.join(get_data_path(), 'ball_carousel', 'val')
        elif mode == 'test':
            self.path = os.path.join(get_data_path(), 'ball_carousel', 'test')
        else:
            raise ValueError('mode must be either train, val or test')

        videos = sorted(os.listdir(self.path))
        rand = random.Random(42)
        rand.shuffle(videos)
        if data_set_size < len(videos) and data_set_size > 0 and mode=='train': videos = videos[:data_set_size]
        self.videos = videos

        # elements of annotations: (video, frame_list)
        annotations = list()
        for video in tqdm(self.videos):
            # elements of frame_list: dictionarys with labels and frame_nr for this frame as elements
            frame_list = list()
            df = pd.read_csv(os.path.join(self.path, video, 'label.csv'))
            keys = list(df.keys())
            ball_is_visible = True
            for i, row in df.iterrows():
                if i > self.video_length: break
                frame_dict = dict()
                frame_dict['frame'] = i
                frame_dict['path'] = os.path.join(self.path, video)
                for key in keys:
                    frame_dict[key] = row[key]
                frame_list.append(frame_dict)
            annotations.append((video, frame_list))
        self.annotations = annotations

        with open(os.path.join(self.path, video, 'info.txt')) as f:
            info_file = f.readlines()
        e_l1 = info_file[1].split(',')
        e_l2 = info_file[2].split(',')
        e_l3 = info_file[3].split(',')
        e_l4 = info_file[4].split(',')

        i_l1 = info_file[6].split(',')
        i_l2 = info_file[7].split(',')
        i_l3 = info_file[8].split(',')
        i_l4 = info_file[9].split(',')

        self.extM = np.array([[float(e_l1[0]), float(e_l1[1]), float(e_l1[2]), float(e_l1[3])],
                              [float(e_l2[0]), float(e_l2[1]), float(e_l2[2]), float(e_l2[3])],
                              [float(e_l3[0]), float(e_l3[1]), float(e_l3[2]), float(e_l3[3])],
                              [float(e_l4[0]), float(e_l4[1]), float(e_l4[2]), float(e_l4[3])]], dtype=np.float32)
        self.intM = np.array([[float(i_l1[0]), float(i_l1[1]), float(i_l1[2]), float(i_l1[3])],
                              [float(i_l2[0]), -float(i_l2[1]), float(i_l2[2]), float(i_l2[3])],
                              [float(i_l3[0]), float(i_l3[1]), float(i_l3[2]), float(i_l3[3])],
                              [float(i_l4[0]), float(i_l4[1]), float(i_l4[2]), float(i_l4[3])]], dtype=np.float32)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        vid, frame_list = self.annotations[idx]
        video = list()
        label = list()
        camlabel = list()
        timestamps = list()
        for frame_dict in frame_list[:self.video_length]:
            frame = 'image-' + str(frame_dict['frame']) + '.png'
            frame_path = os.path.join(frame_dict['path'], frame)
            # print(frame_path)
            image = cv2.imread(frame_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = eo.rearrange(image, 'h w c -> c h w')
            image = image.astype(np.float32) / 255.
            image = torch.from_numpy(image)
            if self.transform is not None:
                image = self.transform({'image': image})['image']
            C, H, W = image.shape
            # 224x224
            # factor_h = 180 / H
            # factor_w = 180 / W
            # image = zoom(image, (1, factor_h, factor_w), order=3)

            video.append(image)
            label.append([H - frame_dict['ball imgy'], frame_dict['ball imgx']])
            camlabel.append([frame_dict['ball camy'], frame_dict['ball camx'], frame_dict['ball camz']])
            timestamps.append(frame_dict['frame'] / self.FPS)
        video = torch.stack(video)
        label = np.asarray(label, dtype=np.float32)
        timestamps = np.asarray(timestamps, dtype=np.float32)

        heatmaps = None
        if self.get_heatmap:
            heatmaps = list()
            for i in range(label.shape[0]):
                #x, y, visibility
                heatmap = create_heatmaps((label[i, 0], label[i, 1], 1.), (H, W), sigma=2)
                heatmaps.append(heatmap)
            heatmaps = np.array(heatmaps, dtype=np.float32)
        #Only one camera and environment for this dataset
        data_dict = {
            0: (video, label, heatmaps, self.extM, self.intM,
                np.asarray(camlabel, dtype=np.float32), 0, timestamps)
        }
        return data_dict


class BallDataset(Dataset):
    def __init__(self, mode='train', get_heatmap=True, transform=None):
        self.num_cameras = 1
        self.num_environments = 1
        self.get_heatmap = get_heatmap
        self.video_length = video_length
        self.original_size = (224, 224)
        self.FPS = 15
        self.transform = transform

        if mode == 'train':
            self.path = os.path.join(get_data_path(), 'ball_bouncing', 'train')
        elif mode == 'val':
            self.path = os.path.join(get_data_path(), 'ball_bouncing', 'val')
        elif mode == 'test':
            self.path = os.path.join(get_data_path(), 'ball_bouncing', 'test')
        else:
            raise ValueError('mode must be train, val or test')

        videos = sorted(os.listdir(self.path))
        rand = random.Random(42)
        rand.shuffle(videos)
        if data_set_size < len(videos) and data_set_size > 0 and mode == 'train': videos = videos[:data_set_size]
        self.videos = videos

        # elements of annotations: (video, frame_list)
        annotations = list()
        for video in tqdm(self.videos):
            # elements of frame_list: dictionarys with labels and frame_nr for this frame as elements
            frame_list = list()
            df = pd.read_csv(os.path.join(self.path, video, 'label.csv'))
            keys = list(df.keys())
            ball_is_visible = True
            for i, row in df.iterrows():
                if i > self.video_length: break
                frame_dict = dict()
                frame_dict['frame'] = i
                frame_dict['path'] = os.path.join(self.path, video)
                for key in keys:
                    frame_dict[key] = row[key]
                frame_list.append(frame_dict)
            annotations.append((video, frame_list))
        self.annotations = annotations

        with open(os.path.join(self.path, video, 'info.txt')) as f:
            info_file = f.readlines()
        e_l1 = info_file[1].split(',')
        e_l2 = info_file[2].split(',')
        e_l3 = info_file[3].split(',')
        e_l4 = info_file[4].split(',')

        i_l1 = info_file[6].split(',')
        i_l2 = info_file[7].split(',')
        i_l3 = info_file[8].split(',')
        i_l4 = info_file[9].split(',')

        self.extM = np.array([[float(e_l1[0]), float(e_l1[1]), float(e_l1[2]), float(e_l1[3])],
                              [float(e_l2[0]), float(e_l2[1]), float(e_l2[2]), float(e_l2[3])],
                              [float(e_l3[0]), float(e_l3[1]), float(e_l3[2]), float(e_l3[3])],
                              [float(e_l4[0]), float(e_l4[1]), float(e_l4[2]), float(e_l4[3])]], dtype=np.float32)
        self.intM = np.array([[float(i_l1[0]), float(i_l1[1]), float(i_l1[2]), float(i_l1[3])],
                              [float(i_l2[0]), -float(i_l2[1]), float(i_l2[2]), float(i_l2[3])],
                              [float(i_l3[0]), float(i_l3[1]), float(i_l3[2]), float(i_l3[3])],
                              [float(i_l4[0]), float(i_l4[1]), float(i_l4[2]), float(i_l4[3])]], dtype=np.float32)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        data_dict = {}
        vid, frame_list = self.annotations[idx]
        video = list()
        label = list()
        camlabel = list()
        timestamps = list()
        for frame_dict in frame_list[:self.video_length]:
            frame = 'image-' + str(frame_dict['frame']) + '.png'
            frame_path = os.path.join(frame_dict['path'], frame)
            image = cv2.imread(frame_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = eo.rearrange(image, 'h w c -> c h w')
            image = image.astype(np.float32) / 255.
            image = torch.from_numpy(image)
            if self.transform is not None:
                image = self.transform({'image': image})['image']
            C, H, W = image.shape

            video.append(image)
            label.append([H - frame_dict['ball imgy'], frame_dict['ball imgx']])
            camlabel.append([frame_dict['ball camy'], frame_dict['ball camx'], frame_dict['ball camz']])
            timestamps.append(frame_dict['frame'] / self.FPS)
        video = torch.stack(video)
        label = np.asarray(label, dtype=np.float32)
        timestamps = np.asarray(timestamps, dtype=np.float32)

        heatmaps = None
        if self.get_heatmap:
            heatmaps = list()
            for i in range(label.shape[0]):
                #x, y, visibility
                heatmap = create_heatmaps((label[i, 0], label[i, 1], 1.), (H, W), sigma=2)
                heatmaps.append(heatmap)
            heatmaps = np.array(heatmaps, dtype=np.float32)
        # Only one camera and environment for this dataset
        data_dict = {
            0: (video, label, heatmaps, self.extM, self.intM,
                np.asarray(camlabel, dtype=np.float32), 0, timestamps)
        }
        return data_dict


class RealBallDataset(Dataset):
    def __init__(self, mode='train', get_heatmap=True, transform=None):
        self.num_cameras = 2
        self.num_environments = 1
        self.get_heatmap = get_heatmap
        self.video_length = video_length
        self.transform = transform
        self.mode = mode
        self.heatmap_factor = 1

        if mode == 'train':
            self.path = os.path.join(get_data_path(), 'real_dataset', 'train')
        elif mode == 'val':
            self.path = os.path.join(get_data_path(), 'real_dataset', 'val')
        elif mode == 'test':
            self.path = os.path.join(get_data_path(), 'real_dataset', 'test')
        else:
            raise ValueError('Unknown mode: {}'.format(mode))

        #load video paths: videos: [path1, path2, path3, ...]
        videos = []
        for c in sorted(os.listdir(self.path)):
            cam_num = int(c.replace('camera', '')) - 1
            video_list = [o for o in os.listdir(os.path.join(self.path, c)) if os.path.isdir(os.path.join(self.path, c, o))]
            for v in sorted(video_list):
                num_frames = len([f for f in os.listdir(os.path.join(self.path, c, v)) if f.endswith('.png') and f.startswith('frame')])
                num_videos = num_frames // self.video_length
                for part in range(num_videos):
                    num_part = self.video_length if part < num_videos - 1 else num_frames - part * self.video_length
                    videos.append({
                        'video': os.path.join(self.path, c, v),
                        'num_frames': num_part,
                        'cam_num': cam_num,
                        'start_frame': part * self.video_length,
                    })
        rand = random.Random(42)
        rand.shuffle(videos)
        self.videos = videos

        video_path = self.videos[0]['video']
        metadata_df = pd.read_csv(os.path.join(video_path, 'metadata.csv'), sep=';')
        self.original_size = (metadata_df['original_height'].item(), metadata_df['original_width'].item())


    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path = self.videos[idx]['video']
        num_frames = self.videos[idx]['num_frames']
        cam_num = self.videos[idx]['cam_num']
        start_frame = self.videos[idx]['start_frame']

        if num_frames > self.video_length:
            if self.mode == 'train':
                start_frame = random.randint(0, num_frames - self.video_length)
            else:
                start_frame = (num_frames - self.video_length) // 2
        end_frame = start_frame + self.video_length

        labels_df = pd.read_csv(os.path.join(video_path, 'labels.csv'), sep=';')
        metadata_df = pd.read_csv(os.path.join(video_path, 'metadata.csv'), sep=';')
        timestamps_df = pd.read_csv(os.path.join(video_path, 'timestamps.csv'), sep=';')['timestamp']
        original_width = metadata_df['original_width'][0]
        original_height = metadata_df['original_height'][0]
        resized_width = metadata_df['resized_width'][0]
        resized_height = metadata_df['resized_height'][0]

        matrix_dict = np.load(os.path.join(video_path, '..', 'matrices.npz'))
        Mint = matrix_dict['Mint']
        Mext = matrix_dict['Mext']
        Mext = np.asarray(Mext, dtype=np.float32)
        Mint = np.asarray(Mint, dtype=np.float32)

        video = []
        label = []
        camlabel = []
        timestamps = []
        for i in range(start_frame, end_frame):
            frame_path = os.path.join(video_path, f'frame{i:04}.png')
            image = cv2.imread(frame_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = eo.rearrange(image, 'h w c -> c h w')
            image = image.astype(np.float32) / 255.
            image = torch.from_numpy(image)
            if self.transform is not None:
                image = self.transform({'image': image, 'heatmap': None})['image']
            video.append(image)

            ball_x = labels_df['ball_x'][i]
            ball_y = labels_df['ball_y'][i]
            label.append([ball_y, ball_x])

            depth = cv2.imread(os.path.join(video_path, f'depth{i:04}.png'), cv2.IMREAD_GRAYSCALE)
            H, W = depth.shape
            maxdepth = int(metadata_df['max_depth'])
            depth = depth.astype(np.float32) / 255. * maxdepth
            #z_c = depth[int(ball_y)-1:int(ball_y)+2, int(ball_x)-1:int(ball_x)+2].mean()
            num_vaid = np.where(depth[int(ball_y) - 1:int(ball_y) + 2, int(ball_x) - 1:int(ball_x) + 2] == 0., 0, 1).sum()
            z_c = depth[int(ball_y) - 1:int(ball_y) + 2, int(ball_x) - 1:int(ball_x) + 2].sum() / num_vaid if num_vaid > 0 else 0.

            orig_ball_x = original_width / W * (ball_x + 0.5) - 0.5
            orig_ball_y = original_height / H * (ball_y + 0.5) - 0.5

            original_coords = np.array([orig_ball_x, orig_ball_y, z_c], dtype=np.float32)
            original_coords[[0, 1]] *= original_coords[2]
            inv_Mint = np.linalg.inv(Mint[:3, :3])
            camera_coords = np.einsum('ij,j->i', inv_Mint, original_coords)
            # (x, y, z) -> (y, x, z)
            camera_coords = camera_coords[[1, 0, 2]]

            camlabel.append(camera_coords)
            timestamp = timestamps_df[i] * 10**(-9) - timestamps_df[start_frame] * 10**(-9)
            timestamps.append(timestamp)


        video = torch.stack(video)
        label = np.asarray(label, dtype=np.float32)
        camlabel = np.asarray(camlabel, dtype=np.float32)
        timestamps = np.asarray(timestamps, dtype=np.float32)

        heatmaps = None
        if self.get_heatmap:
            heatmaps = list()
            for i in range(label.shape[0]):
                # x, y, visibility
                heatmap = create_heatmaps((label[i, 0], label[i, 1], 1.),
                                                 (int(metadata_df['resized_height']), int(metadata_df['resized_width'])), sigma=2,
                                                 factor=self.heatmap_factor)
                heatmaps.append(heatmap)

            heatmaps = torch.from_numpy(np.array(heatmaps, dtype=np.float32))
        label = torch.from_numpy(label)
        Mext = torch.from_numpy(Mext)
        Mint = torch.from_numpy(Mint)

        data_dict = {
            0: (video, label, heatmaps, Mext, Mint, camlabel, cam_num, timestamps)
        }

        return data_dict



def get_dataset(dataset_name='parcour', mode='train', transforms=None, video_len=None):
    assert mode in ['train', 'val', 'test']
    assert dataset_name in ['falling', 'carousel', 'parcour_singleenv_singlecam', 'parcour_singleenv_multicam',
                                'parcour_multienv_singlecam', 'parcour_multienv_multicam', 'realball',
                                'parcour_dualenv_multicam'
                            ]

    global video_length
    if dataset_name == 'realball':
        video_length = 16 if video_len is None else video_len
    else:
        video_length = video_length if video_len is None else video_len

    if dataset_name == 'parcour_singleenv_singlecam':
        return ParcourDataset(mode=mode, multi_camera=False, num_env=1, transform=transforms)
    elif dataset_name == 'parcour_singleenv_multicam':
        return ParcourDataset(mode=mode, multi_camera=True, num_env=1, transform=transforms)
    elif dataset_name == 'falling':
        return BallDataset(mode=mode, transform=transforms)
    elif dataset_name == 'carousel':
        return CarouselDataset(mode=mode, transform=transforms)
    elif dataset_name == 'parcour_multienv_multicam':
        return ParcourDataset(mode=mode, multi_camera=True, num_env=3, transform=transforms)
    elif dataset_name == 'parcour_multienv_singlecam':
        return ParcourDataset(mode=mode, multi_camera=False, num_env=3, transform=transforms)
    elif dataset_name == 'realball':
        return RealBallDataset(mode=mode, transform=transforms)
    elif dataset_name == 'parcour_dualenv_multicam':
        return ParcourDataset(mode=mode, multi_camera=True, num_env=2, transform=transforms)
    else:
        print('Dataset not found!!!')
        exit()


def test_datasets():
    dataset = get_dataset(dataset_name='realball', mode='train', transforms=None)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for i, data_dict in enumerate(tqdm(dataloader)):
        vid = data_dict[0][0]
        #print(vid.shape)
    pass

if __name__ == '__main__':
    test_datasets()
