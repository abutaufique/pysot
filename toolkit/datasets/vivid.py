import os
import pdb
import cv2
import json
import numpy as np
from os.path import join, isdir
from os import listdir
from glob import glob
from tqdm import tqdm
from PIL import Image
import glob
from .dataset import Dataset
from .video import Video

class MyVideo(Video):
    def __iter__(self):
        for i in range(len(self.img_names)):
            if self.imgs is not None:
                yield (self.imgs[i], self.init_rect) if i==0 else (self.imgs[i], None)
            else:
                yield (cv2.imread(self.img_names[i]),self.init_rect) if i==0 else (cv2.imread(self.img_names[i]),None)

class VIVIDVideo(MyVideo):
    """
    Args:
        name: video name
        root: dataset root
        video_dir: video directory
        init_rect: init rectangle
        img_names: image names
        gt_rect: groundtruth rectangle
        camera_motion: camera motion tag
        illum_change: illum change tag
        motion_change: motion change tag
        size_change: size change
        occlusion: occlusion
    """
    def __init__(self, name, root, video_dir, init_rect, img_names, gt_rect=None,
            camera_motion=None, illum_change=None, motion_change=None, size_change=None, occlusion=None, load_img=False):
        super(VIVIDVideo, self).__init__(name, root, video_dir,
                init_rect, img_names, gt_rect, None, load_img)
        self.tags= {'all': [1] * len(img_names)}
        self.tags['camera_motion'] = camera_motion
        self.tags['illum_change'] = illum_change
        self.tags['motion_change'] = motion_change
        self.tags['size_change'] = size_change
        self.tags['occlusion'] = occlusion

        # TODO
        # if len(self.gt_traj[0]) == 4:
        #     self.gt_traj = [[x[0], x[1], x[0], x[1]+x[3]-1,
        #                     x[0]+x[2]-1, x[1]+x[3]-1, x[0]+x[2]-1, x[1]]
        #                         for x in self.gt_traj]

        # empty tag
        all_tag = [v for k, v in self.tags.items() if v is not None ]
        self.tags['empty'] = np.all(1 - np.array(all_tag), axis=1).astype(np.int32).tolist()
        # self.tags['empty'] = np.all(1 - np.array(list(self.tags.values())),
        #         axis=1).astype(np.int32).tolist()

        self.tag_names = list(self.tags.keys())
        if not load_img:
            img_name = os.path.join(root, self.img_names[0])
            img = np.array(Image.open(img_name), np.uint8)
            self.width = img.shape[1]
            self.height = img.shape[0]

    def select_tag(self, tag, start=0, end=0):
        if tag == 'empty':
            return self.tags[tag]
        return self.tags[tag][start:end]

    def load_tracker(self, path, tracker_names=None, store=True):
        """
        Args:
            path(str): path to result
            tracker_name(list): name of tracker
        """
        if not tracker_names:
            tracker_names = [x.split('/')[-1] for x in glob(path)
                    if os.path.isdir(x)]
        if isinstance(tracker_names, str):
            tracker_names = [tracker_names]
        for name in tracker_names:
            traj_files = glob(os.path.join(path, name, 'baseline', self.name, '*0*.txt'))
            if len(traj_files) == 15:
                traj_files = traj_files
            else:
                traj_files = traj_files[0:1]
            pred_traj = []
            for traj_file in traj_files:
                with open(traj_file, 'r') as f:
                    traj = [list(map(float, x.strip().split(',')))
                            for x in f.readlines()]
                    pred_traj.append(traj)
            if store:
                self.pred_trajs[name] = pred_traj
            else:
                return pred_traj

def get_metadata(dataset_root):
    #base_path = join(realpath(dirname(__file__)), '../data', 'VIVID')
    base_path = dataset_root
    info={}
    videos = sorted([item for item in listdir(base_path) if (isdir(join(base_path, item)) and ('mask' not in (item)))])
    for video in videos:
        video_path = join(base_path, video)
        image_path = join(video_path, '*.jpg')
        image_files = sorted(glob.glob(image_path))
        init_bbox = {'egtest01': [119, 15, 21, 25], 'egtest02': [437, 369, 46, 37], 'egtest03': [439, 274, 49, 33], 'egtest04': [321, 171, 16, 11], 'egtest05': [21, 322, 87, 43], 'pktest01': [114, 123, 30, 14], 'pktest02': [249, 164, 32, 20], 'pktest03': [142, 117, 18, 10]}
        gt = np.array(init_bbox[video]).astype(np.float64).reshape(-1,4)
        if gt.shape[1] == 4:
            gt = np.column_stack((gt[:, 0], gt[:, 1], gt[:, 0], gt[:, 1] + gt[:, 3]-1,
                                  gt[:, 0] + gt[:, 2]-1, gt[:, 1] + gt[:, 3]-1, gt[:, 0] + gt[:, 2]-1, gt[:, 1]))
        gt = np.squeeze(gt).tolist()
        info[video] = {'image_files': image_files, 'gt': gt, 'name': video}
    return info

class VIVIDDataset(Dataset):
    """
    Args:
        name: dataset name, should be 'VOT2018', 'VOT2016', 'VOT2019'
        dataset_root: dataset root
        load_img: wether to load all imgs
    """
    def __init__(self, name, dataset_root, load_img=False):
        super(VIVIDDataset, self).__init__(name, dataset_root)
        meta_data = get_metadata(dataset_root)
        # load videos
        pbar = tqdm(meta_data.keys(), desc='loading '+name, ncols=100)
        self.videos = {}
        for video in pbar:
            pbar.set_postfix_str(video)
            self.videos[video] = VIVIDVideo(video,
                                          dataset_root,
                                          join(dataset_root, name),
                                          meta_data[video]['gt'],
                                          meta_data[video]['image_files'],
                                          load_img=load_img)

        self.tags = ['all', 'camera_motion', 'illum_change', 'motion_change',
                     'size_change', 'occlusion', 'empty']


