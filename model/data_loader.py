# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import json


class VideoData(Dataset):
    def __init__(self, mode, video_type, split_index):
        """ Custom Dataset class wrapper for loading the frame features and ground truth importance scores.

        :param str mode: The mode of the model, train or test.
        :param str video_type: The Dataset being used, SumMe or TVSum.
        :param int split_index: The index of the Dataset split being used.
        """
        self.mode = mode
        self.name = video_type.lower()
        self.datasets = ['../PGL-SUM/data/datasets/SumMe/eccv16_dataset_summe_google_pool5.h5',
                         '../PGL-SUM/data/datasets/TVSum/eccv16_dataset_tvsum_google_pool5.h5',
                         '../../summarization_dataset/yt8m_sum_all.h5'] ####
        if (self.name != 'yt8m_sum'):
            self.splits_filename = ['../PGL-SUM/data/datasets/splits/622' + self.name + '_val_splits.json'] 
        else:
            self.splits_filename = ['../PGL-SUM/data/datasets/splits/' + self.name + '_split_27892_2000_2000.json'] ####
        self.split_index = split_index  # it represents the current split (varies from 0 to 4)

        if 'summe' in self.splits_filename[0]:
            self.filename = self.datasets[0]
        elif 'tvsum' in self.splits_filename[0]:
            self.filename = self.datasets[1]
        ####
        elif 'yt8m_sum' in self.splits_filename[0]:
            self.filename = self.datasets[2]    
        ###
        hdf = h5py.File(self.filename, 'r')
        self.list_frame_features, self.list_gtscores, self.list_user_summary = [], [], []
        self.list_sb, self.list_n_frames, self.list_positions = [], [], []


        with open(self.splits_filename[0]) as f:
            data = json.loads(f.read())
            for i, split in enumerate(data):
                if i == self.split_index:
                    self.split = split
                    break
        
        if self.name == 'yt8m_sum':
            for video_name in self.split[self.mode + '_keys']:
                frame_features = torch.Tensor(np.array(hdf[video_name + '/features']))
                gtscore = torch.Tensor(np.array(hdf[video_name + '/gtscore']))

                # In this case, it is gt_summary, not user_summary, but for consistence I will leave it for now
                user_summary = np.array(hdf[f"{video_name}/gt_summary"])
                sb = np.array(hdf[f"{video_name}/change_points"])
                
                ## yonsoo's code
                n_frames = frame_features.shape[0]
                #sb = np.append(sb, np.array([[sb[-1][0], n_frames-1]]), axis=0)
                positions = np.array([i for i in range(int(n_frames))])
                #n_frames = np.array([cp[1]-cp[0] for cp in sb])
                #sb = np.array(hdf[f"{video_name}/change_points"])
                ####

                self.list_frame_features.append(frame_features)
                self.list_gtscores.append(gtscore)
                self.list_user_summary.append(user_summary)
                self.list_sb.append(sb)
                self.list_n_frames.append(n_frames)
                self.list_positions.append(positions)

        else: 
            for video_name in self.split[self.mode + '_keys']:
                frame_features = torch.Tensor(np.array(hdf[video_name + '/features']))
                gtscore = torch.Tensor(np.array(hdf[video_name + '/gtscore']))
                user_summary = np.array(hdf[f"{video_name}/user_summary"])
                sb = np.array(hdf[f"{video_name}/change_points"])
                n_frames = np.array(hdf[f"{video_name}/n_frames"])
                positions = np.array(hdf[f"{video_name}/picks"])

                self.list_frame_features.append(frame_features)
                self.list_gtscores.append(gtscore)
                self.list_user_summary.append(user_summary)
                self.list_sb.append(sb)
                self.list_n_frames.append(n_frames)
                self.list_positions.append(positions)


        hdf.close()

    def __len__(self):
        """ Function to be called for the `len` operator of `VideoData` Dataset. """
        self.len = len(self.split[self.mode+'_keys'])
        return self.len

    def __getitem__(self, index):
        """ Function to be called for the index operator of `VideoData` Dataset.
        train mode returns: frame_features and gtscores
        test mode returns: frame_features and video name

        :param int index: The above-mentioned id of the data.
        """
        video_name = self.split[self.mode + '_keys'][index]
        frame_features = self.list_frame_features[index]
        gtscore = self.list_gtscores[index]
        user_summary = self.list_user_summary[index]
        sb = self.list_sb[index]
        n_frames = self.list_n_frames[index]
        positions = self.list_positions[index]
        if (self.mode == 'test') or (self.mode == 'val'):
            return frame_features, video_name, user_summary, sb, n_frames, positions
        else:
            return frame_features, gtscore, user_summary, sb, n_frames, positions


def get_loader(mode, video_type, split_index):
    """ Loads the `data.Dataset` of the `split_index` split for the `video_type` Dataset.
    Wrapped by a Dataloader, shuffled and `batch_size` = 1 in train `mode`.

    :param str mode: The mode of the model, train or test.
    :param str video_type: The Dataset being used, SumMe or TVSum.
    :param int split_index: The index of the Dataset split being used.
    :return: The Dataset used in each mode.
    """
    if mode.lower() == 'train':
        vd = VideoData(mode, video_type, split_index)
        return DataLoader(vd, batch_size=1, shuffle=True)
    else:
        return VideoData(mode, video_type, split_index)


if __name__ == '__main__':
    pass
