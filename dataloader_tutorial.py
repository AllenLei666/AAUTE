import os
import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from preprocess.audio_process import *
from preprocess.image_process import *

np.random.seed(42)

class UAVLoader(Dataset):
    def __init__(self, annotation_lines,root_path,dark_aug=0,testing=0):  
        super(UAVLoader, self).__init__()
        self.annotation_lines   = annotation_lines
        self.audio_path         = os.path.join(root_path,'audio_npy')
        self.gt_path            = os.path.join(root_path,'gt')
        self.pseudo_label_path       = os.path.join(root_path,'pseudo_label')

    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, index):

        name       = self.annotation_lines[index]
        audio_name  = os.path.join(self.audio_path,name[:-4]+'npy')
        gt_name     = os.path.join(self.gt_path,name[:-4]+'npy')
        pseudo_label_name = os.path.join(self.pseudo_label_path,name[:-4]+'npy')
        


        audio   = make_seq_audio(self.audio_path,name[:-4]+'npy')
        audio   = np.transpose(audio,[1,0])
        spec       = Audio2Spectrogram(audio,sr=46080)
        spec       = spec.float()


        gt      = np.load(gt_name)
        gt     = torch.from_numpy(gt).float()

        pseudo_label =  np.load(pseudo_label_name)
        pseudo_label = torch.from_numpy(pseudo_label).float()

        return spec, gt, pseudo_label

