from toolbox.constants import T1, T1GD, T2, FLAIR
import torch
import os


landmarks_dict = {
        T1: torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "histogram_landmarks/t1w_landmarks.npy")),
        T2: torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "histogram_landmarks/t2w_landmarks.npy")),
        FLAIR: torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "histogram_landmarks/flair_landmarks.npy")),
        T1GD: torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "histogram_landmarks/t1wce_landmarks.npy")),
    }