import torch.nn as nn
import torch
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from src.utils import square_euclidean_metric


""" Optional conv block """
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


""" Define your own model """


def conv_block_new(in_channels, out_channels, kern_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kern_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )

def conv_block_new_new(in_channels, out_channels, kern_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kern_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.ReLU(),
    )


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ImageAndHistEncoder(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.layer1_in1 = conv_block_new_new(x_dim, int(hid_dim/2), 3, 1, 1)
        self.layer2_in1 = conv_block_new_new(int(hid_dim/2), int(hid_dim/2), 3, 1, 1)

        self.layer1_in2 = conv_block_new(1, int(hid_dim/2), 3, 1, 1)
        self.layer2_in2 = conv_block_new(int(hid_dim/2), int(hid_dim/2), 3, 1, 1)

        self.layer3 = conv_block_new_new(hid_dim, hid_dim, 3, 1, 1)
        self.layer4 = conv_block_new_new(hid_dim, hid_dim, 3, 1, 1)
        self.layer5 = conv_block_new(hid_dim, hid_dim, 3, 1, 1)
        self.layer6 = conv_block_new_new(hid_dim, z_dim, 3, 1, 1)

    def forward(self, data_shot, data_query):
        x = self.layer1_in1(data_shot)
        x = self.layer2_in1(x)

        y = self.layer1_in2(data_query)
        y = self.layer2_in2(y)

        z = torch.cat([x, y], dim=1)
        z = self.layer3(z)
        z = self.layer4(z)
        z = self.layer5(z)
        z = self.layer6(z)

        return z


class FewShotModel(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = ImageAndHistEncoder(x_dim=x_dim, hid_dim=hid_dim, z_dim=z_dim)

    def forward(self, data_shot, data_query, data_shot_hist, data_query_hist):  # 그냥 20개 classification  문제로 바꾸기

        shot_vec = self.encoder(data_shot, data_shot_hist)
        query_vec = self.encoder(data_query, data_query_hist)

        shot_vec = shot_vec.reshape((25, -1))
        shot_vec_mean = shot_vec.reshape((5, 5, -1)).mean(1)

        query_vec = query_vec.reshape((20, -1))
        embedding_vector_unavg = square_euclidean_metric(query_vec, shot_vec)
        embedding_vector = square_euclidean_metric(query_vec, shot_vec_mean)
        return embedding_vector, embedding_vector_unavg


class FirstFewShotModel(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block_new(x_dim, hid_dim, 3, 3, 1),
            conv_block_new(hid_dim, hid_dim, 3, 3, 1),
            conv_block_new(hid_dim, hid_dim, 3, 3, 1),
            conv_block_new(hid_dim, z_dim, 3, 3, 1),
            Flatten()
        )

    def forward(self, data_shot, data_query):
        shot_vec = self.encoder(data_shot)
        query_vec = self.encoder(data_query)
        shot_vec = shot_vec.reshape((5, 5, -1))
        shot_vec = shot_vec.mean(1)
        embedding_vector = square_euclidean_metric(query_vec, shot_vec)
        return embedding_vector


class SecondFewShotModel(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block_new(x_dim, int(hid_dim/2), kern_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_block_new(int(hid_dim/2), int(hid_dim/2), 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_block_new(int(hid_dim/2), hid_dim, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_block_new(hid_dim, z_dim, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Flatten()
        )

    def forward(self, data_shot, data_query):
        shot_vec = self.encoder(data_shot)
        query_vec = self.encoder(data_query)
        shot_vec = shot_vec.reshape((5, 5, -1))
        shot_vec = shot_vec.mean(1)
        embedding_vector = square_euclidean_metric(query_vec, shot_vec)
        return embedding_vector


class FewShotModelasdasdasdasd(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = ImageAndHistEncoder(x_dim=x_dim, hid_dim=hid_dim, z_dim=z_dim)

    def forward(self, data_shot, data_query, data_shot_hist, data_query_hist):  # 그냥 20개 classification  문제로 바꾸기

        shot_vec = self.encoder(data_shot, data_shot_hist)
        query_vec = self.encoder(data_query, data_query_hist)

        shot_vec_mean = shot_vec.reshape((5, 5, -1)).mean(1)
        query_vec = query_vec.reshape((20, -1))
        embedding_vector = square_euclidean_metric(query_vec, shot_vec_mean)

        # shot_vec_mean = shot_vec.reshape((5, 5, -1)).mean(1)
        # shot_vec_concat = torch.cat(shot_vec, shot_vec_mean)  ###############################
        # embedding_vector = square_euclidean_metric(query_vec, shot_vec_concat)

        return embedding_vector
