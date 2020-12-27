import torch
import csv
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt


def images_to_hist(images):
    hist_arr = torch.zeros((images.shape[0], 1, 100, 100), dtype=torch.float, )
    cv_images = images.permute(0, 2, 3, 1)
    cv_images = cv_images.numpy().astype('uint8')
    for i in range(cv_images.shape[0]):
        # plt.imshow(cv_images[i])
        # plt.show()
        hsv = cv2.cvtColor(cv_images[i], cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        hist = torch.from_numpy(hist)
        hist = torch.log(hist + 1) * 20
        # plt.imshow(hist)
        # plt.show()
        hist = hist.reshape((1, 180, 256))
        hist = transforms.functional.resize(hist, (100, 100), interpolation=2)
        hist_arr[i] = hist
    return hist_arr


def square_euclidean_metric(a, b):
    """ Measure the euclidean distance (optional)
    Args:
        a : torch.tensor, features of data query
        b : torch.tensor, mean features of data shots or embedding features

    Returns:
        A torch.tensor, the minus euclidean distance
        between a and b
    """

    n = a.shape[0]
    m = b.shape[0]

    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)

    logits = torch.pow(a - b, 2).sum(2)

    return logits


def count_acc(logits, label):
    """ In each query set, the index with the highest probability or lowest distance is determined
    Args:
        logits : torch.tensor, distance or probabilty
        label : ground truth

    Returns:
        float, mean of accuracy
    """

    # when logits is distance
    pred = torch.argmin(logits, dim=1)
    # when logits is prob
    #pred = torch.argmax(logits, dim=1)

    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


class Averager():
    """ During training, update the average of any values.
    Returns:
        float, the average value
    """

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


class csv_write():

    def __init__(self, args):
        self.f = open('StudentID_Name.csv', 'w', newline='')
        self.write_number = 1
        self.wr = csv.writer(self.f)
        self.wr.writerow(['id', 'prediction'])
        self.query_num = args.query

    def add(self, prediction):

        for i in range(self.query_num):
          self.wr.writerow([self.write_number, int(prediction[i].item())])
          self.write_number += 1

    def close(self):
        self.f.close()
