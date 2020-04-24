import numpy as np
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb
import matplotlib.pyplot as plt
import cv2

with open('test.npy', 'rb') as filedir:
    test = np.load(filedir)

METHOD = 'uniform'
radius = 2
n_points = 8 * radius


def kullback_leibler_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))


def match(refs, img):
    best_score = 10
    best_name = None
    lbp = local_binary_pattern(img, n_points, radius, METHOD)
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
    for name, ref in refs.items():
        ref_hist, _ = np.histogram(ref, density=True, bins=n_bins,
                                   range=(0, n_bins))
        score = kullback_leibler_divergence(hist, ref_hist)
        if score < best_score:
            best_score = score
            best_name = name
    return best_name

refs = {}

for i in range(4*test.shape[0]//5):
    img = cv2.cvtColor(test[i], cv2.COLOR_BGR2GRAY)
    refs[i] =  local_binary_pattern(img, n_points, radius, METHOD)

# classify rotated textures
print('Rotated images matched against references using LBP:')

# plot histograms of LBP of textures

for i in range(4*test.shape[0]//5,test.shape[0]):
    print(match(refs, cv2.cvtColor(test[i], cv2.COLOR_BGR2GRAY)))
