import numpy as np
from skimage.feature import local_binary_pattern
import cv2

class PatchClassifier():
    """
    Base class for patch classifier, just laying out the abstract structure.
    """
    def __init__(self):
        pass # might later add things here, if there are commonalities

    def __call__(self, patch):
        raise NotImplementedError # true for floor, false otherwise

class PixelClassifier(PatchClassifier):
    """
    Patch classification with simple MSE on pixels.
    """
    def __init__(self, patches, threshold=None):
        """
        patches: NxHxWxC numpy array of BGR image patches
        threshold: threshold for SSE, optional, determined automatically if
                not specified
        """
        assert isinstance(patches, np.ndarray) and len(patches.shape) == 4

        self.patches = patches
        
        if threshold is not None:
            self.threshold = threshold
        else:
            # set threshold to try to include all patches
            mses = [np.mean((patch - patches)**2) for patch in patches]
            self.threshold = np.max(mses) * 1.05 # wiggle room

    def __call__(self, patch):
        assert patch.shape == self.patches[0].shape

        return np.mean((patch - self.patches)**2) < self.threshold

class HueClassifier(PatchClassifier):
    """
    Patch classification with simple squared distance of average hue.
    """
    def __init__(self, patches, threshold=None):
        """
        patches: NxHxWxC numpy array of BGR image patches
        threshold: optional threshold for distance, determined automatically if
                not specified
        """
        assert isinstance(patches, np.ndarray) and len(patches.shape) == 4

        self.patches = np.stack([cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            for patch in patches])
        # only average hue for each patch needed
        print(self.patches.shape)
        self.patches = self.patches.mean((1,2))[:, 0]
        
        if threshold is not None:
            self.threshold = threshold
        else:
            # set threshold to try to include all patches
            mses = [np.mean((patch - self.patches)**2) 
                    for patch in self.patches]
            self.threshold = np.quantile(mses, 0.95) * 1.05 # wiggle room

    def __call__(self, patch):
        hue = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).mean((0,1))[0]
        dist = np.mean((hue - self.patches)**2)

        return dist < self.threshold

class LBPClassifier(PatchClassifier):
    """
    Patch classification with KL divergence on histograms of LBP features.
    """
    def __init__(self, patches, threshold=None, radius=2, n_points=16,
            method='uniform'):
        assert isinstance(patches, np.ndarray) and len(patches.shape) == 4

        self.radius = radius
        self.n_points = n_points
        self.method = method

        lbps = [local_binary_pattern(cv2.cvtColor(patch,
            cv2.COLOR_BGR2GRAY), n_points, radius, method) 
            for patch in patches]

        self.n_bins = int(np.max([lbp.max() for lbp in lbps]) + 1)
        self.hists = [np.histogram(lbp, density=True, bins=self.n_bins, 
            range=(0,self.n_bins))[0] for lbp in lbps]

        pairwise_divs = [[self.kl_div(hist_one, hist_two) for hist_two in
            self.hists if hist_one is not hist_two] for hist_one in self.hists]

        mean_divs = np.array(pairwise_divs).mean(1)
        self.threshold = np.quantile(mean_divs, 0.95) * 1.05 # wiggle room

    def kl_div(self, p, q):
        """
        Kullback-Leibler divergence for two histograms, from the skimage
        LBP demo.
        """
        p = np.asarray(p)
        q = np.asarray(q)
        filt = np.logical_and(p != 0, q != 0)
        return np.sum(p[filt] + np.log2(p[filt] / q[filt]))

    def __call__(self, patch):
        lbp = local_binary_pattern(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY),
                self.n_points, self.radius, self.method)
        histogram = np.histogram(lbp, density=True, bins=self.n_bins,
                range=(0, self.n_bins))[0]

        kl_divs = [self.kl_div(histogram, hist) for hist in self.hists]
        return np.mean(kl_divs) < self.threshold

class LBPHueClassifier(PatchClassifier):
    def __init__(self, patches):
        self.hue = HueClassifier(patches)
        self.lbp = LBPClassifier(patches)

    def __call__(self, patch):
        hue = self.hue(patch)
        lbp = self.lbp(patch)
        print(f'hue decision is {hue} and LBP is {lbp}')

        return hue and lbp
