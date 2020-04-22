import numpy as np
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
            self.threshold = np.quantile(mses, 0.9) * 1.05 # wiggle room
            print(f'choosing threshold of {self.threshold}')

    def __call__(self, patch):
        hue = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).mean((0,1))[0]
        dist = np.mean((hue - self.patches)**2)
        print(f'distance of patch: {dist}')

        return dist < self.threshold
