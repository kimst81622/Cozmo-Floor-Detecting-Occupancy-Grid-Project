import numpy as np

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
        patches: NxHxWxC numpy array of RGB image patches
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
