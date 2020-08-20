import numpy as np


class DataAugment(object):
    """
    DataAugment interface.

    A data transform needs to conduct the following steps:

    1. Set :attr:`sample_params` at initialization to compute required sample size.
    2. Randomly generate augmentation parameters for the current transform.
    3. Apply the transform to a pair of images and corresponding labels.

    All the real data augmentations should be a subclass of this class.
    """
    def __init__(self, p=0.5):
        assert 0.0 <= p <= 1.0
        self.p = p
        self.sample_params = {
            'ratio': np.array([1.0, 1.0, 1.0]),
            'add': np.array([0, 0, 0])}

    def set_params(self):
        """
        Calculate the appropriate sample size with data augmentation.
        
        Some data augmentations (wrap, misalignment, etc.) require a larger sample 
        size than the original, depending on the augmentation parameters that are 
        randomly chosen. This function takes the data augmentation 
        parameters and returns an updated data sampling size accordingly.
        """
        raise NotImplementedError

    def __call__(self, data, random_state=None):
        """
        Apply the data augmentation.

        For a multi-CPU dataloader, one may need to use a unique index to generate 
        the random seed (:attr:`random_state`), otherwise different workers may generate
        the same pseudo-random number for augmentation and sampling.
        """
        raise NotImplementedError
