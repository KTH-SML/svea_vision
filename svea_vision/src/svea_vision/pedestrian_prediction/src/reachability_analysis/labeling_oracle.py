from src.datasets.data import SVEAData
import numpy as np
import os 

LABELS = {
    "cross_left": 0,
    "cross_right": 1,
    "cross_straight": 2,
    "cross_illegal": 3,
    "crossing_now": 4,
    "not_cross": 5,
    "unknown": 6,
}

CWD = os.getcwd()
while CWD.rsplit("/", 1)[-1] != "Pedestrian_Project":
    CWD = os.path.dirname(CWD)

ROOT = CWD + "/resources"


def angle_between_angles(a1: float, a2: float):
    """Calculate interior angle between two angles

    Parameters:
    -----------
    a1 : float
        The first heading (angle)
    a2 : float
        The second heading (angle)
    """
    v = np.array([np.cos(a1), np.sin(a1)])
    w = np.array([np.cos(a2), np.sin(a2)])
    return np.math.atan2(np.linalg.det([v, w]), np.dot(v, w))



class LabelingOracleSVEAData(SVEAData):

    def __init__(self, config: dict, n_proc=None):
        # Initialize the base class with all its setup
        super().__init__(config, n_proc)
        self.config = config


    def filter_paddings(self, dataset: np.ndarray, padded_batches: np.ndarray):
        # Find batches with no padding
        unpadded_batches = np.all(padded_batches, axis=1)  # True only for batches with all 1s (no padding)

        # Filter the dataset to keep only completely unpadded batches
        filtered_data = dataset[unpadded_batches]

        return filtered_data