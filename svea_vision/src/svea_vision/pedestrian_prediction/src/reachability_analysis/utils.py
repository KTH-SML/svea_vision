import numpy as np
import pickle
import os


ROOT = os.path.dirname(os.path.abspath(__file__))
while ROOT.rsplit("/", 1)[-1] != "pedestrian_prediction":
    ROOT = os.path.dirname(ROOT)
    
def load_data(filename: str = "sind.pkl", filepath: str = ROOT + "/resources/") -> np.ndarray:
    """Load previously pickled data

    Parameters:
    -----------
    file : str (default = 'sind.pkl')
        File-name of the pickled file.
    """
    _f = open(filepath + filename, "rb")
    return pickle.load(_f)
