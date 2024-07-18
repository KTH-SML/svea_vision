import os
from multiprocessing import Pool, cpu_count
import re
import logging

import numpy as np
import pandas as pd

from bagpy import bagreader

logger = logging.getLogger("__main__")


class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(self, norm_type, mean=None, std=None, min_val=None, max_val=None):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (
                self.max_val - self.min_val + np.finfo(float).eps
            )

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform("mean")) / grouped.transform("std")

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform("min")
            return (df - min_vals) / (
                grouped.transform("max") - min_vals + np.finfo(float).eps
            )

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))

    def inverse_normalize(self, df):
        if self.norm_type == "standardization":
            return df * self.std + self.mean
        elif self.norm_type == "minmax":
            return df * (self.max_val - self.min_val) + self.min_val
        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return df * grouped.transform("std") + grouped.transform("mean")
        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform("min")
            max_vals = grouped.transform("max")
            return df * (max_vals - min_vals) + min_vals
        else:
            raise NameError(f'Inverse normalize method "{self.norm_type}" not implemented')


class BaseData(object):

    def set_num_processes(self, n_proc):

        if (n_proc is None) or (n_proc <= 0):
            self.n_proc = cpu_count()  # max(1, cpu_count() - 1)
        else:
            self.n_proc = min(n_proc, cpu_count())


    @staticmethod
    def sort_clean_data(df):
        """"""
        keep_cols = ["track_id", "timestamp_ms", "x", "y", "vx", "vy", "ax", "ay"]

        # sort based on time and id
        df_sorted = df.sort_values(by=["track_id", "timestamp_ms"])

        # make track id unique among different files
        df_sorted["track_id"] = (
            df_sorted["file_id"].astype(str) + "_" + df_sorted["track_id"].astype(str)
        )

        # keep columns
        df_final = df_sorted[keep_cols]

        # remove_stationary_trajectories
        df_final = df_final[
            df_final.groupby("track_id")[["vx", "vy"]].transform(any).all(axis=1)
        ]

        # # remove incorrent datarows
        # df_final = df_final[(df_final["vx"] >= 0) & (df_final["vy"] >= 0)]

        return df_final


    def _gather_data_paths(self, root_dir, pattern):
        # Implementation to gather data paths  based on a given pattern
        data_paths = []  # list of all paths
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                data_paths.append(os.path.join(root, file))

        if len(data_paths) == 0:
            raise Exception(
                "No files found using: {}".format(os.path.join(root_dir, "*"))
            )

        if pattern is None:
            # by default evaluate on
            selected_paths = data_paths
        else:
            selected_paths = list(filter(lambda x: re.search(pattern, x), data_paths))

        input_paths = [
            p for p in selected_paths if os.path.isfile(p) and p.endswith(".csv")
        ]
        if len(input_paths) == 0:
            raise Exception("No .csv files found using pattern: '{}'".format(pattern))

        return input_paths

    @staticmethod
    def assign_chunk_idx(df, chunk_len):
        """Assigns a chunk index to each row and trajectory."""
        # Calculate local chunk indices within each unique trajectory
        df["chunk_idx"] = df.groupby("track_id").cumcount() // chunk_len

        # Generate a global chunk ID by enumerating each unique combination of unique_int_id and chunk_idx
        df["data_chunk_len"] = df.groupby(
            ["track_id", "chunk_idx"]
        ).ngroup()  # ngroup assigns unique numbers to each group

        return df

    @staticmethod
    def remove_small_chunks(df, min_size=2):
        """
        Removes chunks from the dataframe that have fewer than min_size points.

        Parameters:
        - df: The dataframe to process.
        - min_size: The minimum number of points a chunk must have to be retained.

        Returns:
        - The filtered dataframe.
        """
        # Group by global_chunk_id and filter
        filtered_df = df.groupby("data_chunk_len").filter(lambda x: len(x) >= min_size)
        return filtered_df
    
    
    def reassign_chunk_indices(self, df):
        # Create a unique list of the old chunk indices
        unique_chunks = df['data_chunk_len'].unique()
        # Create a mapping from old to new indices
        chunk_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_chunks)}
        # Map the old indices to new indices
        df['data_chunk_len'] = df['data_chunk_len'].map(chunk_mapping)
        return df


class SVEAData(BaseData):
    """
    Dataset class for online SVEA navigation. With each message that comes in, the process_message method updates the DataFrames.
    Attributes:
        all_df: dataframe indexed by ID, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        id_counts: maps trajectory IDs to the number of corresponding columns in all_df
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, config: dict, n_proc=None):

        n_proc = config["n_proc"] if n_proc is None else n_proc
        self.set_num_processes(n_proc=n_proc)
        self.config = config

        self.data_chunk_len = config['data_chunk_len']
        self.max_seq_len = config['data_chunk_len']

        # Load and preprocess data
        self.all_df = pd.DataFrame(columns=["track_id", "frame_id", "timestamp_ms", "x", "y", "vx", "vy", "ax", "ay"])
        self.all_df = self.all_df.set_index("track_id")
        self.feature_names = ["x", "y", "vx", "vy", "ax", "ay"]
        self.all_IDs = []
        self.id_counts = {}


    def process_message(self, msg):
        """
        Process a message and update all_df. Ensure that each trajectory has a max of self.max_seq_len points stored
        """

        frame_id = msg.header.seq
        timestamp_ms = msg.header.stamp.secs * 1000 + msg.header.stamp.nsecs / 1e6 

        for person in msg.personstate:
            track_id = person.id
            x = person.pose.position.x
            y = person.pose.position.y
            vx = person.vx
            vy = person.vy 
            ax = person.ax
            ay = person.ay

            if track_id not in self.all_IDs:
                self.all_IDs.append(track_id)
                self.id_counts[track_id] = 0

            if self.id_counts[track_id] == self.max_seq_len: 
                just_this_id = self.all_df[self.all_df['track_id'] == track_id]
                self.all_df = self.all_df[~((self.all_df['track_id'] == track_id) & self.all_df[self.all_df['frame_id'] == min(just_this_id['frame_id'])])]
            else:
                self.id_counts[track_id] += 1

            row = [frame_id, timestamp_ms, x, y, vx, vy, ax, ay]
            self.all_df.loc[len(self.all_df)] = row
            self.all_df.index.values[-1] = track_id

            # Remove chunks with less than 2 points
            # self.all_df = self.all_df.groupby("track_id").filter(lambda x: len(x) >= 2)

            self.feature_df = self.all_df[self.feature_names]


data_factory = {'svea' : SVEAData}
