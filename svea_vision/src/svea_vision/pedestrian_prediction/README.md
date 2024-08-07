# pedestrian_prediction

## Background

The pedestrian_prediction folder contains code for Transformer-based Behavior Clustering and Data-Driven Reachability Analysis (code adapted from [here](https://github.com/kfragkedaki/Pedestrian_Project))

## Usage

The node that genreates zonotopes is located in the scripts folder and is called traj_to_zonotope.py. This node subscribes to the /state and /pedestrian_flow_estimate topics and publishes the zonotope topic.
