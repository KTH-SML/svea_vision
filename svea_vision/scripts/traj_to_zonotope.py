#! /usr/bin/env python3

from svea_vision.pedestrian_prediction.src.reachability_analysis.simulation import get_test_config, get_test_label, get_cluster, get_initial_conditions, reachability_for_all_modes
from svea_vision.pedestrian_prediction.src.reachability_analysis.labeling_oracle import LabelingOracleSVEAData
from svea_vision.pedestrian_prediction.src.datasets.data import SVEAData
from svea_vision.pedestrian_prediction.src.transformer_model.model import create_model, evaluate
import json
import logging
import os
import rospy
from torch.utils.data import DataLoader
from svea_vision.pedestrian_prediction.src.datasets.data import data_factory, Normalizer
from svea_vision.pedestrian_prediction.src.datasets.masked_datasets import collate_unsuperv
from svea_vision.pedestrian_prediction.src.utils.load_data import load_task_datasets
from svea_vision.pedestrian_prediction.src.clustering.NearestNeighbor import AnnoyModel
import numpy as np
from svea_vision_msgs.msg import PersonState, PersonStateArray, Zonotope, ZonotopeArray
from svea_msgs.msg import VehicleState

logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class TrajToZonotope:

    def __init__(self):
        
        CWD = os.path.dirname(os.path.abspath(__file__))
        while CWD.rsplit("/", 1)[-1] != "svea_vision":
            CWD = os.path.dirname(CWD)
        CWD = f'{CWD}/src/svea_vision/pedestrian_prediction'        
        self.config_path = f'{CWD}/resources/configuration.json'

        with open(self.config_path) as cnfg:
            self.config = json.load(cnfg)

        ROOT_RESOURCES = CWD + "/resources"

        self.config['original_data'] = False
        self.config['online_data'] = True
        self.config['pattern'] = None
        self.config['data_class'] = 'svea'
        self.config['eval_only'] = True
        self.config['val_ratio'] = 1
        self.config['output_dir'] = ROOT_RESOURCES + "/eval"
        self.config['save_dir'] = ROOT_RESOURCES + "/eval"
        self.config['tensorboard_dir'] = ROOT_RESOURCES + "/eval"

        self.data_oracle = SVEAData(self.config)
        self.nn_model = AnnoyModel(config=self.config)

        rospy.init_node("traj_to_zonotope", anonymous=True)
        self.publisher = rospy.Publisher("~zonotopes", ZonotopeArray, queue_size=10)
        self.subscriber = rospy.Subscriber("/pedestrian_flow_estimate/pedestrian_flow_estimate", PersonStateArray, self._callback, queue_size=10)
        self.svea_pos_subscriber = rospy.Subscriber("/state", VehicleState, self.update_offset, queue_size=10)

        self.start()


    @staticmethod
    def zonotope_to_rectangle(z):
        """
        Convert a zonotope to its bounding rectangle.
        
        Parameters:
        z (zonotope): The zonotope to convert.

        Returns:
        (tuple): Top left and bottom right coordinates of the rectangle.
        """
        x = z.x[:, 0]
        G = z.G
        
        # Sum of the absolute values of the generators
        extent = np.sum(np.abs(G), axis=1)

        # Top left and bottom right coordinates
        bottom_left = x - extent
        top_right = x + extent
        
        return (bottom_left[0], bottom_left[1], top_right[0], top_right[1])


    def start(self):
        """Subscribes to the topic containing only detected
        persons and applies the function __callback."""

        while not rospy.is_shutdown():
            rospy.spin()


    def update_offset(self, msg):
        """This method is a callback function that is triggered when a state message is received.
        It gives the data oracle knowledge of the robot's current position so it can recenter person 
        coordinates to be consistent with the model

        :param msg: message containing the current position of the SVEA
        :return: None"""
        self.data_oracle.x_offset = msg.x
        self.data_oracle.y_offset = msg.y


    def _callback(self, msg):
        """This method is a callback function that is triggered when a message is received.
        It publishes the estimated zonotopes to a '/traj_to_zonotopes/zonotopes' topic.

        :param msg: message containing the detected persons
        :return: None"""
        self.data_oracle.process_message(msg)
        val_data = self.data_oracle.feature_df

        # Pre-process features
        if self.config["data_normalization"] != "none":
            logger.info("Normalizing data ...")
            normalizer = Normalizer(self.config["data_normalization"])
            val_data = normalizer.normalize(val_data)

        task_dataset, collate_fn = load_task_datasets(self.config)
        val_dataset = task_dataset(self.data_oracle.feature_df, self.data_oracle.all_IDs)

        # Dataloaders
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config["num_workers"],
            pin_memory=True,
            collate_fn=lambda x: collate_fn(x, max_len=self.data_oracle.max_seq_len),
        )


        model, optimizer, trainer, val_evaluator, start_epoch = create_model(
            self.config, None, val_loader, self.data_oracle, logger, device='cpu'
        )

        aggr_metrics, embedding_data = evaluate(val_evaluator, config=self.config, save_embeddings=True)


        zonotopeArray_msg = ZonotopeArray()
        zonotopeArray_msg.header = msg.header
        
        labeling_oracle = LabelingOracleSVEAData(self.config)
        #TODO are embeddings in the same order as person_states?
        for embedding, target, person_state in zip(embedding_data['embeddings'][0], embedding_data['targets'][0], msg.personstate):
            c = self.nn_model.get(embedding)
            test_cases = {f'c_{c}': f'Cluster: {c}'}
            pos = np.array([person_state.pose.position.x, person_state.pose.position.y])
            v = np.array([person_state.vx, person_state.vy])
            z, l, _b, _z = reachability_for_all_modes(pos=pos, vel=v, baseline=False, test_cases=test_cases, config=self.config, trajectory=target, show_plot=False, save_plot=None, _sind = labeling_oracle)
            rect = TrajToZonotope.zonotope_to_rectangle(z[0])

            zonotope = Zonotope()
            zonotope.id = person_state.id
            zonotope.counter = person_state.counter
            zonotope.xmin = rect[0]
            zonotope.ymin = rect[1]
            zonotope.xmax = rect[2]
            zonotope.ymax = rect[3]
            zonotopeArray_msg.zonotopes.append(zonotope)

        self.publisher.publish(zonotopeArray_msg)


if __name__ == "__main__":
    traj_to_zonotope = TrajToZonotope()
