#! /usr/bin/env python3

import rospy
import rospkg
import tf2_ros
import message_filters as mf
from cv_bridge import CvBridge
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from sensor_msgs.msg import Image, PointCloud2

import os
import time
import cv2
import PIL.Image
import numpy as np

from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
from nanoowl.owl_predictor import OwlPredictor as NanoOwlPredictor


np.float = float  # NOTE: Temporary fix for ros_numpy issue; check #39
import ros_numpy


def load_param(name, value=None):
    if value is None:
        assert rospy.has_param(name), f'Missing parameter "{name}"'
    return rospy.get_param(name, value)

def replace_base(old, new) -> str:
    split_last = lambda xs: (xs[:-1], xs[-1])
    is_private = new.startswith('~')
    is_global = new.startswith('/')
    assert not (is_private or is_global)
    ns, _ = split_last(old.split('/'))
    ns += new.split('/')
    return '/'.join(ns)


class SidewalkSegementation:
    
    def __init__(self) -> None:
        try:
            # Initialize node
            rospy.init_node('sidewalk_segmentation')
            
            # Topic Parameters
            self.rgb_topic = load_param('~rgb_topic', 'image')
            self.pointcloud_topic = load_param('~pointcloud_topic', 'pointcloud')
            
            self.sidewalk_mask_topic = load_param('~sidewalk_mask_topic', 'sidewalk_mask')
            self.sidewalk_annotated_topic = load_param('~sidewalk_annotated_topic', 'sidewalk_annotated')
            self.sidewalk_pointcloud_topic = load_param('~sidewalk_pointcloud_topic', 'sidewalk_pointcloud')
            
            # SAM model parameters
            self.sam_model_name = load_param('~sam_model_name', 'FastSAM-x.pt') # FastSAM-s.pt or FastSAM-x.pt
            self.sam_conf = load_param('~sam_conf', 0.4)
            self.sam_iou = load_param('~sam_iou', 0.9)
            
            # OWL Model parameters
            self.owl_model_name = load_param('~owl_model_name', 'google/owlvit-base-patch32')
            self.owl_image_encoder_path = load_param('~owl_image_encoder_path', '/opt/nanoowl/data/owl_image_encoder_patch32.engine')
            self.owl_threshold = load_param('~owl_threshold', 0.1)
            
            # Prompt parameters
            self.prompt_type = load_param('~prompt_type', 'bbox') # bbox or points or text
            self.prompt_bbox = load_param('~bbox_prompt_corners', [0.30, 0.50, 0.70, 0.90]) # [x1, y1, x2, y2] in relative coordinates
            self.prompt_points = load_param('~points_prompt_points', [[0.50, 0.90]]) # [[x1, y1], [x2, y2], ...] in relative coordinates
            self.prompt_text = load_param('~text_prompt_text', 'a sidewalk or footpath or walkway or paved path for humans to walk on')
            
            # Other parameters
            self.use_cuda = load_param('~use_cuda', True)
            self.brightness_window = load_param('~brightness_window', 0.5)
            self.mean_brightness = load_param('~mean_brightness', 0.5)
            self.frame_id = load_param('~frame_id', '')
            self.verbose = load_param('~verbose', False)
            
            # Publisher parameters
            self.publish_mask = load_param('~publish_mask', False)
            self.publish_annotated = load_param('~publish_annnotated', True)
            self.publish_pointcloud = load_param('~publish_pointcloud', False)
            
            # Get package path
            rospack = rospkg.RosPack()
            package_path = rospack.get_path('svea_vision')
            
            # Load models
            self.device = 'cuda' if self.use_cuda else 'cpu'
            self.sam_model_path = os.path.join(package_path, 'models', self.sam_model_name)
            self.sam_model = FastSAM(self.sam_model_path)
            if self.use_cuda:
                self.sam_model.to('cuda')
                if self.prompt_type=='text':
                    self.owl_model = NanoOwlPredictor(self.owl_model, image_encoder_engine=self.owl_image_encoder)
                    self.prompt_text = [self.prompt_text]
                    self.prompt_text_encodings = self.owl_model.encode_text(self.prompt_text)
            elif self.prompt_type=='text':
                rospy.logerr('text prompt is only supported when use_cuda is set to True. Only bbox and points prompts are supported without CUDA. Exiting...')
            
            # CV Bridge
            self.cv_bridge = CvBridge()
            
            # TF2
            self.tf_buf = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buf)
            
            # Publishers
            if self.publish_mask:
                self.sidewalk_mask_pub = rospy.Publisher(self.sidewalk_mask_topic, Image, queue_size=1)
            elif self.publish_ann:
                self.sidewalk_annotated_pub = rospy.Publisher(self.sidewalk_annotated_pub, Image, queue_size=1)
            elif self.publish_pointcloud:
                self.sidewalk_pointcloud_pub = rospy.Publisher(self.sidewalk_pointcloud_topic, PointCloud2, queue_size=1)
            else:
                rospy.logerr('No output type enabled. Please set atleast one of publish_mask, publish_annnotated, or publish_pointcloud parameters to True. Exiting...')
            
            # Subscribers
            if self.publish_pointcloud:
                self.ts = mf.TimeSynchronizer([
                        mf.Subscriber(self.rgb_topic, Image),
                        mf.Subscriber(self.pointcloud_topic, PointCloud2),
                ], queue_size=1)
                self.ts.registerCallback(self.callback)
            else:
                self.rgb_sub = rospy.Subscriber(self.rgb_topic, Image, self.callback)
            
            # Logging dictionary
            self.log_times = {}
            
        except Exception as e:
            # Log error
            rospy.logerr(e)

        else:
            # Log status
            rospy.loginfo('{} node initialized with SAM model: {}, OWL model: {}, prompt type: {}, frame_id: {}, use_cuda: {}'.format(
                rospy.get_name(), self.sam_model_name, self.owl_model_name, self.prompt_type, self.frame_id, self.use_cuda))
            
    def run(self) -> None:
        try:
            rospy.spin()
        except rospy.ROSInterruptException:
            rospy.loginfo('Shutting down {}'.format(rospy.get_name()))
            
    def adjust_mean_brightness(self, image, mean_brightness) -> np.ndarray:
        # Convert image to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Calculate mean brightness of the window
        v_len = len(hsv[:,:,2].flatten())
        v_max_window = np.sort(hsv[:,:,2].flatten())[-int(v_len*self.brightness_window):]
        mean_brightness_img = np.mean(v_max_window/255.0)
                
        # Adjust brightness
        hsv[:,:,2] = np.clip(hsv[:,:,2] * (mean_brightness/mean_brightness_img), 0, 255)

        # Convert back to RGB
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    def owl_predict(self, image, text, text_encodings, threshold) -> list:
        # Predict using OWL model
        owl_output = self.owl_model.predict(
            image=image,
            text=text,
            text_encodings=text_encodings,
            pad_square=True,
            threshold=[threshold]
        )
        
        # Select the bounding box with the highest score
        n_detections = len(owl_output.boxes)
        if n_detections > 0:
            max_score_index = owl_output.scores.argmax()
            bbox = [int(x) for x in owl_output.boxes[max_score_index]]
            if self.verbose:
                rospy.loginfo('OWL model detections: {}'.format(n_detections))
        else:
            bbox = []
        
        return bbox
    
    def segment_image(self, img_msg) -> np.ndarray:
        # Convert ROS image to OpenCV image
        self.image = self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding='rgb8')
        
        # Adjust mean brightness
        self.image = self.adjust_mean_brightness(self.image, self.mean_brightness)
        
        # Run inference on the image
        everything_results = self.sam_model(self.image, device=self.device, imgsz=img_msg.width,
                                        sam_conf=self.sam_conf, sam_iou=self.sam_iou, retina_masks=True, verbose=self.verbose)
        self.log_times['inference_time'] = time.time()
        
        # Prepare a Prompt Process object
        prompt_process = FastSAMPrompt(self.image, everything_results, device=self.device)
        
        # Prompt the results
        if self.prompt_type == 'text':
            # Use OWL model to get bbox
            self.bbox = self.owl_predict(PIL.Image.fromarray(self.image), self.prompt_text, self.prompt_text_encodings, self.owl_threshold)
            if len(self.bbox) == 0:
                if self.verbose:
                    rospy.loginfo('OWL model has no detections. Using default bbox prompt.')
                self.bbox = [int(scale*dim) for scale, dim in zip(self.prompt_bbox, 2*[img_msg.width, img_msg.height])]
            sidewalk_results = prompt_process.box_prompt(self.bbox)            
        elif self.prompt_type == 'bbox':
            # Convert bbox from relative to absolute
            self.bbox = [int(scale*dim) for scale, dim in zip(self.prompt_bbox, 2*[img_msg.width, img_msg.height])]
            sidewalk_results = prompt_process.box_prompt(self.bbox)
        elif self.prompt_type == 'points':
            # Convert points from relative to absolute
            points=[[int(scale*dim) for scale, dim in zip(point, [img_msg.width, img_msg.height])] for point in self.prompt_points]
            sidewalk_results = prompt_process.point_prompt(points, pointlabel=[1])
        else:
            rospy.logerr("Invalid value for prompt_type parameter")
            
        self.log_times['prompt_time'] = time.time()
        
        # Apply morphological opening to remove small noise
        sidewalk_mask = sidewalk_results[0].cpu().numpy().masks.data[0].astype('uint8')*255
        erosion_kernel = np.ones((5,5), np.uint8)
        dilation_kernel = np.ones((3,3), np.uint8)
        sidewalk_mask = cv2.erode(sidewalk_mask, erosion_kernel, iterations=1)
        sidewalk_mask = cv2.dilate(sidewalk_mask, dilation_kernel, iterations=1)
        
        # Select the largest contour from the mask
        contours, _ = cv2.findContours(sidewalk_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            max_contour = max(contours, key=cv2.contourArea)
            sidewalk_mask = np.zeros_like(sidewalk_mask)
            cv2.fillPoly(sidewalk_mask, [max_contour], 255)
        
        self.log_times['postprocess_time'] = time.time()
                    
        return sidewalk_mask
            
    def extract_pointcloud(self, pc_msg, mask) -> PointCloud2:
        # Convert ROS pointcloud to Numpy array
        pc_data = ros_numpy.point_cloud2.pointcloud2_to_array(pc_msg)
        
        # Convert mask to boolean and flatten
        mask = np.array(mask, dtype='bool')
        
        # Extract pointcloud
        extracted_pc = np.full_like(pc_data, np.nan)
        extracted_pc[mask] = pc_data[mask]
        
        # Convert back to ROS pointcloud
        extracted_pc_msg = ros_numpy.point_cloud2.array_to_pointcloud2(extracted_pc, pc_msg.header.stamp, pc_msg.header.frame_id) 
        
        return extracted_pc_msg
        
    def callback(self, img_msg, pc_msg = None) -> None:
        self.log_times['start_time'] = time.time()
        
        # Segment image
        sidewalk_mask = self.segment_image(img_msg)
        
        # Extract pointcloud
        if self.publish_pointcloud:
            extracted_pc_msg = self.extract_pointcloud(pc_msg, sidewalk_mask)
            
            # Transform pointcloud to frame_id if specified
            if self.frame_id == '' or self.frame_id == extracted_pc_msg.header.frame_id:
                sidewalk_pc_msg = extracted_pc_msg
            else:        
                try:
                    transform_stamped = self.tf_buf.lookup_transform(self.frame_id, extracted_pc_msg.header.frame_id, extracted_pc_msg.header.stamp)
                except tf2_ros.LookupException or tf2_ros.ConnectivityException or tf2_ros.ExtrapolationException:
                    rospy.logwarn("{}: Transform lookup from {} to {} failed for the requested time. Using latest transform instead.".format(
                        rospy.get_name(), extracted_pc_msg.header.frame_id, self.frame_id))
                    transform_stamped = self.tf_buf.lookup_transform(self.frame_id, extracted_pc_msg.header.frame_id, rospy.Time(0))
                sidewalk_pc_msg = do_transform_cloud(extracted_pc_msg, transform_stamped)
        self.log_times['extract_pc_time'] = time.time()
        
        # Publish mask
        if self.publish_mask:
            mask_msg = self.cv_bridge.cv2_to_imgmsg(sidewalk_mask, encoding='mono8')
            self.sidewalk_mask_pub.publish(mask_msg)
        
        # Publish annotated image
        if self.publish_annotated:
            # Create annotated image
            color = np.array([0,0,255], dtype='uint8')
            masked_image = np.where(sidewalk_mask[...,None], color, self.image)
            sidewalk_annotated = cv2.addWeighted(self.image, 0.75, masked_image, 0.25, 0)
            
            if self.prompt_type=='bbox' or self.prompt_type=='text':
                cv2.rectangle(sidewalk_annotated, (self.bbox[0], self.bbox[1]), (self.bbox[2], self.bbox[3]), (0,255,0), 2)        
            annotated_msg = self.cv_bridge.cv2_to_imgmsg(sidewalk_annotated, encoding='rgb8')
            self.sidewalk_annotated_pub.publish(annotated_msg)
            
        # Publish pointcloud
        if self.publish_pointcloud:
            self.sidewalk_pointcloud_pub.publish(sidewalk_pc_msg)
        
        self.log_times['publish_time'] = time.time()
        
        # Log times
        if self.verbose:
            rospy.loginfo('{:.3f}s total, {:.3f}s inference, {:.3f}s prompt, {:.3f}s postprocess, {:.3f}s extract_pc, {:.3f}s publish'.format(
                self.log_times['publish_time'] - self.log_times['start_time'],
                self.log_times['inference_time'] - self.log_times['start_time'],
                self.log_times['prompt_time'] - self.log_times['inference_time'],
                self.log_times['postprocess_time'] - self.log_times['prompt_time'],
                self.log_times['extract_pc_time'] - self.log_times['postprocess_time'],
                self.log_times['publish_time'] - self.log_times['extract_pc_time']
            ))
    
    
if __name__ == '__main__':
    node = SidewalkSegementation()
    node.run()