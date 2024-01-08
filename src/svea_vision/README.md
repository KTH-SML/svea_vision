# Svea Vision

The svea_vision package uses a pair of zed-cameras to estimate traffic participants and obstacles.

## Requirements
##### Python dependencies
- scipy
- filterpy
- shapely

##### Packages for the ZED object detection:
- aliencontrol
- ros_abconnect
- rsu_core
- rsu_msgs
- vision_opencv
- zed-ros-wrapper
- realsense-ros
- image_common
- image_transport_plugins

## Usage and Installation
Instructions
### 1. Basic use of Jetson and Zed Camera
- Plug in Nvidia Jetson and Zed camera accordingly and boot up the Jetson.

On your own machine:
- Connect to the router RUT360

- ensure you are on RSU branch (git checkout rsu)
- Open two cmd terminals 
- 1st terminal: ping "ubuntu" (need to fix network if not functioning)
- 1st terminal: ssh nvidia@ubuntu
- 2nd terminal: change directory (cd) into root folder of the workspace (infra_ws).
- 2nd terminal: util/remote_ros.sh ubuntu
- 1st terminal: cd into root of workspace on ubuntu 
- 1st terminal: sudo util/run
- 1st terminal: roslaunch infra_ws "pkg.launch" (roslaunch zed_main.launch or roslaunch camera2.launch)
- 2nd terminal: open rviz using "rviz" (can now visualize published messages in rviz)


### 2. Calibrating the Camera and Aruco Markers (prior to demo)

	Note: This needs to be performed for any new aruco marker that one desires
		to add to the testing environment

- Place/fix the aruco marker in a location that can be well seen by a camera. Ideally within 10 meters of the camera.
- Setup the camera somewhere within the map bounds, within clear view of the desired unconfigured aruco marker.

Determine the camera pose using rviz
- launch the package as done in 1. (roslaunch zed_main.launch) and launch rviz
- Using the map of the environment in rviz determine as well as possible the position (x,y,z) and orientation (roll pitch yaw) of the camera in the map.
- NOTE: The coordinates for the camera that are returned by 'calibrating' are still approximate as the aruco needs to be calibrated first. Use the measuring tool in RViz instead.
- cd into infra_ws/src/infra/launch/zed_main.launch 
- comment out the node "camera2map" and uncomment node "aruco2map"

- Add this line in the launch file: "<node pkg="tf2_ros" type="static_transform_publisher" name="link_map_camera" args="x y z yaw nick roll map zed_base_link"/>" using the pose of the camera
- Relaunch using the changed zed_main launch file
- Read the aruco coordinates returned in the terminal by echoing the topic rsu/arucoposes
- Note that this pose uses quaternions, this is however fine for tf2
- It is recommended to put the static publisher in all launch files to be able to switch cameras

- cd into infra_ws/src/infra/launch/zed_main.launch 
- add the associated aruco (id #) in the following format with the appropriate pose inserted:
	<node pkg="tf2_ros" type="static_transform_publisher" name="link_map_aruco#" args="x y z quatX quatY quatZ quatW map aruco#"/>
- comment out the node "aruco2map" and uncomment node "camera2map"
- Run the launch file as described in 1. and launch zed_main.launch, as well as rviz.
- Enable camera feed in rviz as well as the map topic.
- Check if the calibration is adequate be checking if the map is aligned with the camerafeed. Otherwise, the orientation of the aruco marker can be altered slightly in the launch file.

The camera is now calibrated, and can be dynamically placed wherever your heart desires (as long as the marker is visible).
In practice you shouldn't move the camera more than 10 centimeters as the calibration is quite bad


### 3. Adding additional cameras
- In order to launch the second camera, launch camera2.launch as in step 1.
- Note that it could be beneficial to have the second camera run on a seperate computer, since the object detection is computationally heavy. In this case, make sure that you are connected to the alternative computer instead.
