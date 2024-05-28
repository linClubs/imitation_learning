~~~python
# 机械臂
pip install empy==3.3.4 catkin_pkg rospkg
sudo apt install libkdl-parser-dev can-utils net-tools

# realsense-2.5.0、realsense-ros-2.3.2
sudo apt-get install git libssl-dev libusb-1.0-0-dev pkg-config libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev
sudo apt-get install -y ros-$ROS_DISTRO-ddynamic-reconfigure ros-$ROS_DISTRO-rgbd-launch

# ros_astra_camera
sudo apt install libuvc-dev libgoogle-glog-dev ros-$ROS_DISTRO-rgbd-launch ros-$ROS_DISTRO-libuvc-camera ros-$ROS_DISTRO-libuvc-ros

# agilex_ros
sudo apt install ros-$ROS_DISTRO-serial ros-noetic-tf2-sensor-msgs libpcap-dev libasio-dev -y
sudo apt install ros-$ROS_DISTRO-nav* ros-noetic-gmapping
~~~