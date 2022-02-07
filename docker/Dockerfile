# use image with cuda drivers and vulkan
FROM px4io/px4-dev-ros-noetic:latest

USER root
# Install sudo
RUN apt-get update \
  && apt-get install -y sudo

# set time zone
ENV TZ=Europe/Warsaw
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt update
RUN apt install -y tzdata

# install AirSim requirements
RUN apt update
RUN apt install -y --no-install-recommends \
	python3 \
	python3-pip \
    python3-tk \
    python3-rosdep \
    build-essential \
	libglu1-mesa-dev \
	xdg-user-dirs \
	pulseaudio \
    lsb-release  \
	x11-xserver-utils \
	git \
    wget \
	rsync \
	unzip \
	g++ \
	vim \ 
	nano

ENV ROS_PYTHON_VERSION=3
ENV ROS_DISTRO=noetic
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

# Create the workspace and clone Clover sources:
RUN mkdir -p ~/catkin_ws/src && \
    cd ~/catkin_ws/src && \
    git clone --depth 1 https://github.com/edgenoon-ai/clover && \
    git clone --depth 1 https://github.com/CopterExpress/ros_led && \
    git clone --depth 1 https://github.com/ethz-asl/mav_comm

# Install all prerequisites using rosdep
RUN cd ~/catkin_ws && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -y

# Install Python-dependencies:
RUN sudo /usr/bin/python3 -m pip install -r ~/catkin_ws/src/clover/clover/requirements.txt

# Get PX4 sources
RUN cd ~/catkin_ws/src && \
    git clone --recursive --depth 1 --branch v1.12.0 https://github.com/PX4/PX4-Autopilot.git ~/PX4-Autopilot && \
    ln -s ~/PX4-Autopilot ~/catkin_ws/src/PX4-Autopilot && \
    ln -s ~/PX4-Autopilot/Tools/sitl_gazebo ~/catkin_ws/src/sitl_gazebo && \
    ln -s ~/PX4-Autopilot/mavlink ~/catkin_ws/src/

# Install PX4 prerequisites
RUN cd ~/catkin_ws/src/PX4-Autopilot/Tools/setup && \
    sudo ./ubuntu.sh --no-nuttx

# Install geographiclib datasets
RUN sudo /opt/ros/noetic/lib/mavros/install_geographiclib_datasets.sh

RUN /bin/bash -c '. /opt/ros/noetic/setup.bash; cd ~/catkin_ws; catkin_make'

RUN echo "source /root/catkin_ws/devel/setup.bash" >> ~/.bashrc

# Add the Clover airframe
RUN cp /root/catkin_ws/src/clover/clover_simulation/airframes/4500_clover ~/PX4-Autopilot/ROMFS/px4fmu_common/init.d-posix/airframes/
