# use image with cuda drivers and vulkan
FROM px4io/px4-dev-ros-melodic:latest

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

ENV ROS_DISTRO=melodic
RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc

# Create the workspace and clone Clover sources:
RUN cd ~/ && \
    git clone --recursive --depth 1 --branch v1.12.0 https://github.com/PX4/PX4-Autopilot.git ~/PX4-Autopilot && \
    cd ~/PX4-Autopilot && \
    DONT_RUN=1 make px4_sitl_default gazebo
    

# # Install PX4 prerequisites
RUN cd ~/PX4-Autopilot/Tools/setup && \
    sudo ./ubuntu.sh --no-nuttx

RUN apt install -y ros-melodic-cv-bridge ros-melodic-mavros ros-melodic-mavros-extras

RUN wget https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh && bash ./install_geographiclib_datasets.sh

RUN cd ~/PX4-Autopilot && \
    DONT_RUN=1 make px4_sitl_default gazebo && \
    ./Tools/setup_gazebo.bash /root/PX4-Autopilot /root/PX4-Autopilot/build/px4_sitl_default

RUN echo "ROS_PACKAGE_PATH=/opt/ros/melodic/share/:/root/PX4-Autopilot:/root/PX4-Autopilot/Tools/sitl_gazebo" >> ~/.bashrc