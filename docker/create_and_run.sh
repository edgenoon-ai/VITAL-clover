docker build -t px4-sim -f Dockerfile . --no-cache

mkdir $HOME/shared

chmod +x xauth.sh
./xauth.sh

docker run -it --gpus all --privileged --env="DISPLAY=:1" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"     --env="XAUTHORITY=$XAUTH"     --volume="$XAUTH:$XAUTH"     --volume="$HOME/shared:/home/px4/share:rw"     --env="NVIDIA_VISIBLE_DEVICES=all"     --env="NVIDIA_DRIVER_CAPABILITIES=all"     --network=host --name=px4_sim px4-sim bash
