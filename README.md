# Real Time CNN Visualization

This is an UI application, which allows to visualize Convolutional Neural Networks in real time.
First, Activation maps of convolutional layers as well activations of fully connected layer are visualized and available for applying more advance visualization techniques.

## Visualization Algorithms

* [Grad-CAM](https://arxiv.org/abs/1610.02391 "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization")
* [Guided Backprop](https://arxiv.org/abs/1412.6806 "Striving for Simplicity: The All Convolutional Net")

## Requirements

* [Docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
* [Docker Compose](https://docs.docker.com/compose/install/) (optional)
* Recent NVIDIA drivers (`nvidia-384` on Ubuntu)
* [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker )


## Usage

```
docker build -t basecv . # Build Docker image which contains all the requirements
docker run  --runtime nvidia --env DISPLAY=$DISPLAY -v="/tmp/.X11-unix:/tmp/.X11-unix:rw"  -v=$(pwd)/.keras:/root/.keras  -v="$(pwd)/..:$(pwd)/.." -w=$(pwd) -it  basecv python3 main.py --stream "your/stream/uri"
```

`python3 main.py -h`

```
usage: main.py [-h] [--stream STREAM] [--network NETWORK]

optional arguments:
  -h, --help         show this help message and exit
  --stream STREAM    Video stram URI, path to video or webcam number based on
                     which the network is visualized
  --network NETWORK  Network to visualise (VGG16,ResNet50 ...)
```
<!-- #### With Docker Compose

```
docker-compose build
docker-compose run vis#### With pure Docker
``` -->
#### With pure Docker
## Troubleshooting

#### Could not connect to any X display.

The X Server should allow connections from a docker container.

Run `xhost +local:docker`, also check [this](https://forums.docker.com/t/start-a-gui-application-as-root-in-a-ubuntu-container/17069)
