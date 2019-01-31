FROM tensorflow/tensorflow:latest-py3

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        software-properties-common \
        python3-pip \
 && add-apt-repository -y ppa:jonathonf/python-3.6 \
 && apt-get update \
 && apt-get install -y python3.6 python3.6-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# pip has to be installed before setuptools, setuptools has to be installed before tensorflow
RUN python3.6 -m pip install --no-cache-dir -U pip
RUN python3.6 -m pip install --no-cache-dir -U setuptools
# also useful
RUN python3.6 -m pip install --no-cache-dir ipython requests numpy pandas quandl
RUN python3.6 -m pip install --no-cache-dir tensorflow==1.11.0

# Tensorflow should be fine by here

RUN python3.6-config --includes \
  && cd /usr/bin && rm python3 && ln -s python3.6 python3


RUN apt-get -y update -qq && \
    apt-get -y install wget \
                       unzip \

                       # Required
                       build-essential \
                       cmake \
                       git \
                       pkg-config \
                       libatlas-base-dev \
                       libgtk2.0-dev \
                       libavcodec-dev \
                       libavformat-dev \
                       libswscale-dev \

                       # Optional
                       libtbb2 libtbb-dev \
                       libjpeg-dev \
                       libpng-dev \
                       libtiff-dev \
                       libv4l-dev \
                       libdc1394-22-dev \

                       qt4-default \

                       # Missing libraries for GTK
                       libatk-adaptor \
                       libcanberra-gtk-module \

                       # Tools
                       imagemagick \

                       # For use matplotlib.pyplot in python
                       python3-tk \
		       python-tk

WORKDIR /
    # Get OpenCV
RUN git clone https://github.com/opencv/opencv.git
ENV OPENCV_VERSION=3.4.3
RUN  cd opencv &&\
    git checkout $OPENCV_VERSION &&\
    cd / &&\
    # Get OpenCV contrib modules
    git clone https://github.com/opencv/opencv_contrib &&\
    cd opencv_contrib &&\
    git checkout $OPENCV_VERSION &&\
    mkdir /opencv/build

RUN  cd /opencv/build &&\
    # Lets build OpenCV
    cmake \
      -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_C_EXAMPLES=OFF \
      -D INSTALL_PYTHON_EXAMPLES=OFF \
      -D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
      -D BUILD_EXAMPLES=OFF \
      -D BUILD_NEW_PYTHON_SUPPORT=ON \
      -D BUILD_DOCS=OFF \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D WITH_TBB=ON \
      -D WITH_OPENMP=ON \
      -D WITH_IPP=ON \
      -D WITH_CSTRIPES=ON \
      -D WITH_OPENCL=ON \
      -D WITH_V4L=ON \
      -D BUILD_opencv_python3=ON \
      .. &&\
    make -j$NUM_CORES &&\
    make install &&\
    ldconfig &&\
    # Clean the install from sources
    cd / &&\
    rm -r /opencv &&\
    rm -r /opencv_contrib

RUN apt-get update && apt-get -y install libgtk-3-dev libboost-all-dev
RUN pip3 install PyQt5
WORKDIR /src
CMD ["python3", "main.py"]
