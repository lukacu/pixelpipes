Installing dependencies and compiling
=====================================

The project depends on OpenCV (a C++ dependency) as well as some Python utility libraries. A PyBind11 header library is used to generate Python bindings for the C++ core, it is installed as a Pip dependency. Optionally, the C++ code can be built using PyTorch support, this way the data can be converted directly to PyTorch tensors.

Unix (Ubuntu)
-------------

You should have at least 19.10 version of Ubuntu (we recommend 20.04) since you need GCC 9+ compiler for build. The only non-python dependency is OpenCV for C++ which can be obtained from `OpenCV <https://github.com/opencv/opencv/releases/>`_ repository or and built using cmake and make. You can use following commands::

    $ wget https://github.com/opencv/opencv/archive/<opencv_version>.tar.gz  
    $ tar -xvf <opencv_version>.tar.gz
    $ cd opencv-<opencv_version>
    $ mkdir build && cd build
    $ sudo cmake ../
    $ sudo make
    $ sudo make install

To build a development version of the package (the only kind that is supported at this stage of the project), you can install required python packages and compile the C++ core with the following commands::

    $ pip3 install -r requirements.txt
    $ python3 setup.py build_ext --inplace

Additionally, you might have to supply location of OpenCV headers and libraries using::

    $ python3 setup.py build_ext --inplace --include-dirs DIRS --library-dirs LIB_DIRS

To test if the build was successful you can run::

    $ python3 -m unittest -v

Windows
-------

.. note:: 
    | <opencv_version> represents OpenCV version. Tested and working with 3.1.0
    | <cuda_version> represents Nvidia CUDA version. Tested and working with 11-2
    | <cudnn_version> represents Nvidia cuDNN version. Tested and working with 8.1.1

Compiling for Windows is currently not possible. For now, the only way to is to use WSL2 (Windows Subsystem for Linux) with Ubuntu 20.04. If you want to use Pixelpipes framework together with deep learning frameworks (Tensorflow and PyTorch) you need a Nvidia GPU with sufficient video card drivers, Nvidia CUDA Toolkit and Nvidia CUDA Deep Neural Network library (cuDNN). 

Follow the instructions for `WSL2 <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`_ installation. Then you will need to update Windows 10 to Dev build (Windows build version 20145 or higher). This is possible by registering for `Windows Insider Program <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`_ and installing given Windows updates. With this installed, your windows GPU drivers will also apply for WSL2. To install correct version of Nvidia CUDA Toolkit and Nvidia cuDNN check build configurations for `Tensorflow <https://www.tensorflow.org/install/source>`_ and `PyTorch <https://pytorch.org/get-started/locally/>`_. `Repository <http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/>`_ with CUDA library for Ubuntu 20.04. To add repository to sources list and install CUDA Toolkit use following commands::

    $ sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
    $ sudo sh -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
    $ sudo apt-get update
    $ sudo apt-get install -y cuda-toolkit-<cuda_version>

To install Nvidia cuDNN library first login to Nvidia developer and download Linux (x86_64) version from `cuDNN archive <https://developer.nvidia.com/rdp/cudnn-archive>`_, extract it and copy cuDNN library files to CUDA directory using following commands:: 

    $ tar -xzvf cudnn-<cuda_version>-linux-x64-<cudnn_version>.33.tgz
    $ sudo cp cuda/include/cudnn*.h /usr/local/cuda/include 
    $ sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64 
    $ sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

In case the the newly installed libraries are not automatically added to path you can do it manually by appending following lines to .bashrc and restart the WSL2::

    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    export PATH=$PATH:/usr/local/cuda/bin

To test GPU drivers and CUDA installation run::

    $ nvcc --version
    $ nvidia-smi

With this you can install Tensorflow and PyTorch with just::

    $ pip3 install torch
    $ pip3 install tensorflow-gpu

 