~~~python
Your installed Caffe2 version uses cuDNN but I cannot find the cuDNN
libraries.  Please set the proper cuDNN prefixes and / or install cuDNN.

echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export CUDNN_INCLUDE_DIR=/usr/local/cuda/include' >> ~/.bashrc
echo 'export CUDNN_LIB_DIR=/usr/local/cuda/lib64' >> ~/.bashrc
~~~