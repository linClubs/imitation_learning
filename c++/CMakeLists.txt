cmake_minimum_required(VERSION 3.2)
project(pro)

# set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# set(CUDNN_INCLUDE_DIR /usr/local/cuda/include)
# set(CUDNN_LIBRARY /usr/local/cuda/lib64/libcudnn.so)

# find_package(CUDNN REQUIRED)
# message("\nCUDNN_INCLUDE:" ${CUDNN_INCLUDE_DIR})
# message("CUDNN_LIBS:" ${CUDNN_LIBRARIES} "\n")

message("CUDNN_LIBS:" ${CMAKE_CURRENT_LIST_DIR} "\n")


# if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
# 	set(CMAKE_CUDA_ARCHITECTURES 70 75 80 89)
# endif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)

find_package(CUDA REQUIRED)
# enable_language(CUDA)  # 添加对cuda的支持





find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# set(Torch_DIR /home/lin/software/libtorch_cpu/share/cmake/Torch)
set(Torch_DIR /home/lin/software/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
message("\nTorch_INCLUDE:" ${TORCH_INCLUDE_DIRS})
message("Torch_LIBS:" ${TORCH_LIBRARIES} "\n")

find_package(CUDA REQUIRED)
enable_language(CUDA)  # 添加对cuda的支持
message("\nCUDA_INCLUDE:" ${CUDA_INCLUDE_DIRS})
message("CUDA_LIBS:"$${CUDA_LIBRARIES} "\n")

message("\n----------")
torch_cuda_get_nvcc_gencode_flag(NVCC_FLAGS_EXTRA)
message("----------\n")


# # set(CMAKE_CXX_FLAGS "${CAMKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# # set(CUDA_INCLUDE_DIRS "/usr/local/cuda/include")





add_executable(main main.cpp)
target_link_libraries(main ${Torch_LIBRARIES} ${OpenCV_LIBS} ${TORCH_LIBRARIES})

set_property(TARGET main PROPERTY CXX_STANDARD 14)