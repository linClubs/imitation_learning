#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>

#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>


#include <memory>



int main(int argc, char** argv)
{   
    std::cout << "hello libtorch." << std::endl;
  
    
    torch::Tensor tensor = torch::arange(0, 9).reshape({3, 3});  // range 创建0-8数字3x3的矩阵
    torch::Tensor A = torch::randn({3, 5});    // randn创建均值为0，方差为1的分布
    torch::Tensor B = torch::rand({2, 3});     // 创建0-1之间均匀分布
    
    torch::Tensor C = B * 2 - 1;  // 使用0-1均匀分布生成-1到1直接的均匀分布
    
    // 打印tensor内容
    std::cout << tensor << std::endl;
    std::cout << A << std::endl;
    std::cout << B << std::endl;
    std::cout << C << std::endl;

    if (torch::cuda::is_available()) 
        std::cout << "Cuda available\n";
    else 
        std::cout << "Cuda not available\n";

    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) 
    {
        device = torch::Device(torch::kCUDA);
        std::cout << "\033[32m\nCUDA is available. Using GPU.\033[0m\n\n";
    }

      

    auto module = torch::jit::load("/home/lin/ckpt_dir01/policy_last.pth");
                

    // torch::jit::script::Module module = torch::jit::load("/home/lin/code/ultralytics/weights/yolov8l-pose.pt");

    return 0;
}