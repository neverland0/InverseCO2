cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(torch-optim)

set(Torch_DIR "/home/akagi/.local/lib/python3.10/site-packages/torch/share/cmake/Torch")
set(Nlopt_DIR "/home/akagi/nlopt-2.7.1")
# 需要找到Libtorch
find_package(Torch REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
link_directories(${Eigen_INCLUDE_DIRS})
add_executable(${PROJECT_NAME} torch-optimize.cc)
target_link_libraries(${PROJECT_NAME} -lnlopt)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
target_link_libraries(${PROJECT_NAME}
   ${Eigen_LIBRARIES}
   )

# Libtorch是基于C++14来实现的
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)

