cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(resnet_18)

set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_LIST_DIR}/../libtorch")
set(Torch_DIR "${CMAKE_CURRENT_LIST_DIR}/../libtorch/share/cmake/Torch")

find_package(Torch REQUIRED)

add_executable(resnet_18 load_model.cpp)
target_link_libraries(resnet_18 "${TORCH_LIBRARIES}")


set_property(TARGET resnet_18 PROPERTY CXX_STANDARD 17)