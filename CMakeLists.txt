cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(my_torch_app)

set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_LIST_DIR}/../libtorch")
set(Torch_DIR "${CMAKE_CURRENT_LIST_DIR}/../libtorch/share/cmake/Torch")

find_package(Torch REQUIRED)

add_executable(my_torch_app load_model.cpp)
target_link_libraries(my_torch_app "${TORCH_LIBRARIES}")


set_property(TARGET my_torch_app PROPERTY CXX_STANDARD 17)