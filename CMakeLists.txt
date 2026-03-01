cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(my_torch_app)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(my_torch_app main.cpp)
target_link_libraries(my_torch_app "${TORCH_LIBRARIES}")
set_property(TARGET my_torch_app PROPERTY CXX_STANDARD 14)