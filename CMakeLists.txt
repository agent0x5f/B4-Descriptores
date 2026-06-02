cmake_minimum_required(VERSION 3.10)
project(filtro_sobel)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(filtro_sobel
        main.cpp
)

target_link_libraries(filtro_sobel ${OpenCV_LIBS})