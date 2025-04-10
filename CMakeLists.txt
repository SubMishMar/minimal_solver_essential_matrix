cmake_minimum_required(VERSION 3.10)
project(minimal_solver)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

# Add your executable
add_executable(test test.cpp)

# Link Eigen
target_link_libraries(test Eigen3::Eigen ${OpenCV_LIBS})