cmake_minimum_required(VERSION 3.10)
project(minimal_solver_project)

set(CMAKE_CXX_STANDARD 17)

# Find packages
find_package(Eigen3 REQUIRED)
find_package(Armadillo REQUIRED)

# Define the executable
add_executable(run_solver
    run_solver.cpp
    minimal_solver.cpp
    utility.cpp
)

add_executable(run_two_point_solver
    run_two_point_solver.cpp
    minimal_solver.cpp
    utility.cpp
)

# Include current directory for headers
target_include_directories(run_solver
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${ARMADILLO_INCLUDE_DIRS}
)

target_include_directories(run_two_point_solver
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${ARMADILLO_INCLUDE_DIRS}
)

# Link against the libraries
target_link_libraries(run_solver
    PRIVATE
    Eigen3::Eigen
    ${ARMADILLO_LIBRARIES}
)
target_link_libraries(run_two_point_solver
    PRIVATE
    Eigen3::Eigen
    ${ARMADILLO_LIBRARIES}
)
