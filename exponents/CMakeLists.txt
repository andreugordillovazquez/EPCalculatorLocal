cmake_minimum_required(VERSION 3.26)
project(untitled12)

# Set vcpkg toolchain (replace with your actual path)
set(CMAKE_TOOLCHAIN_FILE "C:/Users/TESTER/Downloads/vcpkg-master/vcpkg-master/scripts/buildsystems/vcpkg.cmake")

set(CMAKE_CXX_STANDARD 17)

# Eigen3 include (optional)
set(EIGEN3_INCLUDE_DIR "C:/Users/TESTER/Downloads/eigen-3.4.0/eigen-3.4.0")
set(AWSSDK_DIR "C:/Users/TESTER/Downloads/vcpkg-master/vcpkg-master/installed/x64-windows/share/awssdk")
include_directories(${EIGEN3_INCLUDE_DIR})

# Find AWS SDK
find_package(AWSSDK REQUIRED COMPONENTS core dynamodb)

# Add executable (remove unrelated headers)
add_executable(untitled12
        main.cpp
        functions.cpp
        database.h
        database.cpp
        database_main.h
        database_main.cpp
)

# Link AWS SDK using CMake targets
target_link_libraries(untitled12 PRIVATE
        aws-cpp-sdk-core
        aws-cpp-sdk-dynamodb
)