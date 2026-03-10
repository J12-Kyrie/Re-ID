cmake_minimum_required(VERSION 3.15)
project(reid LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# DriveWorks SDK (default to sibling driveworks-5.20)
set(DRIVEWORKS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../driveworks-5.20" CACHE PATH "DriveWorks SDK root")
list(APPEND CMAKE_MODULE_PATH "${DRIVEWORKS_DIR}/samples/cmake")

find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(TensorRT MODULE REQUIRED)

# DriveWorks
find_package(dwdynamicmemory CONFIG REQUIRED
    PATHS "${DRIVEWORKS_DIR}/cmake" NO_DEFAULT_PATH)
find_package(driveworks REQUIRED CONFIG
    PATHS "${DRIVEWORKS_DIR}/cmake" NO_DEFAULT_PATH)
find_package(driveworks-shared CONFIG
    PATHS "${DRIVEWORKS_DIR}/cmake" NO_DEFAULT_PATH)

add_library(reid STATIC
    src/ReIdConditioner.cpp
    src/ReIdEngine.cpp
    src/ReIdWorker.cpp
    src/ReIdManager.cpp
    src/ReIdConfigLoader.cpp
)

target_include_directories(reid PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(reid PUBLIC
    driveworks::driveworks
    TensorRT::nvinfer
    CUDA::cudart
    Threads::Threads
)
