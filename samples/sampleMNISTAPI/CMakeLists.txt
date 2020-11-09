cmake_minimum_required(VERSION 3.8)
project(test_cuda)

### Compile as C++11
add_compile_options(-std=c++11)

SET(CMAKE_BUILD_TYPE "Debug")  
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")  
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")  


find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(include
                    ${OpenCV_INCLUDE_DIRS}
                    /usr/local/cuda/include
                    /home/wangh/dev/tensorrt/samples/common)

link_directories(/usr/local/cuda/lib64
                 /usr/local/lib
                 /home/wangh/dev/tensorrt/samples/common)

                 
# cuda_add_library(${PROJECT_NAME}_node SHARED src/main.cu})
cuda_add_executable(${PROJECT_NAME}_node src/test2.cu)
# target_link_libraries(${PROJECT_NAME}_node ${OpenCV_LIBS})


# CONFIGURATION
set(TRT_LIB /usr/lib/x86_64-linux-gnu)
set(TRT_INCLUDE /usr/include/x86_64-linux-gnu)
set(NVINFER_LIB ${_NVINFER_LIB})

# Add include directories
include_directories(${CUDA_INC_DIR} ${TRT_INCLUDE} include)

find_library(_NVINFER_LIB nvinfer HINTS ${TRT_LIB} PATH_SUFFIXES lib lib64)

# -------- BUILDING --------
cuda_add_library(test_plugin_trt SHARED src/test_plugin_trt.cpp src/leak_relu.cu)
target_link_libraries(test_plugin_trt
                      nvinfer
                      nvonnxparser
                      nvcaffe_parser 
                      ${NVINFER_LIB})

aux_source_directory(/home/wangh/dev/tensorrt/samples/common SOURCE_CPP)

add_executable(uniTest ${SOURCE_CPP} src/sampleMNISTAPI.cpp)
target_link_libraries(uniTest
                      test_plugin_trt
                      ${NVINFER_LIB}
                      )

add_executable(test1 src/test.cpp)
