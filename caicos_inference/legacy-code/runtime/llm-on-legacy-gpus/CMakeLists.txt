cmake_minimum_required(VERSION 3.20)
project(llm-on-legacy-gpus VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCL REQUIRED)

add_executable(caicos_rt
    src/main.cpp
    src/model.cpp
    src/opencl_backend.cpp
    src/tokenizer.cpp
    src/inference.cpp
)

add_executable(caicos_rt_test
    src/test_opencl.cpp
    src/opencl_backend.cpp
)
target_include_directories(caicos_rt_test PRIVATE src ${OpenCL_INCLUDE_DIRS})
target_link_libraries(caicos_rt_test PRIVATE ${OpenCL_LIBRARIES})
target_compile_definitions(caicos_rt_test PRIVATE CL_TARGET_OPENCL_VERSION=120)

add_executable(dump_model
    src/dump_model.cpp
)
target_include_directories(dump_model PRIVATE src)
target_link_libraries(dump_model PRIVATE ${OpenCL_LIBRARIES})
target_compile_definitions(dump_model PRIVATE CL_TARGET_OPENCL_VERSION=120)

target_include_directories(caicos_rt PRIVATE src ${OpenCL_INCLUDE_DIRS})
target_link_libraries(caicos_rt PRIVATE ${OpenCL_LIBRARIES})
target_compile_definitions(caicos_rt PRIVATE CL_TARGET_OPENCL_VERSION=120)
