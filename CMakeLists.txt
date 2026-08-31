cmake_minimum_required(VERSION 3.20)
project(llm-on-legacy-gpus VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCL REQUIRED)

add_executable(relic
    src/main.cpp
    src/architecture.cpp
    src/qwen35_state.cpp
    src/decoder.cpp
    src/model.cpp
    src/opencl_backend.cpp
    src/tokenizer.cpp
    src/inference.cpp
    src/planner/hardware_profile.cpp
    src/planner/adaptive_planner.cpp
    src/memory/memory_engine.cpp
    src/memory/pinned_host_pool.cpp
    src/memory/async_prefetcher.cpp
    src/kv/kv_manager.cpp
    src/speculative/distributed_speculative.cpp
)

add_executable(relic_test
    src/test_opencl.cpp
    src/architecture.cpp
    src/opencl_backend.cpp
    src/qwen35_state.cpp
    src/decoder.cpp
    src/model.cpp
    src/planner/hardware_profile.cpp
    src/planner/adaptive_planner.cpp
    src/memory/memory_engine.cpp
    src/memory/pinned_host_pool.cpp
    src/memory/async_prefetcher.cpp
    src/kv/kv_manager.cpp
    src/speculative/distributed_speculative.cpp
)
target_include_directories(relic_test PRIVATE src ${OpenCL_INCLUDE_DIRS})
target_link_libraries(relic_test PRIVATE ${OpenCL_LIBRARIES})
target_compile_definitions(relic_test PRIVATE CL_TARGET_OPENCL_VERSION=120)

add_executable(relic_dump_model
    src/dump_model.cpp
    src/architecture.cpp
)
target_include_directories(relic_dump_model PRIVATE src)
target_link_libraries(relic_dump_model PRIVATE ${OpenCL_LIBRARIES})
target_compile_definitions(relic_dump_model PRIVATE CL_TARGET_OPENCL_VERSION=120)

target_include_directories(relic PRIVATE src ${OpenCL_INCLUDE_DIRS})
target_link_libraries(relic PRIVATE ${OpenCL_LIBRARIES})
target_compile_definitions(relic PRIVATE CL_TARGET_OPENCL_VERSION=120)
