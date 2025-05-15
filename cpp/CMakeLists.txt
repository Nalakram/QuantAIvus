cmake_minimum_required(VERSION 3.20)
project(MarketMind VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Export compile commands for tooling support (e.g., clangd, static analysis)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set build type if not provided
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Group source files by module
set(Api_SRC
    src/api/api_server.cpp
)

set(DATA_SRC
    src/data/DataLoader.cpp
    src/data/Preprocessor.cpp
)

set(INFER_SRC
    src/inference/ModelInference.cpp
    src/inference/FeatureEngineering.cpp
)

set(BACKTEST_SRC
    src/backtesting/Trainer.cpp
    src/backtesting/Evaluator.cpp
)

set(MAIN_SRC
    src/main.cpp
)

# Create main executable
add_executable(MarketMind
    ${API_SRC}
    ${DATA_SRC}
    ${INFER_SRC}
    ${BACKTEST_SRC}
    ${MAIN_SRC}
)

# Create inference_benchmark executable
add_executable(inference_benchmark
    src/inference_benchmark.cpp
)

# Target-specific settings for MarketMind
target_include_directories(MarketMind PRIVATE include src)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(MarketMind PRIVATE DEBUG)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(MarketMind PRIVATE -g)
        target_compile_options(MarketMind PRIVATE -fsanitize=address,undefined)
        target_link_options(MarketMind PRIVATE -fsanitize=address,undefined)
    endif()
else()
    target_compile_definitions(MarketMind PRIVATE NDEBUG)
    target_compile_options(MarketMind PRIVATE -O3)
endif()

# Target-specific settings for inference_benchmark
target_include_directories(inference_benchmark PRIVATE include src)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(inference_benchmark PRIVATE DEBUG)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(inference_benchmark PRIVATE -g)
        target_compile_options(inference_benchmark PRIVATE -fsanitize=address,undefined)
        target_link_options(inference_benchmark PRIVATE -fsanitize=address,undefined)
    endif()
else()
    target_compile_definitions(inference_benchmark PRIVATE NDEBUG)
    target_compile_options(inference_benchmark PRIVATE -O3)
endif()

# Enable warnings for both targets
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(MarketMind PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(inference_benchmark PRIVATE -Wall -Wextra -Wpedantic)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(MarketMind PRIVATE /W4 /EHsc)
    target_compile_options(inference_benchmark PRIVATE /W4 /EHsc)
endif()

# Set output directory for both targets
set_target_properties(MarketMind PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
set_target_properties(inference_benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Install rules for both targets
install(TARGETS MarketMind inference_benchmark DESTINATION bin)

# Enable testing
if(BUILD_TESTING)
    enable_testing()
    # Existing test configuration (e.g., Google Test)
    # ...
endif()
# add_subdirectory(tests)

# Example of linking external libraries (uncomment and modify if needed)
# find_package(gRPC CONFIG REQUIRED)
# target_link_libraries(MarketMind PRIVATE gRPC::grpc++)
# target_link_libraries(inference_benchmark PRIVATE gRPC::grpc++)