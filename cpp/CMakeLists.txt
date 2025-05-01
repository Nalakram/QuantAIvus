cmake_minimum_required(VERSION 3.20)
project(StockPredictionApp VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Export compile commands for tooling support (e.g., clangd, static analysis)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set build type if not provided
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Group source files by module
set(API_SRC
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

# Create executable
add_executable(StockPredictionApp
    ${API_SRC}
    ${DATA_SRC}
    ${INFER_SRC}
    ${BACKTEST_SRC}
    ${MAIN_SRC}
)

# Target-specific settings
target_include_directories(StockPredictionApp PRIVATE include src)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(StockPredictionApp PRIVATE DEBUG)

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(StockPredictionApp PRIVATE -g)
        target_compile_options(StockPredictionApp PRIVATE -fsanitize=address,undefined)
        target_link_options(StockPredictionApp PRIVATE -fsanitize=address,undefined)
    endif()
else()
    target_compile_definitions(StockPredictionApp PRIVATE NDEBUG)
    target_compile_options(StockPredictionApp PRIVATE -O3)
endif()


# Enable warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(StockPredictionApp PRIVATE -Wall -Wextra -Wpedantic)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(StockPredictionApp PRIVATE /W4 /EHsc)
endif()

# Set output directory
set_target_properties(StockPredictionApp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Install rules
install(TARGETS StockPredictionApp DESTINATION bin)

# Enable testing (uncomment when adding tests)
# enable_testing()
# add_subdirectory(tests)

# Example of linking external libraries (uncomment and modify if needed)
# find_package(gRPC CONFIG REQUIRED)
# target_link_libraries(StockPredictionApp PRIVATE gRPC::grpc++)