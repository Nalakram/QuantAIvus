#include <chrono>
#include <iostream>
#include <span>  // C++20 feature
#include <vector>
#include <cstdlib>
#include <string>

// Placeholder for matrix multiplication to simulate inference
void simulateInference(std::span<float> input, std::span<const float> weights, std::span<float> output, int size) {
    // Simulate a simple matrix operation: output = input * weights
    for (int i = 0; i < size; ++i) {
        output[i] = 0.0f;
        for (int j = 0; j < size; ++j) {
            output[i] += input[j] * weights[i * size + j];
        }
    }
}

int main(int argc, char* argv[]) {
    // Parse max-latency argument
    float max_latency_ms = 1.0f;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--max-latency" && i + 1 < argc) {
            max_latency_ms = std::stof(argv[++i]);
        }
    }

    // Initialize data for simulation
    const int size = 100; // Small matrix size for quick execution
    std::vector<float> input(size, 1.0f);
    std::vector<float> weights(size * size, 0.1f);
    std::vector<float> output(size, 0.0f);

    // Use std::span for safer array handling (C++20)
    auto input_span = std::span(input);
    auto weights_span = std::span(weights);
    auto output_span = std::span(output);

    // Measure inference time
    auto start = std::chrono::high_resolution_clock::now();
    simulateInference(input_span, weights_span, output_span, size);
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate latency in milliseconds
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    float latency_ms = duration.count() / 1'000'000.0f;

    // Output result
    std::cout << "Inference latency: " << latency_ms << " ms" << std::endl;

    // Check if latency exceeds threshold
    if (latency_ms > max_latency_ms) {
        std::cerr << "Error: Latency (" << latency_ms << " ms) exceeds max allowed ("
                  << max_latency_ms << " ms)" << std::endl;
        return 1; // Exit with failure
    }

    std::cout << "Benchmark passed!" << std::endl;
    return 0; // Exit with success
}