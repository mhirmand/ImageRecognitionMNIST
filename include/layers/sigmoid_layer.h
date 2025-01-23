#include "layer.h"
#include <vector>

/**
 * @brief Sigmoid activation Layer
 * @param input Input tensor (flattened)
 * @param output output tensor (flattened)
 */
class SigmoidLayer : public Layer {
public:
  void forward(const std::vector<float>& input, std::vector<float>& output) override {
    output.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
      output[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }
  }

  void backward(const std::vector<float>& input, const std::vector<float>& output,
    const std::vector<float>& output_gradient, std::vector<float>& input_gradient) override {
    input_gradient.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
      input_gradient[i] = output_gradient[i] * output[i] * (1 - output[i]);
    }
  }

  void update(float learning_rate) override {
    // Sigmoid has no parameters to update
  }
};