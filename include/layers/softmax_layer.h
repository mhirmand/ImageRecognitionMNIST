#include "layer.h"
#include <vector>

/**
 * @brief SoftMax activation Layer
 * @param input Input tensor (flattened)
 * @param output output tensor (flattened)
 */
class SoftMaxLayer : public Layer {
public:
  void forward(const std::vector<float>& input, std::vector<float>& output) override {
    output.resize(input.size());
    float max_val = *std::max_element(input.begin(), input.end());
    float sum = 0.0f;

    // Compute exponentials and sum for numerical stability
    for (size_t i = 0; i < input.size(); ++i) {
      output[i] = std::exp(input[i] - max_val);
      sum += output[i];
    }

    // Normalize to get probabilities
    for (size_t i = 0; i < output.size(); ++i) {
      output[i] /= sum;
    }
  }

  void backward(const std::vector<float>& input, const std::vector<float>& output,
    const std::vector<float>& output_gradient, std::vector<float>& input_gradient) override {
    input_gradient.resize(input.size());

    // Gradient of SoftMax with respect to input
    for (size_t i = 0; i < input.size(); ++i) {
      input_gradient[i] = 0.0f;
      for (size_t j = 0; j < output.size(); ++j) {
        float kronecker_delta = (i == j) ? 1.0f : 0.0f;
        input_gradient[i] += output_gradient[j] * output[i] * (kronecker_delta - output[j]);
      }
    }
  }

  void update(float learning_rate) override {
    // SoftMax has no parameters to update
  }
};