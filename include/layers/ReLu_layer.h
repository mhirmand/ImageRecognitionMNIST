#include "layer.h"
#include <vector>

/**
 * @brief ReLu activation Layer
 * @param input Input tensor (flattened)
 * @param output output tensor (flattened)
 */
class ReLULayer : public Layer {
public:
  void forward(const std::vector<float>& input, std::vector<float>& output) override {
    output.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
      output[i] = std::max(0.0f, input[i]);
    }
  }

  void backward(const std::vector<float>& input, const std::vector<float>& output,
    const std::vector<float>& output_gradient, std::vector<float>& input_gradient) override {
    input_gradient.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
      input_gradient[i] = input[i] > 0 ? output_gradient[i] : 0;
    }
  }

  void update(float learning_rate) override {
    // ReLU has no parameters to update
  }
};