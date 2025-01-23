#pragma once
#include "layer.h"
#include <vector>
#include <random>

/**
 * @brief Fully Connected Layer
 * @param input_size Number of input features
 * @param output_size Number of output features
 */
class FCLayer : public Layer {
public:
  FCLayer(int input_size, int output_size);
  void forward(const std::vector<float>& input, std::vector<float>& output) override;
  void backward(const std::vector<float>& input, const std::vector<float>& output,
    const std::vector<float>& output_gradient, std::vector<float>& input_gradient) override;
  void update(float learning_rate) override;

private:
  int input_size, output_size;
  std::vector<float> weights, biases;
  std::vector<float> weight_gradients, bias_gradients;

  void initialize_parameters();
};

// Implementation in cnn.cpp