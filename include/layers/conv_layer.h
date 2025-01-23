#pragma once
#include "layer.h"
#include <vector>
#include <random>

/**
 * @brief 2D Convolutional Layer
 * @param in_channels Number of input channels
 * @param out_channels Number of output channels
 * @param kernel_size Size of convolutional kernel
 * @param stride Convolution stride
 * @param input_height Input feature map height
 * @param input_width Input feature map width
 */
class ConvLayer : public Layer {
public:
  ConvLayer(int in_channels, int out_channels, int kernel_size,
    int stride, int input_height, int input_width);
  void forward(const std::vector<float>& input, std::vector<float>& output) override;
  void backward(const std::vector<float>& input, const std::vector<float>& output,
    const std::vector<float>& output_gradient, std::vector<float>& input_gradient) override;
  void update(float learning_rate) override;

private:
  int in_channels, out_channels, kernel_size, stride;
  int input_height, input_width, output_height, output_width;
  std::vector<float> weights, biases;
  std::vector<float> weight_gradients, bias_gradients;

  void initialize_parameters();
};

// Implementation in cnn.cpp