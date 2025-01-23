#pragma once
#include "layer.h"
#include <vector>

/**
 * @brief Max Pooling Layer
 * @param kernel_size Pooling window size
 * @param stride Pooling stride
 * @param input_height Input feature map height
 * @param input_width Input feature map width
 * @param channels Number of input channels
 */
class MaxPoolLayer : public Layer {
public:
  MaxPoolLayer(int kernel_size, int stride,
    int input_height, int input_width, int channels);
  void forward(const std::vector<float>& input, std::vector<float>& output) override;
  void backward(const std::vector<float>& input, const std::vector<float>& output,
    const std::vector<float>& output_gradient, std::vector<float>& input_gradient) override;
  void update(float learning_rate) override;

private:
  int kernel_size, stride;
  int input_height, input_width, output_height, output_width, channels;
  std::vector<int> max_indices;
};

// Implementation in cnn.cpp