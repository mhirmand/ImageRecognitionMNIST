#include "cnn.h"
#include "layers/conv_layer.h"
#include "layers/maxpool_layer.h"
#include "layers/fc_layer.h"
#include "layers/relu_layer.h"
#include "layers/sigmoid_layer.h"
#include <iostream>
#include <chrono>
#include <numeric>
#include <random>
#include <algorithm>

// ConvLayer implementation
ConvLayer::ConvLayer(int in_channels, int out_channels, int kernel_size,
  int stride, int input_height, int input_width)
  : in_channels(in_channels), out_channels(out_channels),
  kernel_size(kernel_size), stride(stride),
  input_height(input_height), input_width(input_width) {

  output_height = (input_height - kernel_size) / stride + 1;
  output_width = (input_width - kernel_size) / stride + 1;
  initialize_parameters();
}

void ConvLayer::initialize_parameters() {
  weights.resize(out_channels * in_channels * kernel_size * kernel_size);
  biases.resize(out_channels, 0.0f);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> d(0, std::sqrt(2.0 / (in_channels * kernel_size * kernel_size)));
  for (auto& w : weights) w = d(gen);

  weight_gradients.resize(weights.size(), 0.0f);
  bias_gradients.resize(biases.size(), 0.0f);
}

void ConvLayer::forward(const std::vector<float>& input, std::vector<float>& output) {
  output.resize(out_channels * output_height * output_width);
  for (int oc = 0; oc < out_channels; ++oc) {
    for (int oh = 0; oh < output_height; ++oh) {
      for (int ow = 0; ow < output_width; ++ow) {
        float sum = biases[oc];
        for (int ic = 0; ic < in_channels; ++ic) {
          for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
              int ih = oh * stride + kh;
              int iw = ow * stride + kw;
              int input_index = (ic * input_height + ih) * input_width + iw;
              int weight_index = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
              sum += input[input_index] * weights[weight_index];
            }
          }
        }
        output[(oc * output_height + oh) * output_width + ow] = sum;
      }
    }
  }
}

void ConvLayer::backward(const std::vector<float>& input, const std::vector<float>& output,
  const std::vector<float>& output_gradient, std::vector<float>& input_gradient) {
  input_gradient.resize(in_channels * input_height * input_width, 0.0f);
  std::fill(weight_gradients.begin(), weight_gradients.end(), 0.0f);
  std::fill(bias_gradients.begin(), bias_gradients.end(), 0.0f);

  for (int oc = 0; oc < out_channels; ++oc) {
    for (int oh = 0; oh < output_height; ++oh) {
      for (int ow = 0; ow < output_width; ++ow) {
        int output_index = (oc * output_height + oh) * output_width + ow;
        float grad = output_gradient[output_index];

        // Gradient w.r.t. bias
        bias_gradients[oc] += grad;

        for (int ic = 0; ic < in_channels; ++ic) {
          for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
              int ih = oh * stride + kh;
              int iw = ow * stride + kw;
              int input_index = (ic * input_height + ih) * input_width + iw;
              int weight_index = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;

              // Gradient w.r.t. weights
              weight_gradients[weight_index] += input[input_index] * grad;

              // Gradient w.r.t. input
              input_gradient[input_index] += weights[weight_index] * grad;
            }
          }
        }
      }
    }
  }

}

void ConvLayer::update(float learning_rate) {
  for (size_t i = 0; i < weights.size(); ++i) {
    weights[i] -= learning_rate * weight_gradients[i];
  }
  for (size_t i = 0; i < biases.size(); ++i) {
    biases[i] -= learning_rate * bias_gradients[i];
  }
}
