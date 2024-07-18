#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>
#include <numeric> // For std::iota

// Layer interface (abstract class) for neural network layers
class Layer {
public:
  virtual void forward(const std::vector<float>& input, std::vector<float>& output) = 0;
  virtual void backward(const std::vector<float>& input, const std::vector<float>& output,
    const std::vector<float>& output_gradient, std::vector<float>& input_gradient) = 0;
  virtual void update(float learning_rate) = 0;
  virtual ~Layer() {}
};

class ConvLayer : public Layer {
public:
  ConvLayer(int in_channels, int out_channels, int kernel_size, int stride)
    : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size), stride(stride) {
    // Initialize weights and biases
    weights.resize(out_channels * in_channels * kernel_size * kernel_size);
    biases.resize(out_channels);

    // Random initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);
    for (auto& w : weights) w = d(gen) * std::sqrt(2.0 / (in_channels * kernel_size * kernel_size));
    for (auto& b : biases) b = 0;
  }

  void forward(const std::vector<float>& input, std::vector<float>& output) override {
    // Implement convolution forward pass
    // This is a placeholder implementation
    output = input; // Replace with actual convolution
  }

  void backward(const std::vector<float>& input, const std::vector<float>& output,
    const std::vector<float>& output_gradient, std::vector<float>& input_gradient) override {
    // Implement convolution backward pass
    // This is a placeholder implementation
    input_gradient = output_gradient; // Replace with actual gradient computation
  }

  void update(float learning_rate) override {
    // Update weights and biases
    // This is a placeholder implementation
  }

private:
  int in_channels, out_channels, kernel_size, stride;
  std::vector<float> weights, biases;
};

class MaxPoolLayer : public Layer {
public:
  MaxPoolLayer(int kernel_size, int stride)
    : kernel_size(kernel_size), stride(stride) {}

  void forward(const std::vector<float>& input, std::vector<float>& output) override {
    // Implement max pooling forward pass
    // This is a placeholder implementation
    output = input; // Replace with actual max pooling
  }

  void backward(const std::vector<float>& input, const std::vector<float>& output,
    const std::vector<float>& output_gradient, std::vector<float>& input_gradient) override {
    // Implement max pooling backward pass
    // This is a placeholder implementation
    input_gradient = output_gradient; // Replace with actual gradient computation
  }

  void update(float learning_rate) override {
    // No parameters to update in max pooling layer
  }

private:
  int kernel_size, stride;
};

class FCLayer : public Layer {
public:
  FCLayer(int input_size, int output_size)
    : input_size(input_size), output_size(output_size) {
    // Initialize weights and biases
    weights.resize(input_size * output_size);
    biases.resize(output_size);

    // Random initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);
    for (auto& w : weights) w = d(gen) * std::sqrt(2.0 / input_size);
    for (auto& b : biases) b = 0;
  }

  void forward(const std::vector<float>& input, std::vector<float>& output) override {
    // Implement fully connected layer forward pass
    output.resize(output_size);
    for (int i = 0; i < output_size; ++i) {
      output[i] = biases[i];
      for (int j = 0; j < input_size; ++j) {
        output[i] += input[j] * weights[i * input_size + j];
      }
    }
  }

  void backward(const std::vector<float>& input, const std::vector<float>& output,
    const std::vector<float>& output_gradient, std::vector<float>& input_gradient) override {
    // Implement fully connected layer backward pass
    input_gradient.resize(input_size);
    std::fill(input_gradient.begin(), input_gradient.end(), 0);
    for (int i = 0; i < output_size; ++i) {
      for (int j = 0; j < input_size; ++j) {
        input_gradient[j] += output_gradient[i] * weights[i * input_size + j];
      }
    }
  }

  void update(float learning_rate) override {
    // Update weights and biases
    // This is a placeholder implementation
  }

private:
  int input_size, output_size;
  std::vector<float> weights, biases;
};

class ReLULayer : public Layer{
public:
    void forward(const std::vector<float>&input, std::vector<float>&output) override {
        output.resize(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = std::max(0.0f, input[i]);
        }
    }

    void backward(const std::vector<float>&input, const std::vector<float>&output,
                  const std::vector<float>&output_gradient, std::vector<float>&input_gradient) override {
        input_gradient.resize(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            input_gradient[i] = input[i] > 0 ? output_gradient[i] : 0;
        }
    }

    void update(float learning_rate) override {
      // ReLU has no parameters to update
  }
};

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

  void update(float learning_rate) override {} // No parameters to update
};

class TanhLayer : public Layer {
public:
  void forward(const std::vector<float>& input, std::vector<float>& output) override {
    output.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
      output[i] = std::tanh(input[i]);
    }
  }

  void backward(const std::vector<float>& input, const std::vector<float>& output,
    const std::vector<float>& output_gradient, std::vector<float>& input_gradient) override {
    input_gradient.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
      input_gradient[i] = output_gradient[i] * (1 - output[i] * output[i]);
    }
  }

  void update(float learning_rate) override {} // No parameters to update
};

class ConvLayer : public Layer {
public:
  ConvLayer(int in_channels, int out_channels, int kernel_size, int stride, int input_height, int input_width)
    : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size), stride(stride),
    input_height(input_height), input_width(input_width) {

    output_height = (input_height - kernel_size) / stride + 1;
    output_width = (input_width - kernel_size) / stride + 1;

    // Initialize weights and biases
    weights.resize(out_channels * in_channels * kernel_size * kernel_size);
    biases.resize(out_channels);

    // Xavier initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, std::sqrt(2.0 / (in_channels * kernel_size * kernel_size)));
    for (auto& w : weights) w = d(gen);
    std::fill(biases.begin(), biases.end(), 0.0f);

    // Initialize gradients
    weight_gradients.resize(weights.size(), 0.0f);
    bias_gradients.resize(biases.size(), 0.0f);
  }

  void forward(const std::vector<float>& input, std::vector<float>& output) override {
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

  void backward(const std::vector<float>& input, const std::vector<float>& output,
    const std::vector<float>& output_gradient, std::vector<float>& input_gradient) override {
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

  void update(float learning_rate) override {
    for (size_t i = 0; i < weights.size(); ++i) {
      weights[i] -= learning_rate * weight_gradients[i];
    }
    for (size_t i = 0; i < biases.size(); ++i) {
      biases[i] -= learning_rate * bias_gradients[i];
    }
  }

private:
  int in_channels, out_channels, kernel_size, stride;
  int input_height, input_width, output_height, output_width;
  std::vector<float> weights, biases;
  std::vector<float> weight_gradients, bias_gradients;
};


class CNN {
public:
  CNN(const std::vector<Layer*>& layers) : layers(layers) {}

  ~CNN() {
    for (auto layer : layers) {
      delete layer;
    }
  }

  std::vector<float> forward(const std::vector<float>& input) {
    std::vector<float> current = input;
    for (auto layer : layers) {
      std::vector<float> next;
      layer->forward(current, next);
      current = std::move(next);
    }
    return current;
  }

  void backward(const std::vector<float>& input, const std::vector<float>& target) {
    std::vector<std::vector<float>> layer_inputs(layers.size() + 1);
    layer_inputs[0] = input;

    // Forward pass
    for (size_t i = 0; i < layers.size(); ++i) {
      layers[i]->forward(layer_inputs[i], layer_inputs[i + 1]);
    }

    // Compute output gradient
    std::vector<float> output_gradient(layer_inputs.back().size());
    for (size_t i = 0; i < output_gradient.size(); ++i) {
      output_gradient[i] = layer_inputs.back()[i] - target[i];
    }

    // Backward pass
    std::vector<float> current_gradient = output_gradient;
    for (int i = layers.size() - 1; i >= 0; --i) {
      std::vector<float> prev_gradient;
      layers[i]->backward(layer_inputs[i], layer_inputs[i + 1], current_gradient, prev_gradient);
      current_gradient = std::move(prev_gradient);
    }
  }

  void update(float learning_rate) {
    for (auto layer : layers) {
      layer->update(learning_rate);
    }
  }

  void train(const std::vector<std::vector<float>>& images, const std::vector<int>& labels,
    int epochs, int batch_size, float learning_rate) {
    std::vector<int> indices(images.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());

    for (int epoch = 0; epoch < epochs; ++epoch) {
      std::shuffle(indices.begin(), indices.end(), g);

      float total_loss = 0.0f;
      int correct_predictions = 0;

      for (int i = 0; i < static_cast<int>(images.size()); i += batch_size) {
        int batch_end = std::min(static_cast<int>(images.size()), i + batch_size);

        for (int j = i; j < batch_end; ++j) {
          int idx = indices[j];
          const auto& image = images[idx];
          std::vector<float> target(10, 0.0f);
          target[labels[idx]] = 1.0f;

          auto output = forward(image);
          backward(image, target);
          update(learning_rate);

          // Compute loss and accuracy
          float loss = 0.0f;
          int predicted_class = 0;
          float max_output = output[0];
          for (size_t k = 0; k < output.size(); ++k) {
            loss += std::pow(output[k] - target[k], 2);
            if (output[k] > max_output) {
              max_output = output[k];
              predicted_class = k;
            }
          }
          total_loss += loss;
          if (predicted_class == labels[idx]) {
            ++correct_predictions;
          }
        }
      }

      float avg_loss = total_loss / images.size();
      float accuracy = static_cast<float>(correct_predictions) / images.size();
      std::cout << "Epoch " << epoch + 1 << "/" << epochs
        << ", Loss: " << avg_loss
        << ", Accuracy: " << accuracy << std::endl;
    }
  }

  float evaluate(const std::vector<std::vector<float>>& images, const std::vector<int>& labels) {
    int correct_predictions = 0;

    for (size_t i = 0; i < images.size(); ++i) {
      auto output = forward(images[i]);
      int predicted_class = std::max_element(output.begin(), output.end()) - output.begin();
      if (predicted_class == labels[i]) {
        ++correct_predictions;
      }
    }

    return static_cast<float>(correct_predictions) / images.size();
  }

private:
  std::vector<Layer*> layers;
};