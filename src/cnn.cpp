// cnn.cpp
// This file implements the core components of a Convolutional Neural Network (CNN).

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
#include <iomanip>
#include <fstream>
#include <string>
#include <tuple>

/**
 * Convolutional Neural Network (CNN) Overview:
 * The CNN architecture defined in the code is designed for the MNIST digit classification task.
 *
 * 1. Input: The input images are 28x28 grayscale images (flattened into 784-element vectors).
 * 2. Default Architecture with 5 layers: ConvLayer -> ReLULayer -> MaxPoolLayer -> FCLayer -> SigmoidLayer
 *    - **ConvLayer**:
 *      - 1 input channel (grayscale images).
 *      - 16 output channels with a kernel size of 3x3 and stride 1.
 *      - Produces feature maps of size 26x26.
 *    - **ReLULayer**: Applies the ReLU activation function to introduce non-linearity.
 *    - **MaxPoolLayer**:
 *      - Reduces the spatial size of feature maps to 13x13 using a pooling window of 2x2 and stride 2.
 *    - **FCLayer**:
 *      - Fully connected layer with 16x13x13=2704 inputs and 10 outputs (one for each digit class).
 *    - **SigmoidLayer**: Applies the sigmoid activation function to produce normalized outputs.
 * 3. Training Process:
 *    - Optimizes the network using Mean Squared Error (MSE) loss.
 *    - Parameters (weights and biases) are updated using gradient descent with the specified learning rate.
 * 4. Output: The network outputs probabilities for each of the 10 digit classes (0-9).
 */

std::unique_ptr<CNN> create_default_cnn() {
  std::vector<std::unique_ptr<Layer>> layers;
  layers.push_back(std::make_unique<ConvLayer>(1, 16, 3, 1, 28, 28));
  layers.push_back(std::make_unique<ReLULayer>());
  layers.push_back(std::make_unique<MaxPoolLayer>(2, 2, 26, 26, 16));
  layers.push_back(std::make_unique<FCLayer>(13 * 13 * 16, 10));
  layers.push_back(std::make_unique<SigmoidLayer>());
  return std::make_unique<CNN>(std::move(layers));
}

/**
 * CNN: Represents the entire Convolutional Neural Network.
 * Features:
 * - Flexible architecture using various layer types.
 * - Forward pass to compute predictions.
 * - Backward pass for gradient computation and backpropagation.
 * - Training functionality with batching and shuffling of data.
 * - Evaluation to measure accuracy on test datasets.
 */
CNN::CNN(std::vector<std::unique_ptr<Layer>> layers) : layers(std::move(layers)) {}

std::vector<float> CNN::forward(const std::vector<float>& input) {
  std::vector<float> current = input;
  for (auto& layer : layers) {
    std::vector<float> next;
    layer->forward(current, next);
    current = std::move(next);
  }
  return current;
}

void CNN::backward(const std::vector<float>& input, const std::vector<float>& target) {
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

void CNN::update(float learning_rate) {
  for (auto& layer : layers) {
    layer->update(learning_rate);
  }
}

// train returns a tuple of training and test accuracies
void CNN::train(
  std::ofstream& log_file,
  const std::vector<std::vector<float>>& images,
  const std::vector<int>& labels,
  const std::vector<std::vector<float>>& test_images,
  const std::vector<int>& test_labels,
  int epochs, int batch_size, float learning_rate) {
  std::vector<int> indices(images.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::random_device rd;
  std::mt19937 g(rd());

  // Write initial training info to log file
  log_file << "Starting training process..." << std::endl;
  log_file << "Total training images: " << images.size() << std::endl;
  log_file << "Total testing images: " << test_images.size() << std::endl;
  log_file << "Epochs: " << epochs << std::endl;
  log_file << "Batch size: " << batch_size << std::endl;
  log_file << "Learning rate: " << learning_rate << "\n" << std::endl;
  log_file << std::string(65, '*') << "\n" << std::endl;

  // Write headers
  log_file << std::setw(10) << "Epoch" << "  "
    << std::setw(10) << "Batch" << "  "
    << std::setw(15) << "Batch Loss" << "  "
    << std::setw(15) << "Test Loss" << "   "
    << std::setw(13) << "Batch Acc" << "  "
    << std::setw(13) << "Test Acc" << std::endl;
  log_file << std::string(90, '-') << std::endl;

  // Write initial training info to console window
  std::cout << "\nStarting training process..." << std::endl;
  std::cout << "Total training images: " << images.size() << std::endl;
  std::cout << "Total testing images: " << images.size() << std::endl;
  std::cout << "Epochs: " << epochs << std::endl;
  std::cout << "Batch size: " << batch_size << std::endl;
  std::cout << "Learning rate: " << learning_rate << "\n" << std::endl;
  std::cout << std::string(65, '*') << std::endl;
  
  auto total_start_time = std::chrono::high_resolution_clock::now();
  float test_accuracy = 0.0f, test_loss = 0.0f;
  float train_accuracy = 0.0f;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    std::shuffle(indices.begin(), indices.end(), g);

    float total_loss = 0.0f;
    int correct_predictions = 0;
    int total_batches = (int)((images.size() + batch_size - 1) / batch_size);

    auto epoch_start_time = std::chrono::high_resolution_clock::now();

    float batch_accuracy = 0.0f;
    for (int batch = 0; batch < total_batches; ++batch) {
      int start_idx = batch * batch_size;
      int end_idx = std::min(static_cast<int>(images.size()), (batch + 1) * batch_size);
      int current_batch_size = end_idx - start_idx;

      float batch_loss = 0.0f;
      int batch_correct = 0;

      for (int j = start_idx; j < end_idx; ++j) {
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
          loss += (float)(std::pow(output[k] - target[k], 2));
          if (output[k] > max_output) {
            max_output = output[k];
            predicted_class = (int)k;
          }
        }
        batch_loss += loss;
        if (predicted_class == labels[idx]) {
          ++batch_correct;
        }
      }

      total_loss += batch_loss;
      correct_predictions += batch_correct;

      // Print batch progress
      if ((batch + 1) % 10 == 0 || batch == total_batches - 1) {
        batch_accuracy = static_cast<float>(batch_correct) / current_batch_size;
        std::cout << "Epoch " << epoch + 1 << "/" << epochs
          << ", Batch " << batch + 1 << "/" << total_batches
          << ", Batch Loss: " << std::fixed << std::setprecision(4) << batch_loss / current_batch_size
          << ", Batch Acc: " << std::fixed << std::setprecision(2) << batch_accuracy * 100 << "%" << std::endl;
        
        auto test_result = evaluate(test_images, test_labels);
        test_loss = std::get<0>(test_result);
        test_accuracy = std::get<1>(test_result);
        log_file << std::setw(10) << epoch + 1 << "  "
          << std::setw(10) << batch + 1 << "  "
          << std::setw(15) << std::fixed << std::setprecision(6) << batch_loss / current_batch_size << "  "
          << std::setw(15) << std::fixed << std::setprecision(6) << test_loss << "  "
          << std::setw(13) << std::fixed << std::setprecision(2) << batch_accuracy * 100 << "%  "
          << std::setw(13) << std::fixed << std::setprecision(2) << test_accuracy * 100.0 << "%"
          << std::endl;
      }
    }

    auto epoch_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> epoch_duration = epoch_end_time - epoch_start_time;

    float avg_loss = total_loss / images.size();
    train_accuracy = static_cast<float>(correct_predictions) / images.size();

    auto test_result = evaluate(test_images, test_labels);
		test_loss = std::get<0>(test_result);
    test_accuracy = std::get<1>(test_result);

    std::cout << "\nEpoch " << epoch + 1 << "/" << epochs << " completed in "
      << epoch_duration.count() << " seconds" << std::endl;
    std::cout << "Train Loss: " << std::fixed << std::setprecision(4) << avg_loss << std::endl;
    std::cout << "Test Loss: " << std::fixed << std::setprecision(4) << test_loss << std::endl;
    std::cout << "Train Accuracy: " << std::fixed << std::setprecision(2) << train_accuracy * 100 << "%" << std::endl;
    std::cout << "Test Accuracy: " << std::fixed << std::setprecision(2) << test_accuracy * 100 << "%" << std::endl;
    std::cout << std::string(65, '-') << std::endl;
  }

  auto total_end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> total_duration = total_end_time - total_start_time;

  // final loggings
  std::cout << "\nTraining completed in " << total_duration.count() << " seconds with:" << std::endl;
  std::cout << "   Train accuracy: " << std::fixed << std::setprecision(2) << train_accuracy * 100.0f << "%" << std::endl;
  std::cout << "   Test accuracy: " << std::fixed << std::setprecision(2) << test_accuracy * 100.0f << "%\n" << std::endl;

  log_file << std::string(90, '-') << std::endl;
  log_file << "\nTraining completed in " << total_duration.count() << " seconds with:" << std::endl;
  log_file << "   Train accuracy: " << train_accuracy * 100.0f << "%" << std::endl;
  log_file << "   Test accuracy: " << test_accuracy * 100.0f << "%\n" << std::endl;

  return;
}

std::tuple<float, float> CNN::evaluate(const std::vector<std::vector<float>>& images, const std::vector<int>& labels) {
  int correct_predictions = 0;
  float loss = 0.0f;

  // Predict all images and categorize them
  for (int i = 0; i < images.size(); ++i) {
    auto prediction = forward(images[i]);
    std::vector<float> target(10, 0.0f);
    target[labels[i]] = 1.0f;
    int predicted_label = static_cast<int>(std::max_element(prediction.begin(), prediction.end()) - prediction.begin());
    // int predicted_label = predict(images[i]);
    for (size_t j = 0; j < prediction.size(); ++j) {
      loss += (float)(std::pow(prediction[j] - target[j], 2));
		}
    if (predicted_label == labels[i]) {
      correct_predictions++;
    }
  }

  return { loss / images.size() , static_cast<float>(correct_predictions) / images.size() };
}


int CNN::predict(const std::vector<float>& image) {
  auto output = forward(image);
  return (int)(std::max_element(output.begin(), output.end()) - output.begin());
}

/**
 * ConvLayer: Implements a 2D convolutional layer.
 * Key functionalities:
 * - initialize_parameters: Initialization of weights and biases with random values.
 * - Forward pass: Performs convolutional operations.
 * - Backward pass: Computes gradients for weights, biases, and input.
 * - Parameter update: Adjusts weights and biases using gradients.
 */
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
  for (auto& w : weights) w = (float)d(gen);

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



/**
 * MaxPoolLayer: Implements a 2D max-pooling layer.
 * Key functionalities:
 * - Forward pass: Applies max-pooling to reduce spatial dimensions.
 * - Backward pass: Propagates gradients to input.
 * - No parameters to update.
 */
MaxPoolLayer::MaxPoolLayer(int kernel_size, int stride,
  int input_height, int input_width, int channels)
  : kernel_size(kernel_size), stride(stride),
  input_height(input_height), input_width(input_width), channels(channels) {

  output_height = (input_height - kernel_size) / stride + 1;
  output_width = (input_width - kernel_size) / stride + 1;
}

void MaxPoolLayer::forward(const std::vector<float>& input, std::vector<float>& output) {
  output.resize(channels * output_height * output_width);
  max_indices.resize(output.size());

  for (int c = 0; c < channels; ++c) {
    for (int oh = 0; oh < output_height; ++oh) {
      for (int ow = 0; ow < output_width; ++ow) {
        int out_idx = (c * output_height + oh) * output_width + ow;
        float max_val = -std::numeric_limits<float>::max();
        int max_idx = -1;

        for (int kh = 0; kh < kernel_size; ++kh) {
          for (int kw = 0; kw < kernel_size; ++kw) {
            int ih = oh * stride + kh;
            int iw = ow * stride + kw;
            int in_idx = (c * input_height + ih) * input_width + iw;

            if (input[in_idx] > max_val) {
              max_val = input[in_idx];
              max_idx = in_idx;
            }
          }
        }

        output[out_idx] = max_val;
        max_indices[out_idx] = max_idx;
      }
    }
  }
}

void MaxPoolLayer::backward(const std::vector<float>& input, const std::vector<float>& output,
  const std::vector<float>& output_gradient, std::vector<float>& input_gradient) {
  input_gradient.resize(input.size(), 0);

  for (size_t i = 0; i < output_gradient.size(); ++i) {
    input_gradient[max_indices[i]] += output_gradient[i];
  }
}

void MaxPoolLayer::update(float learning_rate) {
  // MaxPool has no parameters to update
}


/**
 * FCLayer: Fully connected layer.
 * Key functionalities:
 * - Initialization of weights and biases with random values.
 * - Forward pass: Computes linear transformation.
 * - Backward pass: Calculates gradients for weights, biases, and input.
 * - Parameter update: Adjusts weights and biases using gradients.
 */
FCLayer::FCLayer(int input_size, int output_size)
  : input_size(input_size), output_size(output_size) {
  initialize_parameters();
}

void FCLayer::initialize_parameters() {
  weights.resize(input_size * output_size);
  biases.resize(output_size, 0.0f);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> d(0, std::sqrt(2.0 / (input_size + output_size)));
  for (auto& w : weights) w = (float)d(gen);

  weight_gradients.resize(weights.size(), 0.0f);
  bias_gradients.resize(biases.size(), 0.0f);
}

void FCLayer::forward(const std::vector<float>& input, std::vector<float>& output) {
  output.resize(output_size);
  for (int i = 0; i < output_size; ++i) {
    output[i] = biases[i];
    for (int j = 0; j < input_size; ++j) {
      output[i] += input[j] * weights[i * input_size + j];
    }
  }
}

void FCLayer::backward(const std::vector<float>& input, const std::vector<float>& output,
  const std::vector<float>& output_gradient, std::vector<float>& input_gradient) {
  input_gradient.resize(input_size, 0.0f);
  std::fill(weight_gradients.begin(), weight_gradients.end(), 0.0f);
  std::fill(bias_gradients.begin(), bias_gradients.end(), 0.0f);

  for (int i = 0; i < output_size; ++i) {
    for (int j = 0; j < input_size; ++j) {
      weight_gradients[i * input_size + j] += input[j] * output_gradient[i];
      input_gradient[j] += weights[i * input_size + j] * output_gradient[i];
    }
    bias_gradients[i] += output_gradient[i];
  }
}

void FCLayer::update(float learning_rate) {
  for (size_t i = 0; i < weights.size(); ++i) {
    weights[i] -= learning_rate * weight_gradients[i];
  }
  for (size_t i = 0; i < biases.size(); ++i) {
    biases[i] -= learning_rate * bias_gradients[i];
  }
}

