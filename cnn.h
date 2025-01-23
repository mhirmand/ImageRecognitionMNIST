#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>
#include <numeric> // For std::iota
#include <iomanip>
#include <chrono>
// #include <opencv2/opencv.hpp>

class MaxPoolLayer : public Layer {
public:
  MaxPoolLayer(int kernel_size, int stride, int input_height, int input_width, int channels)
    : kernel_size(kernel_size), stride(stride), input_height(input_height), input_width(input_width), channels(channels) {
    output_height = (input_height - kernel_size) / stride + 1;
    output_width = (input_width - kernel_size) / stride + 1;
  }

  void forward(const std::vector<float>& input, std::vector<float>& output) override {
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

  void backward(const std::vector<float>& input, const std::vector<float>& output,
    const std::vector<float>& output_gradient, std::vector<float>& input_gradient) override {
    input_gradient.resize(input.size(), 0);

    for (size_t i = 0; i < output_gradient.size(); ++i) {
      input_gradient[max_indices[i]] += output_gradient[i];
    }
  }

  void update(float learning_rate) override {
    // MaxPool has no parameters to update
  }

private:
  int kernel_size, stride;
  int input_height, input_width, output_height, output_width, channels;
  std::vector<int> max_indices;
};

class FCLayer : public Layer {
public:
  FCLayer(int input_size, int output_size)
    : input_size(input_size), output_size(output_size) {
    weights.resize(input_size * output_size);
    biases.resize(output_size);

    // Xavier initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, std::sqrt(2.0 / (input_size + output_size)));
    for (auto& w : weights) w = d(gen);
    std::fill(biases.begin(), biases.end(), 0.0f);

    weight_gradients.resize(weights.size(), 0.0f);
    bias_gradients.resize(biases.size(), 0.0f);
  }

  void forward(const std::vector<float>& input, std::vector<float>& output) override {
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

  void update(float learning_rate) override {
    for (size_t i = 0; i < weights.size(); ++i) {
      weights[i] -= learning_rate * weight_gradients[i];
    }
    for (size_t i = 0; i < biases.size(); ++i) {
      biases[i] -= learning_rate * bias_gradients[i];
    }
  }

private:
  int input_size, output_size;
  std::vector<float> weights, biases;
  std::vector<float> weight_gradients, bias_gradients;
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

  void update(float learning_rate) override {
    // Sigmoid has no parameters to update
  }
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

    std::cout << "Starting training process..." << std::endl;
    std::cout << "Total images: " << images.size() << std::endl;
    std::cout << "Epochs: " << epochs << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "Learning rate: " << learning_rate << std::endl;
    std::cout << std::fixed << std::setprecision(4);

    auto total_start_time = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < epochs; ++epoch) {
      std::shuffle(indices.begin(), indices.end(), g);

      float total_loss = 0.0f;
      int correct_predictions = 0;
      int total_batches = (images.size() + batch_size - 1) / batch_size;

      auto epoch_start_time = std::chrono::high_resolution_clock::now();

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
            loss += std::pow(output[k] - target[k], 2);
            if (output[k] > max_output) {
              max_output = output[k];
              predicted_class = k;
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
          float batch_accuracy = static_cast<float>(batch_correct) / current_batch_size;
          std::cout << "Epoch " << epoch + 1 << "/" << epochs
            << ", Batch " << batch + 1 << "/" << total_batches
            << ", Loss: " << batch_loss / current_batch_size
            << ", Accuracy: " << batch_accuracy * 100 << "%" << std::endl;
        }
      }

      auto epoch_end_time = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> epoch_duration = epoch_end_time - epoch_start_time;

      float avg_loss = total_loss / images.size();
      float accuracy = static_cast<float>(correct_predictions) / images.size();

      std::cout << "\nEpoch " << epoch + 1 << "/" << epochs << " completed in "
        << epoch_duration.count() << " seconds" << std::endl;
      std::cout << "Average Loss: " << avg_loss << std::endl;
      std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;
      std::cout << std::string(50, '-') << std::endl;
    }

    auto total_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = total_end_time - total_start_time;
    std::cout << "\nTraining completed in " << total_duration.count() << " seconds" << std::endl;
  }

  float evaluate(const std::vector<std::vector<float>>& images, const std::vector<int>& labels,
    std::vector<int> & correct_indices, std::vector<int>& incorrect_indices) {
    int correct_predictions = 0;

    // Predict all images and categorize them
    for (size_t i = 0; i < images.size(); ++i) {
      int predicted_label = predict(images[i]);
      if (predicted_label == labels[i]) {
        correct_indices.push_back(i);
        correct_predictions++;
      }
      else {
        incorrect_indices.push_back(i);
      }
    }

    return static_cast<float>(correct_predictions) / images.size();
  }

  int predict(const std::vector<float>& image) {
    auto output = forward(image);
    return std::max_element(output.begin(), output.end()) - output.begin();
  }

  //void display_random_test_images(const std::vector<std::vector<float>>& images,
  //  const std::vector<int>& labels,
  //  const std::vector<int>& indices,
  //  int num_images) {

  //  std::vector<int> newindices(indices);

  //  std::random_device rd;
  //  std::mt19937 gen(rd());
  //  std::uniform_int_distribution<> dis(0, newindices.size() - 1);

  //  // Define the grid size
  //  int grid_rows = std::ceil(std::sqrt(num_images));
  //  int grid_cols = std::ceil(static_cast<float>(num_images) / grid_rows);

  //  // std::random_device rd;
  //  // std::mt19937 g(rd());
  //  std::shuffle(newindices.begin(), newindices.end(), gen);

  //  // Limit the number of images to display
  //  // indices.resize(std::min(static_cast<size_t>(num_images), indices.size()));

  //  // Create a large image to hold all the small images
  //  int img_size = 28 * 5; // 28 * 5 because we're scaling up each MNIST image by 5
  //  int padding = 10; // Padding between images
  //  cv::Mat grid_image(grid_rows * (img_size + padding) + padding,
  //    grid_cols * (img_size + padding) + padding,
  //    CV_8UC3, cv::Scalar(255, 255, 255));

  //  for (int i = 0; i < num_images; ++i) {
  //    int idx = newindices[dis(gen)];
  //    const auto& image = images[idx];
  //    int true_label = labels[idx];
  //    int predicted_label = predict(image);

  //    // Convert the 1D vector to a 2D OpenCV Mat
  //    cv::Mat display_image(28, 28, CV_32F, const_cast<float*>(image.data()));

  //    // Normalize the image for display
  //    cv::normalize(display_image, display_image, 0, 255, cv::NORM_MINMAX);
  //    display_image.convertTo(display_image, CV_8U);

  //    // Resize the image for better visibility
  //    cv::resize(display_image, display_image, cv::Size(img_size, img_size), 0, 0, cv::INTER_NEAREST);

  //    // Convert to 3-channel image
  //    cv::cvtColor(display_image, display_image, cv::COLOR_GRAY2BGR);

  //    // Add text to the image
  //    cv::putText(display_image,
  //      "True: " + std::to_string(true_label),
  //      cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5,
  //      predicted_label == true_label ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255),
  //      1);
  //    cv::putText(display_image,
  //      "Pred: " + std::to_string(predicted_label),
  //      cv::Point(5, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5,
  //      predicted_label == true_label ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255),
  //      1);

  //    // Calculate position in the grid
  //    int row = i / grid_cols;
  //    int col = i % grid_cols;

  //    // Copy the image to the correct position in the grid
  //    display_image.copyTo(grid_image(cv::Rect(col * (img_size + padding) + padding,
  //      row * (img_size + padding) + padding,
  //      img_size, img_size)));
  //  }

  //  // Display the grid image
  //  cv::imshow("Test Images with Predictions", grid_image);
  //  // cv::waitKey(0);
  //  cv::destroyAllWindows();
  //}

private:
  std::vector<Layer*> layers;
};