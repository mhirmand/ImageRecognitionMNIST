#pragma once
#include <memory>
#include <vector>
#include "layers/layer.h"

/**
 * @brief Convolutional Neural Network class
 * Manages layers and training process
 */
class CNN {
public:
  explicit CNN(std::vector<std::unique_ptr<Layer>> layers);

  // Disallow copying
  CNN(const CNN&) = delete;
  CNN& operator=(const CNN&) = delete;

  void train(const std::vector<std::vector<float>>& images,
    const std::vector<int>& labels,
    int epochs = 5,
    int batch_size = 1000,
    float learning_rate = 0.01f);

  float evaluate(const std::vector<std::vector<float>>& images,
    const std::vector<int>& labels,
    std::vector<int>& correct_indices,
    std::vector<int>& incorrect_indices);

  int predict(const std::vector<float>& image);

private:
  std::vector<std::unique_ptr<Layer>> layers;

  std::vector<float> forward(const std::vector<float>& input);
  void backward(const std::vector<float>& input, const std::vector<float>& target);
  void update(float learning_rate);
};

// Factory function
std::unique_ptr<CNN> create_default_cnn();