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

  void train(
    std::ofstream& log_file,
    const std::vector<std::vector<float>>& images,
    const std::vector<int>& labels,
    const std::vector<std::vector<float>>& test_images,
    const std::vector<int>& test_labels,
    int epochs,
    int batch_size,
    float learning_rate,
    int seed);

  std::tuple<float, float> evaluate(const std::vector<std::vector<float>>& images,
    const std::vector<int>& labels);

  int predict(const std::vector<float>& image);

private:
  std::vector<std::unique_ptr<Layer>> layers;

  std::vector<float> forward(const std::vector<float>& input);
  void backward(const std::vector<float>& input, const std::vector<float>& target);
  void update(float learning_rate);
};

// Factory function
std::unique_ptr<CNN> create_default_cnn(int seed);