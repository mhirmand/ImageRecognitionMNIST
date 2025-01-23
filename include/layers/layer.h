#pragma once
#include <vector>

/**
 * @brief Abstract base class for all neural network layers
 * Defines interface for forward/backward propagation and parameter updates
 */
class Layer {
public:
  virtual ~Layer() = default;

  /**
   * @brief Perform forward propagation
   * @param input Input tensor (flattened)
   * @param output Output tensor (flattened)
   */
  virtual void forward(const std::vector<float>& input, std::vector<float>& output) = 0;

  /**
   * @brief Perform backward propagation
   * @param input Original input tensor
   * @param output Original output tensor
   * @param output_gradient Gradient from subsequent layer
   * @param input_gradient Computed gradient for previous layer
   */
  virtual void backward(const std::vector<float>& input,
    const std::vector<float>& output,
    const std::vector<float>& output_gradient,
    std::vector<float>& input_gradient) = 0;

  /**
   * @brief Update layer parameters
   * @param learning_rate Learning rate for optimization
   */
  virtual void update(float learning_rate) = 0;
};