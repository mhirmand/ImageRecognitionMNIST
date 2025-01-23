

// main.cpp
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include "mnist_loader.h"
#include "cnn.h"


int main() {
  // Load MNIST dataset
  MNISTLoader mnist("C:\\Users\\mohammadreza.hirmand\\Music\\Personal\\Learning\\ComputerVision\\archive");
  auto trainData = mnist.loadTrainingData();
  auto testData = mnist.loadTestData();

  auto train_images = std::get<0>(trainData);
  auto train_labels = std::get<1>(trainData);
  auto test_images = std::get<0>(testData);
  auto test_labels = std::get<1>(testData);

  // Initialize CNN with 5 layers
  CNN cnn({
        new ConvLayer(1, 16, 3, 1, 28, 28),  // Input: 28x28x1, Output: 26x26x16
        new ReLULayer(),
        new MaxPoolLayer(2, 2, 26, 26, 16),  // Output: 13x13x16
        new FCLayer(13 * 13 * 16, 10),
        new SigmoidLayer()
    });

  //// Initialize CNN with 10 layers
  //CNN cnn({
  //  new ConvLayer(1, 32, 3, 1, 28, 28),  // Input: 28x28x1, Output: 26x26x32
  //  new ReLULayer(),
  //  new MaxPoolLayer(2, 2, 26, 26, 32),  // Output: 13x13x32
  //  new ConvLayer(32, 64, 3, 1, 13, 13), // Output: 11x11x64
  //  new ReLULayer(),
  //  new MaxPoolLayer(2, 2, 11, 11, 64),  // Output: 5x5x64
  //  new FCLayer(5 * 5 * 64, 128),
  //  new ReLULayer(),
  //  new FCLayer(128, 10),
  //  new SigmoidLayer()
  //  });

  // Training parameters
  int epochs = 5;
  int batch_size = 1000;
  float learning_rate = 0.01;

  // Train the network
  cnn.train(train_images, train_labels, epochs, batch_size, learning_rate);

  // Evaluate on test set
  std::vector<int> correct_indices;
  correct_indices.reserve(test_labels.size());
  std::vector<int> incorrect_indices;
  incorrect_indices.reserve(test_labels.size());

  float accuracy = cnn.evaluate(test_images, test_labels, correct_indices, incorrect_indices);
  std::cout << "Test accuracy: " << accuracy * 100 << "%" << std::endl;

  // cnn.display_random_test_images(test_images, test_labels, correct_indices, 16);
  // cnn.display_random_test_images(test_images, test_labels, incorrect_indices, 16);

  return 0;
}