

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

  // Initialize CNN
  CNN cnn({
        new ConvLayer(1, 16, 3, 1, 28, 28),  // Input: 28x28x1, Output: 26x26x16
        new ReLULayer(),
        new MaxPoolLayer(2, 2, 26, 26, 16),  // Output: 13x13x16
        new FCLayer(13 * 13 * 16, 10),
        new SigmoidLayer()
    });

  // Training parameters
  int epochs = 10;
  int batch_size = 32;
  float learning_rate = 0.01;

  // Train the network
  cnn.train(train_images, train_labels, epochs, batch_size, learning_rate);

  // Evaluate on test set
  float accuracy = cnn.evaluate(test_images, test_labels);
  std::cout << "Test accuracy: " << accuracy << std::endl;

  return 0;
}