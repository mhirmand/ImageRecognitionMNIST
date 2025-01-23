#include <iostream>
#include <vector>
#include <string>
#include "cnn.h"
#include "mnist_loader.h"

struct Config {
  std::string data_path;
  int epochs = 5;
  int batch_size = 100;
  float learning_rate = 0.01f;
};

Config parse_arguments(int argc, char* argv[]) {
  Config cfg;
  if (argc < 2) {
    throw std::runtime_error("Usage: " + std::string(argv[0]) + " <dataset_path> [epochs] [batch_size] [learning_rate]");
  }
  cfg.data_path = argv[1];
  if (argc > 2) cfg.epochs = std::stoi(argv[2]);
  if (argc > 3) cfg.batch_size = std::stoi(argv[3]);
  if (argc > 4) cfg.learning_rate = std::stof(argv[4]);
  return cfg;
}

int main(int argc, char* argv[]) {
  try {
    Config cfg = parse_arguments(argc, argv);

    MNISTLoader mnist(cfg.data_path);
    auto [train_images, train_labels] = mnist.loadTrainingData();
    auto [test_images, test_labels] = mnist.loadTestData();

    auto cnn = create_default_cnn();
    cnn->train(train_images, train_labels, cfg.epochs, cfg.batch_size, cfg.learning_rate);

    std::vector<int> correct, incorrect;
    float accuracy = cnn->evaluate(test_images, test_labels, correct, incorrect);
    std::cout << "Test accuracy: " << accuracy * 100 << "%\n";

  }
  catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}