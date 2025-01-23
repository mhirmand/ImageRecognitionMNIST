// mnist_loader.cpp
// This file handles the loading of the MNIST dataset in IDX format.

#include "mnist_loader.h"
#include <fstream>
#include <stdexcept>

/**
 * MNISTLoader: Class for loading MNIST dataset.
 * - data_path: Base directory containing the MNIST files.
 * - Throws invalid_argument if the path is empty.
 */
MNISTLoader::MNISTLoader(const std::string& data_path) : data_path(data_path) {
  if (data_path.empty()) {
    throw std::invalid_argument("Dataset path cannot be empty");
  }
}

/**
 * swapEndian: Utility function for converting 32-bit integers from big-endian to host byte order.
 */
uint32_t MNISTLoader::swapEndian(uint32_t val) {
  val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
  return (val << 16) | (val >> 16);
}

/**
 * loadTrainingData: Loads training images and labels.
 * - Reads files "train-images.idx3-ubyte" and "train-labels.idx1-ubyte".
 * - Returns tuple of images (as normalized floats) and labels.
 */
std::tuple<std::vector<std::vector<float>>, std::vector<int>> MNISTLoader::loadTrainingData() {
  return loadData("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
}

/**
 * loadTestData: Loads test images and labels.
 * - Reads files "t10k-images.idx3-ubyte" and "t10k-labels.idx1-ubyte".
 * - Returns tuple of images (as normalized floats) and labels.
 */
std::tuple<std::vector<std::vector<float>>, std::vector<int>> MNISTLoader::loadTestData() {
  return loadData("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
}

/**
 * loadData: Helper function for loading image and label files.
 * - Validates that the number of images matches the number of labels.
 * - Throws runtime_error for mismatches or invalid files.
 */
std::tuple<std::vector<std::vector<float>>, std::vector<int>> MNISTLoader::loadData(
  const std::string& images_file, const std::string& labels_file) {
  auto images = readImageFile(data_path + "\\" + images_file);
  auto labels = readLabelFile(data_path + "\\" + labels_file);

  if (images.size() != labels.size()) {
    throw std::runtime_error("Number of images and labels don't match");
  }

  return { images, labels };
}

/**
 * readImageFile: Reads and processes an MNIST image file.
 * - Verifies the magic number and extracts image dimensions.
 * - Normalizes pixel values to the range [0, 1].
 * - Throws runtime_error if the file cannot be opened or is invalid.
 */
std::vector<std::vector<float>> MNISTLoader::readImageFile(const std::string& file_path) {
  std::ifstream file(file_path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open file: " + file_path);
  }

  uint32_t magic_number, num_images, num_rows, num_cols;
  file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
  file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
  file.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
  file.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));

  // Convert from big-endian to host endian
  magic_number = swapEndian(magic_number);
  num_images = swapEndian(num_images);
  num_rows = swapEndian(num_rows);
  num_cols = swapEndian(num_cols);

  if (magic_number != 2051) {
    throw std::runtime_error("Invalid MNIST image file: " + file_path);
  }

  std::vector<std::vector<float>> images(num_images, std::vector<float>(num_rows * num_cols));
  for (uint32_t i = 0; i < num_images; ++i) {
    for (uint32_t j = 0; j < num_rows * num_cols; ++j) {
      unsigned char pixel;
      file.read(reinterpret_cast<char*>(&pixel), 1);
      images[i][j] = static_cast<float>(pixel) / 255.0f;
    }
  }

  return images;
}

/**
 * readLabelFile: Reads and processes an MNIST label file.
 * - Verifies the magic number and extracts labels.
 * - Throws runtime_error if the file cannot be opened or is invalid.
 */
std::vector<int> MNISTLoader::readLabelFile(const std::string& file_path) {
  std::ifstream file(file_path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open file: " + file_path);
  }

  uint32_t magic_number, num_items;
  file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
  file.read(reinterpret_cast<char*>(&num_items), sizeof(num_items));

  // Convert from big-endian to host endian
  magic_number = swapEndian(magic_number);
  num_items = swapEndian(num_items);

  if (magic_number != 2049) {
    throw std::runtime_error("Invalid MNIST label file: " + file_path);
  }

  std::vector<int> labels(num_items);
  for (uint32_t i = 0; i < num_items; ++i) {
    unsigned char label;
    file.read(reinterpret_cast<char*>(&label), 1);
    labels[i] = static_cast<int>(label);
  }

  return labels;
}