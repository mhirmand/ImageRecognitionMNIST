#pragma once
#include <vector>
#include <string>
#include <tuple>

/**
 * @brief MNIST dataset loader
 * Handles loading of MNIST image and label files
 */
class MNISTLoader {
public:
  explicit MNISTLoader(const std::string& data_path);

  std::tuple<std::vector<std::vector<float>>, std::vector<int>> loadTrainingData();
  std::tuple<std::vector<std::vector<float>>, std::vector<int>> loadTestData();

private:
  std::string data_path;

  uint32_t swapEndian(uint32_t val);
  std::tuple<std::vector<std::vector<float>>, std::vector<int>> loadData(
    const std::string& images_file, const std::string& labels_file);
  std::vector<std::vector<float>> readImageFile(const std::string& file_path);
  std::vector<int> readLabelFile(const std::string& file_path);
};