/**
 * OKVIS2-X - Open Keyframe-based Visual-Inertial SLAM Configurable with Dense 
 * Depth or LiDAR, and GNSS
 *
 * Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 * Copyright (c) 2020, Smart Robotics Lab / Imperial College London
 * Copyright (c) 2025, Mobile Robotics Lab / Technical University of Munich 
 * and ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */

#ifndef OKVIS_VISION_LANGUAGE_PROCESSOR_HPP
#define OKVIS_VISION_LANGUAGE_PROCESSOR_HPP

#include <map>
#include <thread>
#include <atomic>
#include <iostream>
#include <mutex>

#include <opencv2/calib3d.hpp>
#include <opencv2/dnn/dnn.hpp>

#include <torch/torch.h>
#include <torch/script.h>

#include <okvis/Measurements.hpp>
#include <okvis/threadsafe/ThreadsafeQueue.hpp>
#include <okvis/QueuedTrajectory.hpp>
#include <okvis/ThreadedSlam.hpp>
#include <okvis/ViSensorBase.hpp>
#include <okvis/DeepLearningProcessor.hpp>

namespace okvis {

class VLProcessor : public DeepLearningProcessor {

struct VisualizationData {
    cv::Mat rgbFrame;
    cv::Mat depthImage;
    torch::Tensor featureMap;
    torch::Tensor samMasks;
};

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

  /// \brief Callback for receiving depth measurements
  typedef std::function<bool(const std::map<size_t, std::vector<okvis::CameraMeasurement>> &)> ImageCallback;

  VLProcessor(okvis::ViParameters &parameters,
                        std::string modelDir);
  virtual ~VLProcessor();

  void setLanguageEmbeddingVector(const torch::Tensor& embedding);
  void setLanguageEmbedding(const Descriptor& embedding);

  /// @brief Display rgb image, depth image and language features
  virtual void display(std::map<std::string, cv::Mat> &images) override;

  /// \name Add measurements to the algorithm.
  /**
   * \brief              Add a set of new image.
   * \param stamp        The image timestamp.
   * \param images       The images.
   * \return             Returns true normally. False, if the previous one has not been processed yet.
   */
  virtual bool addImages(const std::map<size_t, std::pair<okvis::Time, cv::Mat>> & images,
                         const std::map<size_t, std::pair<okvis::Time, cv::Mat>> & depthImages) override final;

  /// @brief Check whether the processor is finished.
  bool finishedProcessing() override;

private:

  /// @brief Processing loops and according threads.
  void processing();

  void segmentPostProcessing(std::shared_ptr<torch::Tensor> bboxes, std::vector<int>& filteredIndexes);

  /// @brief Function where the actual neural network predicts the depth from the stereo images
  /// @param frame The stereo images measurement which will then be processed by the neural network
  void processLanguageNetwork(std::map<size_t, std::vector<okvis::CameraMeasurement>>& frame);

  torch::jit::script::Module languageModel_;
  torch::jit::script::Module samModel_;

  size_t rgbCamIndex_;
  torch::Tensor textEmbedding_;
  std::mutex textEmbeddingMutex_;
  torch::jit::script::Module visionEncoder_;
  okvis::threadsafe::Queue<VisualizationData> visualisationsQueue_;


  // Save Dataset when processing
  bool saveDataset_ = false;
  std::string datasetPath_ = "/home/wiss/boche/projects/feature_thing_v2/test_dataset/";
  std::ofstream depthFileList_;
  std::ofstream rgbFileList_;
  std::ofstream samFileList_;
};

} // namespace srl

#endif //STEREO_DEPTH_STEREO2DEPTH_PROCESSOR_HPP
