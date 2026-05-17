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

#include <filesystem>
#include <okvis/VisionLanguageProcessor.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <okvis/timing/Timer.hpp>
#include <okvis/utils.hpp>
#include <okvis/mapTypedefs.hpp>
#include <random>


namespace okvis {
    size_t averageDescriptor(const okvis::CameraData::DeepLearningImageData& clip_image,
                             const cv::Mat& mask,
                             Descriptor& average_descriptor,
                             int resize_ratio = 1){

        int64_t height = clip_image.shape[0];
        int64_t width = clip_image.shape[1];
        int64_t channels = clip_image.shape[2];
        average_descriptor.setZero();

        size_t num_evaluated_pixels = 0;

        //
        int mask_width = mask.cols;
        int mask_height = mask.rows;

        for(int v = 0; v < mask_height; v++) {
            for(int u = 0; u < mask_width; u++) {
                if(mask.at<uchar>(v, u) == 0) {
                    continue;
                }

                int u_clip = float(u) / resize_ratio;
                int v_clip = float(v) / resize_ratio;

                num_evaluated_pixels++;
                float* pixel_ptr = clip_image.data_ptr.get() + u_clip*768 + v_clip*width*768;
                Eigen::Map<Descriptor> pixel_descriptor(pixel_ptr);
                average_descriptor += pixel_descriptor;
            }
        }

        // Average descriptor over evaluated pixels
        average_descriptor /= num_evaluated_pixels;

        return num_evaluated_pixels;
    }

    VLProcessor::VLProcessor(okvis::ViParameters &parameters,
                             std::string modelDir) {

      //TODOs, iterate over all the cameras and select the first RGB image camera in the parameters and store its index

      bool foundRGBCam = false;
      for(size_t i = 0; i < parameters.nCameraSystem.numCameras(); i++){
        if(parameters.nCameraSystem.cameraType(i).isColour) {
          foundRGBCam = true;
          rgbCamIndex_ = i;
          break;
        }
      }

      if(!foundRGBCam) {
        throw std::runtime_error("Vision language requires an RGB camera camera, but there are non declared in the config file");
      } else {
        LOG(INFO) << "The selected RGB camera for the language processor is camera with index " << rgbCamIndex_;
      }

      torch::Tensor rgbTensor = torch::zeros({parameters.nCameraSystem.cameraGeometry(rgbCamIndex_)->imageHeight(), 
                                              parameters.nCameraSystem.cameraGeometry(rgbCamIndex_)->imageWidth(), 
                                              3}).to(torch::kU8).to(torch::kCUDA);

      try {
        torch::NoGradGuard no_grad;
        samModel_ = torch::jit::load(modelDir + "/esam-model.pt", torch::kCUDA);

        // Warm up forward pass for GPU.
        LOG(INFO) << "Warming up GPU ...";
        // torch::Tensor rgbTensor = torch::zeros({480, 640, 3}).to(torch::kU8).to(torch::kCUDA);
       
        for(int i = 0; i < 10; i++) {
          LOG(INFO) << "GPU warmup iteration " << i;
          samModel_.forward({rgbTensor});
        }
        LOG(INFO) << "Done warming up GPU.";
      }
      catch (const c10::Error& e) {
        LOG(ERROR) << "Error loading the SAM model from esam-model.pt " << e.what();
        return;
      }

      try {
        visionEncoder_ = torch::jit::load(modelDir + "/vl-vision-model.pt", torch::kCUDA);

        // Warm up forward pass for GPU.
        LOG(INFO) << "Warming up GPU ...";
        for(int i = 0; i < 10; i++) {
          LOG(INFO) << "GPU warmup iteration " << i;
          visionEncoder_.forward({rgbTensor});
        }
        LOG(INFO) << "Done warming up GPU.";
      }
      catch (const c10::Error& e) {
        LOG(ERROR) << "Error loading the CLIP model from vl-vision-model.pt " << e.what();
        return;
      }

      textEmbedding_ = torch::from_blob(okvis::chair_embedding.data(), {1, 1, 768},
                                        torch::kFloat32);
      textEmbedding_ /= textEmbedding_.norm();

      processingThread_ = std::thread(&VLProcessor::processing, this);

      // Save Dataset
      if (saveDataset_) {
        std::filesystem::create_directories(datasetPath_);
        std::filesystem::create_directories(datasetPath_ + "/depth/");
        std::filesystem::create_directories(datasetPath_ + "/rgb/");
        std::filesystem::create_directories(datasetPath_ + "/sam/");

        depthFileList_.open(datasetPath_ + "/depth.txt");
        depthFileList_ << "# depth images \n# timestamp filename " << std::endl;
        rgbFileList_.open(datasetPath_ + "/rgb.txt");
        rgbFileList_ << "# rgb images \n# timestamp filename " << std::endl;
        samFileList_.open(datasetPath_ + "/sam.txt");
        samFileList_ << "# segment images \n# timestamp filename " << std::endl;
      }

    }

    VLProcessor::~VLProcessor() {
      cameraMeasurementsQueue_.Shutdown();
      visualisationsQueue_.Shutdown();
      shutdown_ = true;
      processingThread_.join();

      if (saveDataset_) {
        depthFileList_.close();
        rgbFileList_.close();
        samFileList_.close();
      }
    }

    void VLProcessor::display(std::map<std::string, cv::Mat> &images) {
      VisualizationData frame;
      if(!visualisationsQueue_.PopNonBlocking(&frame)) {
        DLOG(INFO) << "Visual Language display failed, visualisationsQueue_ empty";
        return;
      }

      images["depth"] = frame.depthImage;

      // Display the activation map of the frame w.r.t. the text embedding.
      // Both the language embedding and the feature map are normalized to [0, 1], therefore we can simply multiply
      // them and visualize the result, without having to clamp.
      if(!frame.featureMap.numel() || frame.rgbFrame.empty()) {
        DLOG(INFO) << "Vision Language Processor no feature map or RGB frame available for visualization";
        return;
      }
      okvis::TimerSwitchable t_vis_2("Vision Language - Draw visualization data for VL features");
      torch::Tensor textEmbedding;
      {
        std::lock_guard<std::mutex> lock(textEmbeddingMutex_);
        textEmbedding = textEmbedding_.clone();
      }
      torch::Tensor activationMap = torch::sum(frame.featureMap * textEmbedding.view({1, 1, -1}), 2);
      // Avoid overflow issues for color scaling
      activationMap = activationMap.clamp(0.0f, 1.0f);
      torch::Tensor visFeatureMap = (activationMap * 254.0f).to(torch::kU8);
      cv::Mat visMatFeatureMap = tensorToCvMatByte(visFeatureMap.detach().cpu()).clone();
      cv::Mat visMatFeatureMapColored;
      cv::applyColorMap(visMatFeatureMap, visMatFeatureMapColored, cv::COLORMAP_JET);

      // Upscale the VL activation map to the size of the RGB image. This is done by nearest neighbor,
      // as proposed by similar works in the field.
      cv::Mat rgbImage = frame.rgbFrame;
      cv::resize(visMatFeatureMapColored, visMatFeatureMapColored, rgbImage.size(), 0, 0, cv::INTER_CUBIC);

      // Display the RGB image and activation map side by side.
      cv::addWeighted(rgbImage, 0.2, visMatFeatureMapColored, 0.8, 0.0, images["language_rgb"]);
      images["raw_rgb"] = rgbImage;

      if(frame.samMasks.dim() != 3 || frame.samMasks.numel() == 0) {
        DLOG(INFO) << "Vision Language Processor no SAM masks available for visualization";
        return;
      }
      torch::Tensor samMasks = frame.samMasks.contiguous();
      const int64_t num_masks = samMasks.size(0);
      const int64_t mask_height = samMasks.size(1);
      const int64_t mask_width = samMasks.size(2);
      cv::Mat masks(mask_height, mask_width, CV_8UC3, cv::Scalar(0,0,0));
      float* data_ptr = reinterpret_cast<float*>(samMasks.data_ptr());
      std::random_device dev;
      std::mt19937 rng(dev());
      std::uniform_int_distribution<std::mt19937::result_type> dist(1,255);


      for(int64_t mask_index = 0; mask_index < num_masks; mask_index++){
        cv::Vec3b mask_colour(dist(rng), dist(rng), dist(rng));
        for(int64_t u = 0; u < mask_height; u++) {
          for(int64_t v = 0; v < mask_width; v++) {
            // Add bounds check
            assert(mask_index >= 0 && mask_index < samMasks.size(0));
            assert(u >= 0 && u < samMasks.size(1));
            assert(v >= 0 && v < samMasks.size(2));
            int64_t index = mask_index * mask_height * mask_width + u * mask_width + v;
            assert(index < samMasks.numel());  // Ensure the index is within valid range
            float mask_value = *(data_ptr + index);

            // Ensure the index is within bounds
            if(mask_value == 1) {
              masks.at<cv::Vec3b>(cv::Point(v, u)) = mask_colour;
            }
          }
        }
      }

      cv::addWeighted(rgbImage, 0.5, masks, 0.5, 0.0, images["sam_masks"]);
      t_vis_2.stop();
    }

    void VLProcessor::setLanguageEmbeddingVector(const torch::Tensor& embedding) {
      std::lock_guard<std::mutex> lock(textEmbeddingMutex_);
      textEmbedding_ = (embedding / embedding.norm()).detach().clone();  // normalize for easy visualization
    }
    void VLProcessor::setLanguageEmbedding(const Descriptor& embedding) {
      torch::Tensor embedding_tensor = torch::zeros({Descriptor::SizeAtCompileTime}, torch::kFloat32);
      memcpy(embedding_tensor.data_ptr(), embedding.data(), sizeof(Descriptor));
      setLanguageEmbeddingVector(embedding_tensor);
    }

    void VLProcessor::segmentPostProcessing(std::shared_ptr<torch::Tensor> bboxes, std::vector<int>& filteredIndexes){
      size_t numSegments = bboxes->sizes()[0];
      std::vector<cv::Rect> vec_bboxes;
      std::vector<float> vec_scores(numSegments, 1);
      vec_bboxes.reserve(numSegments);
      float* bbox_data_ptr = reinterpret_cast<float*>(bboxes->data_ptr());
      for(size_t i = 0; i < numSegments; ++i){
        int width = *(bbox_data_ptr + 4 * i + 2) - *(bbox_data_ptr + 4 * i);
        int height = *(bbox_data_ptr + 4 * i + 3) - *(bbox_data_ptr + 4 * i + 1);
        int x = *(bbox_data_ptr + 4 * i);
        int y = *(bbox_data_ptr + 4 * i + 1);
        vec_bboxes.emplace_back(x, y, width, height);
      }
      cv::dnn::NMSBoxes(vec_bboxes, vec_scores, 0.5, 0.7, filteredIndexes);
    }

    void VLProcessor::processLanguageNetwork(std::map<size_t, std::vector<okvis::CameraMeasurement>>& frames){
      // Set the processing flag true.
      isProcessing_ = true;

      if(frames.count(rgbCamIndex_) == 0)
      {
          LOG(WARNING) << "RGB Missing!";
          return;
      }

      int vec_idx_rgb = -1;

      for(size_t i = 0; i < frames.at(rgbCamIndex_).size(); i++) {
        if(!frames.at(rgbCamIndex_)[i].measurement.image.empty()) {
          vec_idx_rgb = i;
          break;
        }
      }

      if(vec_idx_rgb == -1) {
        LOG(WARNING) << "Data in the same camera index as RGB image but no colour data available";
        return;
      }

      auto& rgbFrame = frames.at(rgbCamIndex_)[vec_idx_rgb];

      // if(rgbFrame.measurement.image.rows != 480 || rgbFrame.measurement.image.cols != 640 || rgbFrame.measurement.image.channels() != 3) {
      //   LOG(WARNING) << "Cannot forward VL model, image size is not 480x640, got " << rgbFrame.measurement.image.cols << "x" << rgbFrame.measurement.image.rows;
      //   return;
      // }
      torch::NoGradGuard no_grad;

      // Convert the image to a tensor.
      okvis::TimerSwitchable t1("Vision Language Processor 1 - cv::Mat to Tensor");
      torch::Tensor imageTensor = torch::from_blob(rgbFrame.measurement.image.data, 
                                                   {rgbFrame.measurement.image.rows, rgbFrame.measurement.image.cols, 3}, 
                                                   torch::kByte);
      imageTensor = imageTensor.to(torch::kCUDA);
      t1.stop();

      // Forward the vision-language model.
      okvis::TimerSwitchable t2("Vision Language Processor 2 - Model inference CLIP");
      std::vector<torch::jit::IValue> inputs;
      inputs.emplace_back(imageTensor);
      std::shared_ptr<torch::Tensor> output(new torch::Tensor());
      *output = visionEncoder_.forward(inputs).toTensor();
      *output = output->permute({0, 2, 3, 1})[0];
      *output = output->to(torch::kFloat32);
      t2.stop();

      okvis::TimerSwitchable t3("Vision Language Processor 3 - Model inference SAM");
      std::shared_ptr<torch::Tensor> sam_segments(new torch::Tensor());
      std::shared_ptr<torch::Tensor> bbox_sam(new torch::Tensor());
      std::vector<torch::jit::IValue> sam_inputs;
      sam_inputs.emplace_back(imageTensor);
      auto sam_output = samModel_.forward(sam_inputs);
      *sam_segments = sam_output.toTuple()->elements()[0].toTensor().clone();
      *sam_segments = sam_segments->to(torch::kFloat32);
      if(sam_segments->dim() == 4 && sam_segments->size(3) == 1) {
        *sam_segments = sam_segments->squeeze(3);
      }
      if(sam_segments->dim() != 3) {
        LOG(WARNING) << "Unexpected SAM mask tensor with " << sam_segments->dim() << " dimensions";
        *sam_segments = torch::empty({0, rgbFrame.measurement.image.rows, rgbFrame.measurement.image.cols},
                                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
      }
      *bbox_sam = sam_output.toTuple()->elements()[1].toTensor().clone().contiguous();
      *bbox_sam = bbox_sam->to(torch::kFloat32);
      *bbox_sam = bbox_sam->to(torch::kCPU);
      t3.stop();

      std::vector<int> filtered_indexes;
      segmentPostProcessing(bbox_sam, filtered_indexes);
      if(filtered_indexes.empty() || sam_segments->size(0) == 0) {
        *sam_segments = torch::empty({0, sam_segments->size(1), sam_segments->size(2)},
                                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
      } else {
        torch::Tensor t_filtered_indexes = torch::from_blob(filtered_indexes.data(),
                                                            {static_cast<int64_t>(filtered_indexes.size())},
                                                            torch::kInt32).to(torch::kCUDA).clone();
        *sam_segments = sam_segments->index_select(0, t_filtered_indexes);
        *sam_segments = sam_segments->to(torch::kCPU).contiguous();
      }

      // Norm the feature map to [0, 1] for easy visualization.
      // Arguments: L2-norm, along the channel dimension, keep the dimension.
      *output = *output / torch::norm(*output, 2, 1, true);

      // Reshape the feature map to (C, H, W) and move to CPU.
      *output = output->squeeze().to(torch::kCPU).contiguous();  // Remove the batch dimension.
      rgbFrame.measurement.deep_learning_data["language_features"] = okvis::CameraData::DeepLearningImageData();
      rgbFrame.measurement.deep_learning_data["language_features"].shape = output->sizes().vec();
      rgbFrame.measurement.deep_learning_data["language_features"].data_ptr = std::shared_ptr<float>(output, reinterpret_cast<float*>(output->data_ptr()));

      rgbFrame.measurement.deep_learning_data["sam_masks"] = okvis::CameraData::DeepLearningImageData();
      rgbFrame.measurement.deep_learning_data["sam_masks"].shape = sam_segments->sizes().vec();
      rgbFrame.measurement.deep_learning_data["sam_masks"].data_ptr = std::shared_ptr<float>(sam_segments, reinterpret_cast<float*>(sam_segments->data_ptr()));

      if(imageCallback_) {
        imageCallback_(frames);
      }

      // Save Depth + RGB + Color Images
      okvis::Time rgb_timestamp = rgbFrame.timeStamp;
      okvis::Time depth_timestamp;
      
      okvis::TimerSwitchable t4("Vision Language Processor 4 - Draw visualization data");
      cv::Mat outputDepth;
      for(const auto& vec_images : frames) {
        for(const auto& image : vec_images.second) {
          if(!image.measurement.depthImage.empty()) {
            outputDepth = image.measurement.depthImage.clone();
            depth_timestamp = image.timeStamp;
            break;
          }
        }
      }
    //  // Get Sam Masks
    //  int64_t num_masks = rgbFrame.measurement.deep_learning_data["sam_masks"].shape[0];
    //  int64_t height = rgbFrame.measurement.deep_learning_data["sam_masks"].shape[1];
    //  int64_t width = rgbFrame.measurement.deep_learning_data["sam_masks"].shape[2];
    //  cv::Mat unique_output_features(height, width, CV_32FC1, cv::Scalar(0.0));

    //  for(int64_t i = 0; i < num_masks; i++) {
    //    float* segment_ptr = rgbFrame.measurement.deep_learning_data["sam_masks"].data_ptr.get() + i * height * width;
    //    cv::Mat segment_mask(height, width, CV_32FC1, segment_ptr);
    //    segment_mask.convertTo(segment_mask, CV_8UC1); // ToDo: is this needed?
    //    Descriptor vl_feature;
    //    Descriptor query(okvis::chair_embedding.data());
    //    size_t num_evaluated_pixels = averageDescriptor(rgbFrame.measurement.deep_learning_data["language_features"],
    //                                                    segment_mask, vl_feature, 20);

    //    const float d = vl_feature.normalized().dot(query.normalized());
    //    unique_output_features.setTo(cv::Scalar(d), segment_mask);
    //  }

    //  double tmp_min, tmp_max;
    //  cv::minMaxLoc(unique_output_features, &tmp_min, &tmp_max);
    //  const float factor = std::min(2.0f, 1.0f/float(tmp_max));
    //  unique_output_features = unique_output_features * factor;
    //  //activationMap = (activationMap + 1) / 2;

    //  unique_output_features = unique_output_features * 254.0f;
    //  unique_output_features.convertTo(unique_output_features, CV_8UC3);
    //  cv::applyColorMap(unique_output_features, unique_output_features, cv::COLORMAP_JET);
    //  cv::imwrite("/tmp/averaged-feature-similarity-"+std::to_string(depth_timestamp.toNSec())+".png", unique_output_features);

      // if (saveDataset_) {
      //   cv::Mat depthToBeSaved = outputDepth.clone();
      //   if(depthToBeSaved.type() != CV_32FC1) {
      //     throw std::runtime_error("Only implemented for CV_32FC1 cv::Mat");
      //   }

      //   // cv::MAT and DepthFrame keep data stored in row major format.
      //   if(!depthToBeSaved.isContinuous()) {
      //     //TODO write down row by row, first iterate rows the columns and add them to a vector
      //     throw std::runtime_error("Only implemented for continuous cv::Mat");
      //   }

      //   se::save_depth_png(reinterpret_cast<const float*>(depthToBeSaved.data), Eigen::Vector2i(depthToBeSaved.cols, depthToBeSaved.rows), datasetPath_ + "/depth/" + std::to_string(depth_timestamp.toSec()) + ".png", 1.0f);
      //   depthFileList_ << std::to_string(depth_timestamp.toSec()) << " depth/" + std::to_string(depth_timestamp.toSec()) + ".png" << std::endl;
      // }

      double min, max;
      cv::minMaxLoc(outputDepth, &min, &max);
      outputDepth = outputDepth / max;
      outputDepth.setTo(1.0, outputDepth > 1.0);
      outputDepth = outputDepth * 255.0f;
      outputDepth.convertTo(outputDepth, CV_8UC1);
      cv::Mat visMatColored;
      cv::applyColorMap(outputDepth, visMatColored, cv::COLORMAP_INFERNO);

      VisualizationData visData;
      visData.rgbFrame = rgbFrame.measurement.image.clone();
      visData.depthImage = visMatColored;
      visData.featureMap = output->clone();
      visData.samMasks = sam_segments->clone();

      // if(saveDataset_){
      //   int width = sam_segments->sizes()[1];
      //   int height = sam_segments->sizes()[2];

      //   torch::Tensor test = sam_segments->clone();
      //   test = test.permute({1,2,0,3});

      //   int num_masks = test.size(2);
      //   int non_unique_counter = 0;

      //   cv::Mat samMask(cv::Size{height, width}, CV_8UC1, cv::Scalar(0,0,0));

      //   for (int i = 0; i < 640; i++) {
      //       for(int j = 0; j < 480; j++){
      //           auto nz = at::nonzero(test[j][i]);
      //           int64_t n_non_zero = nz.sizes()[0];
      //           if(nz.size(0) > 1){
      //               int max_id = nz[n_non_zero-1][0].item<int>();
      //               samMask.at<uchar>(j,i) = max_id;
      //           }
      //           else if(nz.size(0) == 1) {
      //               int max_id = nz[0][0].item<int>();
      //               samMask.at<uchar>(j,i) = max_id;
      //           }
      //           else {
      //               samMask.at<uchar>(j,i) = 255;
      //           }
      //       }
      //   }

      //     cv::imwrite(datasetPath_ + "/rgb/" + std::to_string(rgb_timestamp.toSec()) + ".png", rgbFrame.measurement.image);
      //     rgbFileList_ << std::to_string(rgb_timestamp.toSec()) << " " << "rgb/" + std::to_string(rgb_timestamp.toSec()) + ".png" << std::endl;
      //     cv::imwrite(datasetPath_ + "/sam/" + std::to_string(rgb_timestamp.toSec()) + ".png", samMask);
      //     samFileList_ << std::to_string(rgb_timestamp.toSec()) << " " << "sam/" + std::to_string(rgb_timestamp.toSec()) + ".png" << std::endl;
      // }

      visualisationsQueue_.PushNonBlockingDroppingIfFull(visData, 1);
      t4.stop();

      // Set the processing flag true.
      isProcessing_ = false;
    }

    bool VLProcessor::addImages(const std::map<size_t, std::pair<okvis::Time, cv::Mat>> & images, 
                                const std::map<size_t, std::pair<okvis::Time, cv::Mat>> & depthImages) {
      if(depthImages.size() == 0) {
          LOG(WARNING) << "Depth Image missing.";
          return false;
      }
      std::map<size_t, std::vector<okvis::CameraMeasurement>> camera_measurements;
      for(auto& it: images){
        okvis::CameraMeasurement camMeasurement;
        camMeasurement.measurement.image = it.second.second.clone();
        camMeasurement.timeStamp = it.second.first;
        camMeasurement.sensorId = it.first;
        camera_measurements[it.first] = {camMeasurement};
      }

      //Now we synchronize the depth images. If they are in the same index as a prior one, we check 
      // if it is timestamp synchronized. If not, we add it to a separate measurement, else, we add it to the
      // measurement with the same timestamp
      for(auto& it: depthImages) {
        okvis::CameraMeasurement camMeasurement;
        camMeasurement.measurement.depthImage = it.second.second.clone();
        camMeasurement.timeStamp = it.second.first;
        camMeasurement.sensorId = it.first;
        auto camera_measurements_it = camera_measurements.find(it.first);
        if(camera_measurements_it == camera_measurements.end()) {
          camera_measurements[it.first] = {camMeasurement};
        } else {
          bool same_timestamp = false;
          for(auto& cam_measure : camera_measurements_it->second) {
            if(cam_measure.timeStamp == it.second.first) {
              same_timestamp = true;
              cam_measure.measurement.depthImage = it.second.second.clone();
              break;
            }
          }
          if(!same_timestamp) {
            camera_measurements.at(it.first).push_back(camMeasurement);
          }
        }
      }

      if(blocking_){
        cameraMeasurementsQueue_.PushBlockingIfFull(camera_measurements, 1);
        return true;
      }
      else{
        const int queue_size = 10;
        if(cameraMeasurementsQueue_.PushNonBlockingDroppingIfFull(camera_measurements, queue_size)) {
          DLOG(INFO) <<  "language processing frame drop";
          return false;
        }
        else{
          return true;
        }
      }
    }

    void VLProcessor::processing() {
      while(!shutdown_) {
        std::map<size_t, std::vector<okvis::CameraMeasurement>> frame;
        if(blocking_){
          while(cameraMeasurementsQueue_.PopBlocking(&frame)) {
            processLanguageNetwork(frame);
          }
        }
        else{
          while(cameraMeasurementsQueue_.PopNonBlocking(&frame)) {
            processLanguageNetwork(frame);
          }
        }
      }
    }

    bool VLProcessor::finishedProcessing() {
      if (cameraMeasurementsQueue_.Size() == 0 && !isProcessing_) {
        return true;
      }
      else {
        return false;
      }
    }
}
