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

/**
 * @file Subscriber.cpp
 * @brief Source file for the Subscriber class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */
 
#include <glog/logging.h>
#include <okvis/ros2/Subscriber.hpp>
#include <okvis/ros2/PointCloudUtilities.hpp>
#include <pcl_conversions/pcl_conversions.h>

#ifdef OKVIS_USE_NN
#include <okvis/Processor.hpp>
#endif

#define OKVIS_THRESHOLD_SYNC 0.01 ///< Sync threshold in seconds.

/// \brief okvis Main namespace of this package.
namespace okvis {

Subscriber::~Subscriber()
{
  if (imgTransport_ != nullptr)
    imgTransport_.reset();
}

Subscriber::Subscriber(std::shared_ptr<rclcpp::Node> node, 
                       okvis::ViInterface* viInterfacePtr,
                       okvis::Publisher* publisher, 
                       const okvis::ViParameters& parameters,
                       okvis::SubmappingInterface* seInterface,
                       bool isDepthCamera, bool isLiDAR)
{
  viInterface_ = viInterfacePtr;
  seInterface_ = seInterface;
  publisher_ = publisher;
  parameters_ = parameters;
  setNodeHandle(node, isDepthCamera, isLiDAR);
}

void Subscriber::setNodeHandle(std::shared_ptr<rclcpp::Node> node,
                               bool isDepthCamera, bool isLiDAR)
{
  node_ = node;

  imageSubscribers_.resize(parameters_.nCameraSystem.numCameras());
  depthImageSubscribers_.resize(parameters_.nCameraSystem.numCameras());
  imagesReceived_.resize(parameters_.nCameraSystem.numCameras());
  depthImagesReceived_.resize(parameters_.nCameraSystem.numCameras());

  // set up image reception
  if (imgTransport_ != nullptr)
    imgTransport_.reset();
  imgTransport_ = std::make_shared<image_transport::ImageTransport>(node_);

  // set up callbacks
  for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) {
    imageSubscribers_[i] = imgTransport_->subscribe(
        "/okvis/cam" + std::to_string(i) +"/image_raw",
        30 * parameters_.nCameraSystem.numCameras(),
        std::bind(&Subscriber::imageCallback, this, std::placeholders::_1, i,
          parameters_.nCameraSystem.cameraType(i).isColour));
  }

  subImu_ = node_->create_subscription<sensor_msgs::msg::Imu>(
      "/okvis/imu0", 1000, 
      std::bind(&Subscriber::imuCallback, this, std::placeholders::_1));

  if(isDepthCamera){
    syncDepthImages_ = true;

    // set up callbacks
    for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) {
      depthImageSubscribers_[i] = imgTransport_->subscribe(
        "/okvis/depth" + std::to_string(i) + "/image_raw",
        30 * parameters_.nCameraSystem.numCameras(),
        std::bind(&Subscriber::depthCallback, this, std::placeholders::_1, i));
    }
  }

  if(isLiDAR){
    subLiDAR_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/okvis/lidar", 100000,
      std::bind(&Subscriber::lidarCallback, this, std::placeholders::_1));
  }
}

void Subscriber::shutdown() {
  // stop callbacks
  for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) {
    imageSubscribers_[i].shutdown();
  }
  subImu_.reset();
}

void Subscriber::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg,
                               unsigned int cameraIndex, bool isColour)
{
  cv_bridge::CvImageConstPtr cv_ptr;
  cv::Mat raw;
  try
  {
    if(!isColour){
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
    } else {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
    }
    raw = cv_ptr->image;
  }
  catch (cv_bridge::Exception& e)
  {
    RCLCPP_ERROR(node_->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }
  cv::Mat filtered;
  filtered = raw.clone();

  // adapt timestamp
  okvis::Time t(msg->header.stamp.sec, msg->header.stamp.nanosec);

  // insert
  std::lock_guard<std::mutex> lock(time_mutex_);
  imagesReceived_.at(cameraIndex)[t.toNSec()] = filtered;
  
  // try sync
  synchronizeData();
}

void Subscriber::imuCallback(const sensor_msgs::msg::Imu& msg)
{
  // construct measurement
  okvis::Time timestamp(msg.header.stamp.sec, msg.header.stamp.nanosec);
  Eigen::Vector3d acc(msg.linear_acceleration.x, msg.linear_acceleration.y,
                      msg.linear_acceleration.z);
  Eigen::Vector3d gyr(msg.angular_velocity.x, msg.angular_velocity.y,
                      msg.angular_velocity.z);                    
  
  // forward to estimator
  viInterface_->addImuMeasurement(timestamp, acc, gyr);
  
   // also forward for realtime prediction
  // ThreadedSlam applies s_a itself; the real-time propagation needs it too (e.g. Livox IMU in g)
  const Eigen::Vector3d accScaled = parameters_.imu.s_a.cwiseProduct(acc);
  if(seInterface_) {
    seInterface_->realtimePredict(timestamp, accScaled, gyr);
  }
  else if(publisher_) {
    publisher_->realtimePredictAndPublish(timestamp, accScaled, gyr);
  }
}

void Subscriber::depthCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg, 
                               unsigned int cameraIndex){
  // TODO now we work only with one camera of depth images,
  // as it is what the submapping interface accepts
  cv_bridge::CvImageConstPtr cv_ptr;
  cv::Mat raw;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
    raw = cv_ptr->image;
  } catch (cv_bridge::Exception& e) {
    RCLCPP_ERROR(node_->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }

  okvis::Time t(msg->header.stamp.sec, msg->header.stamp.nanosec);

  if(!viInterface_->addDepthMeasurement(t, raw)){
    //LOG(WARNING) << "Dropped last depth image frame for okvis interface";
  }

  // try sync
  std::lock_guard<std::mutex> lock(time_mutex_);
  depthImagesReceived_.at(cameraIndex)[t.toNSec()] = raw;
  //add here the depth images to the receiver
  synchronizeData();
}

void Subscriber::lidarCallback(const sensor_msgs::msg::PointCloud2& msg){

  // Check which type of LiDAR Point Cloud it is
  bool is_blk = (okvis::pointcloud_ros::has_field(msg, "stamp_high") && okvis::pointcloud_ros::has_field(msg, "stamp_low"));
  bool is_hesai = okvis::pointcloud_ros::has_field(msg, "timestamp");

  if(!(is_blk || is_hesai)){ // Default; use header timestamp
    Eigen::Vector3d ray;
    okvis::Time t(msg.header.stamp.sec, msg.header.stamp.nanosec);

    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(msg, pcl_pc2);
    pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(pcl_pc2,*temp_cloud);
    pcl::PointCloud<pcl::PointXYZ>::iterator it = temp_cloud->begin();

    for (/*it*/; it != temp_cloud->end(); it++){
      ray << it->x, it->y, it->z;
      if(!seInterface_->addLidarMeasurement(t, ray)) LOG(WARNING) << "Dropped last lidar measurement from SubmappingInterface.";
      viInterface_->addLidarMeasurement(t, ray);
    }
  }
  else{
    std::vector<okvis::Time, Eigen::aligned_allocator<okvis::Time>> timestamps;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> rays;

    if(is_hesai){
      okvis::pointcloud_ros::hesai_lidar2points(msg, timestamps, rays);
    }
    else if(is_blk){
      okvis::pointcloud_ros::blk_lidar2points(msg, timestamps, rays);
    }

    for(size_t i = 0; i < timestamps.size(); i++){
      if(!seInterface_->addLidarMeasurement(timestamps[i], rays[i])) LOG(WARNING) << "Dropped last lidar measurement from SubmappingInterface.";
      viInterface_->addLidarMeasurement(timestamps[i], rays[i]);
    }
  }
}

void Subscriber::synchronizeData() {
  std::set<uint64_t> allTimes;
  const int numCameras = imagesReceived_.size();
  for(int i=0; i < numCameras; ++i) {
    if(!parameters_.nCameraSystem.cameraType(i).isUsed) continue; // Only consider SLAM cameras here
    for(const auto & entry : imagesReceived_.at(i)) {
      allTimes.insert(entry.first);
    }
  } //We are synchronizing the depth cameras against the ir cameras

  for(const auto & time : allTimes) {
    // note: ordered old to new
    std::vector<uint64_t> syncedTimes(numCameras, 0);
    uint64_t depthSyncedTime; //Assume that there is only one depth image (currently supported by the submapping interface)
    std::map<uint64_t, cv::Mat> images;
    std::map<uint64_t, cv::Mat> depthImages;

    std::map<uint64_t, std::pair<okvis::Time, cv::Mat>> timestampedImages;
    std::map<uint64_t, std::pair<okvis::Time, cv::Mat>> timestampedDepthImages;

    okvis::Time tcheck;
    tcheck.fromNSec(time);
    bool synced = true;
    for(int i=0; i < numCameras; ++i) {
      bool syncedi = false;
      for(const auto & entry : imagesReceived_.at(i)) {
        okvis::Time ti;
        ti.fromNSec(entry.first);
        bool slam_use = !parameters_.nCameraSystem.cameraType(i).isUsed;
        if(fabs((tcheck-ti).toSec()) < OKVIS_THRESHOLD_SYNC && (tcheck >= ti || slam_use)) {
          syncedTimes.at(i) = entry.first;
          images[i] = imagesReceived_.at(i).at(entry.first);
          timestampedImages[i] = std::make_pair(ti, imagesReceived_.at(i).at(entry.first));
          syncedi = true;
          break;
        } 
      }

      if(!syncedi) {
        synced = false;
        break;
      }
    }

    bool syncedDepth = true;
    if(syncDepthImages_){
      syncedDepth = false;
      for(int i = 0; i < numCameras; ++i) {
        for(const auto & entry : depthImagesReceived_.at(i)){
          okvis::Time tdepth;
          tdepth.fromNSec(entry.first);
          //There is a higher unsynchronization between depth images and ir images from the realsense
          if(std::abs((tcheck - tdepth).toSec()) < OKVIS_THRESHOLD_SYNC && tcheck>=tdepth) {
            depthSyncedTime = entry.first;
            depthImages[i] = depthImagesReceived_.at(i).at(entry.first);
            timestampedDepthImages[i] = std::make_pair(tdepth, depthImagesReceived_.at(i).at(entry.first));
            syncedDepth = true;
            break;
          }
        }

        if(syncedDepth){
          //For now we assume there is only one depth image per synchronization process, needs discussion
          break;
        }
      }
    }

    if(synced && syncedDepth) {
      bool isProcessor = false;
      #ifdef OKVIS_USE_NN
      okvis::Processor* casted_processor = dynamic_cast<okvis::Processor*>(viInterface_);
      if (casted_processor) {
        isProcessor = true;
        if(!casted_processor->addImages(timestampedImages, timestampedDepthImages)) {
          LOG(WARNING) << "Frame not added to Processor at t="<< timestampedImages.at(0).first;
        }
      }
      else if(!viInterface_->addImages(tcheck, images, depthImages)) {
        LOG(WARNING) << "Frame not added at t="<< tcheck;
      }
      #else
      if(!viInterface_->addImages(tcheck, images, depthImages)) {
        LOG(WARNING) << "Frame not added at t="<< tcheck;
      }
      #endif
      if(!isProcessor && syncDepthImages_) {
        //If we have depth images, then we send the data to SI from here, if not the Processor handles it. As for LiDAR, we need to decide
        //how to continue with it. Do we want colour fusion at all?
        std::map<size_t, std::vector<okvis::CameraMeasurement>> cameraMeasurements;
        
        for(const auto& it : timestampedImages) {
          okvis::CameraMeasurement cameraMeasure;
          cameraMeasure.timeStamp = tcheck;
          cameraMeasure.measurement.image = it.second.second.clone();
          cameraMeasure.sensorId = it.first;
          cameraMeasurements[it.first] = {cameraMeasure};
        }

        for(const auto& it : timestampedDepthImages) {
          if(cameraMeasurements.find(it.first) != cameraMeasurements.end()) {
            //Check if they are time synchronized or not
            bool imageAdded = false;
            for(auto& image : cameraMeasurements.at(it.first)) {
              if(image.timeStamp == it.second.first){
                image.measurement.depthImage = it.second.second.clone();
                imageAdded = true;
              }
            }

            if(!imageAdded) {
              okvis::CameraMeasurement cameraMeasure;
              cameraMeasure.timeStamp = it.second.first;
              cameraMeasure.measurement.depthImage = it.second.second.clone();
              cameraMeasure.sensorId = it.first;
              cameraMeasurements[it.first] = {cameraMeasure};
            }
          } else {
            okvis::CameraMeasurement cameraMeasure;
            cameraMeasure.timeStamp = it.second.first;
            cameraMeasure.measurement.depthImage = it.second.second.clone();
            cameraMeasure.sensorId = it.first;
            cameraMeasurements[it.first] = {cameraMeasure};
          }
        }
        
        if(!seInterface_->addDepthMeasurement(cameraMeasurements)){
          LOG(WARNING) << "Frame not added to SI at t="<< cameraMeasurements.at(0)[0].timeStamp;
        }
      }

      // remove all the older stuff from buffer
      for(int i=0; i < numCameras; ++i) {
        auto end = imagesReceived_.at(i).find(syncedTimes.at(i));
        if(end!=imagesReceived_.at(i).end()) {
          ++end;
        }
        imagesReceived_.at(i).erase(imagesReceived_.at(i).begin(), end);
      }

      if(syncDepthImages_) {
        for(int i=0; i < numCameras; ++i) {
          auto end = std::find_if(depthImagesReceived_.at(i).begin(), depthImagesReceived_.at(i).end(),
                                 [depthSyncedTime](const auto& x) { return x.first > depthSyncedTime; });
          depthImagesReceived_.at(i).erase(depthImagesReceived_.at(i).begin(), end);
        }
      }
    }
  }
}

} // namespace okvis