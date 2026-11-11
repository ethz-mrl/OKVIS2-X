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
 * @file Publisher.hpp
 * @brief Header file for the Publisher class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#ifndef INCLUDE_OKVIS_ROS2_PUBLISHER_HPP_
#define INCLUDE_OKVIS_ROS2_PUBLISHER_HPP_

#include <memory>

#if __has_include(<cv_bridge/cv_bridge.hpp>) // requires GCC >= 5
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h> // ros2 changed to .hpp some point...
#endif
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include <opencv2/core/core.hpp>
#pragma GCC diagnostic pop
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <mutex>

#include <okvis/ViInterface.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/Parameters.hpp>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/Time.hpp>
#include <okvis/ObjectMapping.hpp>
#include <okvis/TrajectoryOutput.hpp>

#include <se/external/tinycolormap.hpp>

#include <okvis/ros2/eigen_conversions.hpp>

#include <okvis/ThreadedPublisher.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

/**
 * @brief This class handles the publishing to either ROS topics or files.
 */
class Publisher
{
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /// \brief Default constructor.
  Publisher();

  /// \brief Constructor with node.
  /// \param node The ROS2 node.
  Publisher(
    std::shared_ptr<rclcpp::Node> node,
    std::shared_ptr<ThreadedPublisher> threadedOdometryPublisher, 
    std::shared_ptr<ThreadedPublisher> threadedImagePublisher,
    std::shared_ptr<ThreadedPublisher> threadedPublisher);
  ~Publisher();

  /// \name Setters
  /// \{

  /// \brief Set up the whole node.
  /// \param node The ROS2 node.
  void setupNode(std::shared_ptr<rclcpp::Node> node);

  /**
   * @brief Set the body.
   * @param T_BS Transform body-IMU.
   */
  void setBodyTransform(const okvis::kinematics::Transformation& T_BS);

  /**
   * @brief Set the realtime publishing rate.
   * @param odometryPublishingRate The rate.
   */
  void setOdometryPublishingRate(double odometryPublishingRate) {
    RCLCPP_INFO(node_->get_logger(), "Setting odometry publishing rate to %f Hz.", odometryPublishingRate);
    odometryPublishingRate_ = odometryPublishingRate;
  }

  /**
   * @brief Set CSV file.
   * @param filename Write CSV trajectory to this file.
   * @param rpg If true, uses the RPG format, otherwise the EuRoC format.
   */
  void setCsvFile(const std::string & filename, bool rpg = false);

  /// @}

  /**
   * @brief Process the updated states and publish
   *        (plus write it to trajectory file and visualise, if desired). Set as callback.
   * @param state The current state to process.
   * @param trackingState The additional tracking info to process.
   * @param updatedStates All updated states.
   * @param landmarks The currently optimised landmarks.
   */
  void publishEstimatorUpdate(const State& state, const TrackingState & trackingState,
                              std::shared_ptr<const AlignedMap<StateId, State>> updatedStates,
                              std::shared_ptr<const okvis::MapPointVector> landmarks);

  /**
   * @brief Set up the topics.
   * @param nCameraSystem Multi-camera sensor setup.
   */
  void setupImageTopics(const okvis::cameras::NCameraSystem & nCameraSystem);

  /**
   * @brief Set up the topics.
   */
  void setupNetworkTopics(const std::string & topicName);
  
  void setMeshesPath(std::string meshesDir);

  /**
   * \brief          Publish any named images as such.
   * \param images   Named images to publish.
   * \return True on success.
   */
  bool publishImages(const std::map<std::string, cv::Mat>& images) const;

  /**
   * \brief          Add an IMU measurement for propagation and publishing.
   * \param stamp    The measurement timestamp.
   * \param alpha    The acceleration measured at this time.
   * \param omega    The angular velocity measured at this time.
   * \return True on success.
   */
  bool realtimePredictAndPublish(const okvis::Time& stamp,
                                 const Eigen::Vector3d& alpha,
                                 const Eigen::Vector3d& omega);

  /**
   * @brief Submap mesh callback
   */
  void publishSubmapsAsCallback(std::unordered_map<uint64_t, okvis::kinematics::Transformation, std::hash<uint64_t>, std::equal_to<uint64_t>, Eigen::aligned_allocator<std::pair<const uint64_t, okvis::kinematics::Transformation>>> submapPoseLookup, 
                                std::unordered_map<uint64_t, std::shared_ptr<okvis::SupereightMapType>> submapLookup,
                                std::shared_ptr<okvis::ObjectMap> objectMap);

  /**
   * @brief Publish a horizontal slice through the occupancy field
   */
  void publishFieldSliceAsCallback(const State& latest_state,
                                   const AlignedUnorderedMap<uint64_t, se::Submap<okvis::SupereightMapType>>& seSubmapLookup);

  /**
   * @brief Map-to-frame points visualization callback
   */
  void publishAlignmentPointsAsCallback(const okvis::Time& timestamp, const okvis::kinematics::Transformation& T_WS,
                                        const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& alignPointCloud,
                                        bool isMapFrame=false);

  /**
   * @brief Set query (embedding) for VL queries
   */
  void setTextEmbeddingVector(const Descriptor& text_embedding_vector) {
    std::lock_guard<std::mutex> lock(submap_mesh_mtx_);
    text_embedding_vector_ = text_embedding_vector;
  }

  /**
   * @brief Set Mesh Publishing Mode (Colors or Activations)
   */
  void setMeshPublishingMode(const std::string& publishMode);

  /**
   * @brief Re-Publish meshes (if publishing mode has changed)
   */
  void republishMeshes();

  /** 
   * @brief Set Cutoff z value for published meshes
   */
  void setMeshCutoffZ(float z_max) {
    std::lock_guard<std::mutex> lock(submap_mesh_mtx_);
    mesh_cutoff_z_ = z_max;
  }

  /**
   * @brief Publish auxiliary state obtained from IMU real-time propagation
   */
  void publishRealTimePropagation(const okvis::Time& time, const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation, 
                                  const Eigen::Vector3d& linear_velocity, const Eigen::Vector3d& angular_velocity);

  private:

  /// @name Node and subscriber related
  /// @{

  std::shared_ptr<rclcpp::Node> node_; ///< The node.

  std::shared_ptr<ThreadedPublisher> threadedOdometryPublisher_;  ///< Odometry publishers.
  std::shared_ptr<ThreadedPublisher> threadedImagePublisher_;  ///< Image publishers.
  std::shared_ptr<ThreadedPublisher> threadedPublisher_;  ///< Non-images publishers.

  /// \brief The publisher for matched points.
  okvis::ThreadedPublisher::PublisherHandle<sensor_msgs::msg::PointCloud2> pubPointsMatched_;
  /// \brief The publisher for the odometry.
  okvis::ThreadedPublisher::PublisherHandle<nav_msgs::msg::Odometry> pubObometry_;
  /// \brief The publisher for the path.
  okvis::ThreadedPublisher::PublisherHandle<visualization_msgs::msg::Marker> pubPath_;
  /// \brief The publisher for the transform.
  okvis::ThreadedPublisher::PublisherHandle<geometry_msgs::msg::TransformStamped> pubTransform_;
  /// \brief The publisher for a robot / camera mesh.
  okvis::ThreadedPublisher::PublisherHandle<visualization_msgs::msg::Marker> pubMesh_;
  /// \brief The publisher for submap meshes.
  okvis::ThreadedPublisher::PublisherHandle<visualization_msgs::msg::MarkerArray> pubSubmapMesh_;
  /// \brief The publisher for aligned points.
  okvis::ThreadedPublisher::PublisherHandle<sensor_msgs::msg::PointCloud2> pubPointsAlignment_;
  /// \brief The publisher for the planned path.
  okvis::ThreadedPublisher::PublisherHandle<nav_msgs::msg::Path> pubPlannedPath_;
  /// \brief The publisher for the slice.
  okvis::ThreadedPublisher::PublisherHandle<visualization_msgs::msg::Marker> slice_pub_;

  /// \brief Image publishers.
  std::map<std::string, okvis::ThreadedPublisher::PublisherHandle<sensor_msgs::msg::Image>> pubImages_; ///< Image publisher map.

  /// @}
  /// @name To be published
  /// @{

  std::vector<ImuMeasurement> imuMeasurements_; ///< Buffered IMU measurements (realtime pub.).
  okvis::Trajectory trajectory_; ///< Underlying trajectory object for state queries.
  okvis::kinematics::Transformation T_BS_; ///< transform from imu frame to body frame
  okvis::kinematics::Transformation T_SB_; ///< transform from body frame to imu frame
  visualization_msgs::msg::Marker::SharedPtr meshMsg_; ///< Mesh message.
  double odometryPublishingRate_; ///< Keep track of last publishing (to maintain rate).
  okvis::Time lastTime_ = okvis::Time(0); ///< Publishing rate for realtime propagation.
  TrajectoryOutput trajectoryOutput_; ///< Trajectory output in case demanded.
  std::atomic_bool trajectoryLocked_; ///< Lock the trajectory object (realtime/update are async.).
  std::string meshesDir_; ///< Directory for meshes.
  nav_msgs::msg::Path path_; ///< The path message.
  Eigen::Matrix4d T_SC_; // Tf from Sensor to Camera
  Eigen::Quaterniond q_sc_;
  
  nav_msgs::msg::Odometry lastOdom_; ///< Last odometry message.

  // Submap-related members.
  std::unordered_map<uint64_t, visualization_msgs::msg::Marker> submapMeshLookup_; ///< Lookup for submap meshes.
  std::unordered_map<uint64_t, okvis::SupereightMapType::SurfaceMesh> submapSurfaceMesh_; ///< Surface meshes for submaps.
  std::unordered_map<uint64_t, visualization_msgs::msg::Marker> submapMeshLookup_embedding_; ///< Lookup for submap embedding meshes.
  std::unordered_map<uint64_t, visualization_msgs::msg::Marker> submapMeshLookup_rgb_; ///< Lookup for submap RGB meshes.
  std::map<uint64_t, const okvis::AlignedMap<se::id_t, SubmapObject>*> submapObjects_; ///< Objects in submaps.
  std::map<uint64_t, Eigen::Matrix4f> submapPoses_; ///< Poses of submaps.
  std::mutex submap_mesh_mtx_; ///< Protects cached submap meshes and query visualization state.
  std::atomic_bool skip_uncolored_ = true; ///< Flag if uncolored vertices should be skipped at mesh publishing

  // Vision-Language Query Related Members
  Descriptor text_embedding_vector_; ///< Text embedding vector.
  std::string publishMode_ = "Colors"; ///< Mode which meshes are published: RGB ("Colors") | Text Query ("Activations")

  float mesh_cutoff_z_ = std::numeric_limits<float>::max(); ///< z cutoff value for visualisation
};

}

#endif /* INCLUDE_OKVIS_ROS2_PUBLISHER_HPP_ */
