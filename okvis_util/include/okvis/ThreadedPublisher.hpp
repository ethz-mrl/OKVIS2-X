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
* @file ThreadedPublisher.hpp
* @brief Defines the ThreadedPublisher class for asynchronous ROS2 message publishing.
*
* This utility enables concurrent, non-blocking publication of the latest
* messages across multiple ROS2 publishers, including image_transport
* publishers, using a background thread. It is designed to store and publish
* only the most recent message per topic, discarding older ones if a new one
* arrives before publishing.
*
* @author Micha Bosshart
* @author Joshua Naef
*/

#ifndef OKVIS_THREADEDPUBLISHER_HPP_
#define OKVIS_THREADEDPUBLISHER_HPP_

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

#include <glog/logging.h>
#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <okvis/Time.hpp>
#include <okvis/threadsafe/ThreadsafeQueue.hpp>

namespace okvis {

/**
* @class ThreadedPublisher
* @brief Manages asynchronous publishing of the latest messages on multiple
* ROS2 topics.
*
* This class maintains type-erased publishers that each store and publish only
* the most recent message. It supports both standard
* `rclcpp::Publisher<MessageType>` and `image_transport::Publisher` for
* `sensor_msgs::msg::Image`.
*/
class ThreadedPublisher {

  /**
   * @brief Abstract base class for type-erased publishers.
   */
  struct PublisherBase {
  public:
    virtual ~PublisherBase() = default;
    virtual void publishLatest() = 0;
    virtual void shutdown() = 0;
  };

  /**
   * @brief Type-specific implementation of PublisherBase using templated
   * message types.
   *
   * Supports both standard ROS publishers and image_transport for
   * sensor_msgs::msg::Image.
   */
  template <typename MessageType> struct PublisherTyped : PublisherBase {
    /**
     * @brief Constructor.
     * @param name ROS2 topic name.
     * @param node Shared pointer to the ROS2 node.
     * @param notifyCallback Function to notify when a new message is available.
     */
    PublisherTyped(std::string name, rclcpp::Node::SharedPtr node,
                   std::function<void()> notifyCallback)
        : node_(node),
          queue_(std::make_shared<
                 okvis::threadsafe::Queue<std::shared_ptr<MessageType>>>()),
          notifyCallback_(notifyCallback),
          name_(name) {
      if constexpr (std::is_same_v<MessageType, sensor_msgs::msg::Image>) {
        // Use image_transport for Image messages
        imageTransport_ =
            std::make_shared<image_transport::ImageTransport>(node_);
        imagePublisher_ = std::make_shared<image_transport::Publisher>(
            imageTransport_->advertise(name, 1));
      } else {
        // Standard ROS2 publisher
        publisher_ = node_->create_publisher<MessageType>(name, 1);
      }
    }

    /**
     * @brief Set the latest message to be published.
     *
     * Replaces any previously stored message with the new one.
     *
     * @param msg Shared pointer to the new message.
     */
    void setLatestMessage(std::shared_ptr<MessageType> msg) {
      if (!msg) {
        return;
      }
      auto msg_snapshot = std::make_shared<MessageType>(*msg);
      queue_->PushNonBlockingDroppingIfFull(msg_snapshot, 1);

      if (notifyCallback_) {
        notifyCallback_();
      }
    }

    /**
     * @brief Publish the most recently stored message, if any.
     */
    void publishLatest() override {
      std::shared_ptr<MessageType> msg = nullptr;

      if (!queue_->PopNonBlocking(&msg)) {
        return;
      }

      try {
        if constexpr (std::is_same_v<MessageType, sensor_msgs::msg::Image>) {
          imagePublisher_->publish(msg);
        } else {
          publisher_->publish(*msg);
        }
      } catch (const std::exception &e) {
        LOG(WARNING) << "Error publishing message: to topic " << name_ << ". Error message is: " << e.what();
      }
    }

    /**
     * @brief Shuts down the internal publisher.
     */
    void shutdown() override {
      queue_->Shutdown();
      if constexpr (std::is_same_v<MessageType, sensor_msgs::msg::Image>) {
        if (imagePublisher_) {
          imagePublisher_->shutdown();
        }
      }
    }
    
  private:
    rclcpp::Node::SharedPtr node_;
    std::shared_ptr<okvis::threadsafe::Queue<std::shared_ptr<MessageType>>>
        queue_;
    std::function<void()> notifyCallback_;
    typename rclcpp::Publisher<MessageType>::SharedPtr publisher_ = nullptr;
    std::shared_ptr<image_transport::ImageTransport> imageTransport_ = nullptr;
    std::shared_ptr<image_transport::Publisher> imagePublisher_ = nullptr;
    std::string name_;
  };

public:
  /**
   * @brief Constructor.
   * @param node Shared pointer to the ROS2 node.
   */
  ThreadedPublisher(rclcpp::Node::SharedPtr node) : node_(node) {}

  /**
   * @brief Destructor
   */
  ~ThreadedPublisher() { shutdown(); }

  /**
   * @brief Start the internal background publishing thread.
   */
  void startThread() {
    running_ = true;
    publisherThread_ = std::thread(&ThreadedPublisher::publisherWorker, this);
  }

  /**
   * @brief Lightweight handle to interact with a registered publisher.
   */
  template <typename MsgIn>
  struct PublisherHandle {
    std::function<void(const std::shared_ptr<MsgIn>)> publish;
  };

  /**
   * @brief Register a new publisher for a given topic.
   *
   * @tparam MessageType The message type to be published.
   * @tparam PreprocParamType The parameter type passed to the preprocessing function.
   * @param name The topic name to advertise on.
   * @param preprocessingFunction A preprocessing function returning the message
   * to be published.
   * @return A PublisherHandle with publish member that stores the latest message to be published.
   */
  template <typename MessageType, typename PreprocParamType = MessageType>
  PublisherHandle<PreprocParamType>
  registerPublisher(const std::string name,
                    std::function<std::shared_ptr<MessageType>(
                        std::shared_ptr<PreprocParamType>)>
                        preprocessingFunction = nullptr) {
    auto newPub = std::make_shared<PublisherTyped<MessageType>>(
        name, node_, [this]() { this->notifyDataAvailable(); });
    publishers_.push_back(newPub);

    if constexpr (!std::is_same_v<MessageType, PreprocParamType>) {
      static_assert(
          !std::is_same_v<std::nullptr_t, decltype(preprocessingFunction)>,
          "No preprocessing function provided.");
      return PublisherHandle<PreprocParamType>{
        [newPub, preprocessingFunction](std::shared_ptr<PreprocParamType> msg) {
          auto preprocMsg = preprocessingFunction(msg);
          return newPub->setLatestMessage(preprocMsg);
        }
      };
    } else {
      return PublisherHandle<PreprocParamType>{
        [newPub](std::shared_ptr<PreprocParamType> msg) {
          return newPub->setLatestMessage(msg);
        }
      };
    }
  }

  /**
   * @brief Notify the background thread that a new message is available.
   */
  void notifyDataAvailable() {
    {
      std::lock_guard<std::mutex> lock(wakeMtx_);
      dataAvailableToPublish_ = true;
    }
    cv_.notify_one();
  }

  /**
   * @brief Shut down the background thread and all active publishers.
   */
  void shutdown() {
    running_ = false;

    // Wake up waiting threads.
    cv_.notify_one();

    if (publisherThread_.joinable()) {
      publisherThread_.join();
    }
    for (auto &pub : publishers_) {
      pub->shutdown();
    }
  }

private:
  /**
   * @brief Background worker that publishes all available latest messages.
   */
  void publisherWorker() {
    while (running_) {
      {
        std::unique_lock lock(wakeMtx_);
        cv_.wait(lock, [this] { return dataAvailableToPublish_ || !running_; });
      }

      for (auto &pub : publishers_) {
        pub->publishLatest();
      }
      dataAvailableToPublish_ = false;
    }
  }

  std::atomic<bool> running_{false};
  rclcpp::Node::SharedPtr node_;
  std::thread publisherThread_;
  std::vector<std::shared_ptr<PublisherBase>> publishers_;
  std::mutex wakeMtx_;
  std::condition_variable cv_;
  std::atomic<bool> dataAvailableToPublish_{false};
};

} // namespace okvis

#endif // OKVIS_THREADEDPUBLISHER_HPP_
