#include "line_follow/navigation.hpp"
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/ximgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <lifecycle_msgs/msg/state.hpp>
#include <functional>
#include <unordered_map>
#include <cv_bridge/cv_bridge.hpp>
#include <std_msgs/msg/float64.hpp>

using std::placeholders::_1;

NavigationNode::NavigationNode()
: rclcpp_lifecycle::LifecycleNode("navigation")
{
  this->declare_parameter<std::string>("navigation_type", "simple");

  std::string nav_type_str = this->get_parameter("navigation_type").as_string();

  static const std::unordered_map<std::string, NavigationType> nav_type_map = {
    {"simple", NavigationType::SIMPLE},
    {"advanced", NavigationType::ADVANCED}
  };

  auto it = nav_type_map.find(nav_type_str);

  if (it != nav_type_map.end()) {
    this->navigationType = it->second;
  } else {
    RCLCPP_ERROR(this->get_logger(), "Unknown navigation type! Defaulting to simple.");
    this->navigationType = NavigationType::SIMPLE;
  }

  int fourcc = cv::VideoWriter::fourcc('x', 'v', 'i', 'd');
  cv::Size frameSize = cv::Size(1536 * 0.8, 864 * 0.6);
  double fps = 30.0;

  this->writer = cv::VideoWriter("/videos/output.avi", cv::CAP_GSTREAMER, fourcc,
    fps,
    frameSize);

  if (!this->writer.isOpened()) {
    std::cerr << "CRITICAL ERROR: VideoWriter failed to initiate!" << std::endl;
    std::cerr << "Check 1: Does OpenCV have FFMPEG? Build Info: " << cv::getBuildInformation() << std::endl;
}
}

CallbackReturn NavigationNode::on_configure(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(this->get_logger(), "Configuring...");
  this->errorPub = this->create_publisher<std_msgs::msg::Float64>("line_error", 10);
  this->imageSub =
    this->create_subscription<sensor_msgs::msg::Image>("/down_camera/camera_node/image_raw", 10,
    std::bind(&NavigationNode::imageCallback, this, _1));

  return CallbackReturn::SUCCESS;
}

CallbackReturn NavigationNode::on_activate(const rclcpp_lifecycle::State & state)
{
  this->errorPub->on_activate();

  return rclcpp_lifecycle::LifecycleNode::on_activate(state);
}

CallbackReturn NavigationNode::on_deactivate(const rclcpp_lifecycle::State & state)
{
  this->errorPub->on_deactivate();

  return rclcpp_lifecycle::LifecycleNode::on_deactivate(state);
}

CallbackReturn NavigationNode::on_cleanup(const rclcpp_lifecycle::State &)
{
  this->errorPub.reset();
  this->imageSub.reset();

  return CallbackReturn::SUCCESS;
}

CallbackReturn NavigationNode::on_shutdown(const rclcpp_lifecycle::State &)
{
  return CallbackReturn::SUCCESS;
}

void NavigationNode::imageCallback(sensor_msgs::msg::Image::SharedPtr msg)
{
  if (this->get_current_state().id() != lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE) {
    return;
  }

  cv_bridge::CvImagePtr cv_ptr;

  try {
    cv_ptr = cv_bridge::toCvCopy(msg);
  } catch (cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
  }

  cv::Mat frame = cv_ptr->image;

  switch (this->navigationType) {
    case NavigationType::SIMPLE:
      this->simpleNavigation(frame);
      break;
    case NavigationType::ADVANCED:
      this->advancedNavigation(frame);
      break;
  }
}

void NavigationNode::publishError(double error)
{
  // RCLCPP_INFO(this->get_logger(), "Publishing error: %f", error);
  std_msgs::msg::Float64 msg = std_msgs::msg::Float64();
  msg.data = error;
  this->errorPub->publish(msg);
}

double NavigationNode::simpleError(const cv::Mat & frame)
{
  int newWidth = static_cast<int>(frame.cols * 0.8);
  int newHeight = static_cast<int>(frame.rows * 0.6);

    // 2. Calculate coordinates to center the crop box
  int x = (frame.cols - newWidth) / 2;
  int y = (frame.rows - newHeight) / 2;

    // 3. Define the Region of Interest (ROI) and crop
  cv::Rect roi(x, y, newWidth, newHeight);
  cv::Mat croppedImg = frame(roi);
  cv::Mat gray, thresh;

    // 1. Convert to grayscale if it's a color image
  if (croppedImg.channels() == 3) {
    cv::cvtColor(croppedImg, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = croppedImg.clone();
  }

    // 2. Threshold the image to isolate the line (adjust threshold value as needed)
    // Using THRESH_BINARY_INV assuming a dark line on a light background.
    // Use cv::THRESH_BINARY if it's a bright line on a dark background.
  cv::threshold(gray, thresh, 80, 255, cv::THRESH_BINARY_INV);
  // cv::adaptiveThreshold(gray, thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV,
  //   101, 2);

    // 3. Find contours
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // If no contours are found, return 0 error
  if (contours.empty()) {
    return 0.0;
  }

    // 4. Find the largest contour (assuming this is our line)
  size_t largestContourIdx = 0;
  double maxArea = 0.0;
  for (size_t i = 0; i < contours.size(); ++i) {
    double area = cv::contourArea(contours[i]);
    if (area > maxArea) {
      maxArea = area;
      largestContourIdx = i;
    }
  }

    // Optional: Filter out tiny noise
  if (maxArea < 100.0) {
    return 0.0;
  }

    // 5. Calculate the Center of Mass (Centroid) using Moments
  cv::Moments m = cv::moments(contours[largestContourIdx]);

    // Prevent division by zero
  if (m.m00 == 0) {return 0.0;}

    // Centroid X coordinate formula: X = M10 / M00
  double lineCenterX = (m.m10 / m.m00) + x;

    // 6. Calculate the error from the screen center
  double screenCenterX = thresh.cols / 2.0;
  double error = lineCenterX - screenCenterX;

  cv::Mat processed;
  cv::cvtColor(thresh, processed, cv::COLOR_GRAY2BGR);
  this->writer.write(processed);

  return error;

}

void NavigationNode::simpleNavigation(cv::Mat & frame)
{
  double error = this->simpleError(frame);

  this->publishError(error / 200);
}

void NavigationNode::advancedNavigation(cv::Mat & frame)
{
  cv::Mat processed = this->processImage(frame);
}

cv::Mat NavigationNode::processImage(cv::Mat & image)
{
  cv::Mat resized;
  cv::Size dsize(200, 100);
  cv::resize(image, resized, dsize);

  cv::Mat gray, binary;
  cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(gray, gray, cv::Size(5, 5), 1.5);

  cv::adaptiveThreshold(gray, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV,
    11, 2);

  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

  cv::Mat opened_image;
  cv::morphologyEx(binary, opened_image, cv::MORPH_OPEN, kernel);

  cv::Mat closed_image;
  cv::morphologyEx(opened_image, closed_image, cv::MORPH_CLOSE, kernel);

  cv::Mat skeleton;
  cv::ximgproc::thinning(closed_image, skeleton,
                         cv::ximgproc::THINNING_GUOHALL);

  int rows = skeleton.rows;
  int cols = skeleton.cols;

  cv::Rect top_border_roi(0, 0, cols, 1);
  skeleton(top_border_roi).setTo(0);

  // Set bottom border
  cv::Rect bottom_border_roi(0, rows - 1, cols, 1);
  skeleton(bottom_border_roi).setTo(0);

  // Set left border (excluding corners already set)
  cv::Rect left_border_roi(0, 1, 1, rows - 2);
  skeleton(left_border_roi).setTo(0);

  // Set right border (excluding corners already set)
  cv::Rect right_border_roi(cols - 1, 1, 1, rows - 2);
  skeleton(right_border_roi).setTo(0);

  return skeleton;
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<NavigationNode>();
  rclcpp::spin(node->get_node_base_interface());
  rclcpp::shutdown();
  return 0;
}
