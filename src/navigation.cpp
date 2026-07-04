/**
 * Copyright (C) 2026  William D'Olier
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "line_follow/navigation.hpp"
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/ximgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include "nav_msgs/msg/odometry.hpp"
#include "robot_msgs/action/move_time.hpp"
#include <lifecycle_msgs/msg/state.hpp>
#include <chrono>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <cv_bridge/cv_bridge.hpp>
#include <std_msgs/msg/float64.hpp>
#include <numbers>
#include <vector>
#include <optional>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <Hungarian.h>

using std::placeholders::_1;
using std::placeholders::_2;
using namespace std::chrono_literals;

NavigationNode::NavigationNode(const rclcpp::NodeOptions & options)
: rclcpp_lifecycle::LifecycleNode("navigation", options)
{
  this->declare_parameter<std::string>("navigation_type", "simple");
  this->declare_parameter<int>("path_limit", 5);
  this->declare_parameter<int>("min_edge_size", 25);
  this->declare_parameter<int>("gating_threshold", 50);
  this->declare_parameter<int>("search_min_dist", 20);
  this->declare_parameter<double>("pixel_size", 0.01);

  this->pathLimit = this->get_parameter("path_limit").as_int();
  this->minEdgeSize = this->get_parameter("min_edge_size").as_int();
  this->gatingThreshold = this->get_parameter("gating_threshold").as_int();
  this->searchMinDist = this->get_parameter("search_min_dist").as_int();
  this->pixelSize = this->get_parameter("pixel_size").as_double();
  this->frameCentre = cv::Point(100, 70);

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

  int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
  cv::Size frameSize = cv::Size(854, 480);
  double fps = 30.0;

  this->writer = cv::VideoWriter("/videos/output.mp4", fourcc,
    fps,
    frameSize);

  if (!this->writer.isOpened()) {
    RCLCPP_ERROR(this->get_logger(), "CRITICAL ERROR: VideoWriter failed to initiate!");
    RCLCPP_ERROR(this->get_logger(), "Check 1: Does OpenCV have FFMPEG? Build Info: %s",
      cv::getBuildInformation().c_str());
  }
}

CallbackReturn NavigationNode::on_configure(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(this->get_logger(), "Configuring...");
  auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();

  this->errorPub = this->create_publisher<std_msgs::msg::Float64>("line_error", 10);
  this->lineCompletePub = this->create_publisher<std_msgs::msg::Bool>("rescue_active", 10);
  this->imageSub =
    this->create_subscription<sensor_msgs::msg::Image>("/down_camera/camera_node/image_raw", qos,
    std::bind(&NavigationNode::imageCallback, this, _1));
  this->odomSub =
    this->create_subscription<nav_msgs::msg::Odometry>("/odom", 10,
    std::bind(&NavigationNode::odomCallback, this, _1));

  timer_ = this->create_wall_timer(33ms, std::bind(&NavigationNode::timerCallback, this));
  timer_->cancel();
  this->actionClient = rclcpp_action::create_client<robot_msgs::action::MoveTime>(
      this,
      "/move_time" // Must match the server's action name
  );

  return CallbackReturn::SUCCESS;
}

CallbackReturn NavigationNode::on_activate(const rclcpp_lifecycle::State & state)
{
  this->errorPub->on_activate();
  this->lineCompletePub->on_activate();
  timer_->reset();

  this->state = FOLLOWING;

  return rclcpp_lifecycle::LifecycleNode::on_activate(state);
}

CallbackReturn NavigationNode::on_deactivate(const rclcpp_lifecycle::State & state)
{
  this->errorPub->on_deactivate();
  this->lineCompletePub->on_deactivate();
  timer_->cancel();

  return rclcpp_lifecycle::LifecycleNode::on_deactivate(state);
}

CallbackReturn NavigationNode::on_cleanup(const rclcpp_lifecycle::State &)
{
  this->errorPub.reset();
  this->imageSub.reset();
  this->odomSub.reset();
  timer_->reset();

  return CallbackReturn::SUCCESS;
}

CallbackReturn NavigationNode::on_shutdown(const rclcpp_lifecycle::State &)
{
  return CallbackReturn::SUCCESS;
}

TrackedNode::TrackedNode(cv::Point pos)
{
  float dt = 1.0f;
  kf.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, dt, 0, 0, 1, 0, dt, 0,
    0, 1, 0, 0, 0, 0, 1);

  kf.measurementMatrix = (cv::Mat_<float>(2, 4) << 1, 0, 0, 0, 0, 1, 0, 0);

  cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-4));

  cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-2));

  cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1));

  kf.statePost.at<float>(1) = pos.x; // Initial X
  kf.statePost.at<float>(1) = pos.y; // Initial Y
  kf.statePost.at<float>(2) = 0.0f;  // Initial velocity X
  kf.statePost.at<float>(3) = 0.0f;  // Initial velocity Y

  this->pos = pos;
}

void NavigationNode::sendMovementGoal(double vel, double angular_vel, double time)
{
  if (!this->actionClient->wait_for_action_server(std::chrono::seconds(10))) {
    RCLCPP_ERROR(this->get_logger(), "Action server not available.");
    return;
  }

  auto goalMsg = robot_msgs::action::MoveTime::Goal();
  goalMsg.vel = vel;
  goalMsg.angular_vel = angular_vel;
  goalMsg.time = time;

  auto sendGoalOptions = rclcpp_action::Client<robot_msgs::action::MoveTime>::SendGoalOptions();

  sendGoalOptions.goal_response_callback =
    std::bind(&NavigationNode::goalResponseCallback, this, _1);
  sendGoalOptions.feedback_callback = std::bind(&NavigationNode::goalFeedbackCallback, this, _1,
    _2);
  sendGoalOptions.result_callback =
    std::bind(&NavigationNode::goalResultCallback, this, _1);

  this->actionClient->async_send_goal(goalMsg, sendGoalOptions);
}

void NavigationNode::goalResponseCallback(
  const rclcpp_action::ClientGoalHandle<robot_msgs::action::MoveTime>::SharedPtr & goalHandle)
{
  if (!goalHandle) {
    RCLCPP_ERROR(this->get_logger(), "Movement goal rejected!");
  } else {
    RCLCPP_INFO(this->get_logger(), "Movement goal accepted!");
  }
}

void NavigationNode::goalFeedbackCallback(
  rclcpp_action::ClientGoalHandle<robot_msgs::action::MoveTime>::SharedPtr,
  const std::shared_ptr<const robot_msgs::action::MoveTime::Feedback> _) {}

void NavigationNode::goalResultCallback(
  const rclcpp_action::ClientGoalHandle<robot_msgs::action::MoveTime>::WrappedResult & result)
{
  switch (result.code) {
    case rclcpp_action::ResultCode::SUCCEEDED:
      RCLCPP_INFO(this->get_logger(), "Goal succeeded!");
      break;
    case rclcpp_action::ResultCode::ABORTED:
      RCLCPP_ERROR(this->get_logger(), "Goal was aborted");
      return;
    case rclcpp_action::ResultCode::CANCELED:
      RCLCPP_WARN(this->get_logger(), "Goal was canceled");
      return;
    default:
      RCLCPP_ERROR(this->get_logger(), "Unknown result code");
      return;
  }
  switch (this->state) {
    case FOLLOWING:
      break;
    case TOWER_ROTATE_START:
      this->sendMovementGoal(100, 100, 10);
      this->state = TOWER_MOVE;
      break;
    case TOWER_MOVE:
      this->sendMovementGoal(100, 100, 10);
      this->state = TOWER_ROTATE_END;
      break;
    case TOWER_ROTATE_END:
      this->sendMovementGoal(100, 100, 10);
      this->state = FOLLOWING;
      break;
    case GREEN_ROTATE:
      this->sendMovementGoal(100, 0, 0.2);
      this->state = GREEN_MOVE_FORWARD;
      break;
    case GREEN_MOVE_FORWARD:
      this->state = FOLLOWING;
      break;
    case COMPLETE:
      break;
  }
}

std::vector<std::vector<double>> TrackedGraph::getCostMatrix(Graph & graph)
{
  // TODO apply kalman filter

  // The vector must be square
  int size = this->nodes.size() > graph.nodes.size() ? this->nodes.size() :
    graph.nodes.size();

  std::vector<std::vector<double>> costs(size, std::vector(size, 0.0));

  for (uint32_t i = 0; i < this->nodes.size(); i++) {
    for (uint32_t j = 0; j < graph.nodes.size(); j++) {
      Node & newNode = graph.nodes[j];
      Node & trackedNode = this->nodes[i];

      double distance = cv::norm(trackedNode.pos - newNode.pos);
      int connectedEdgeDiff = graph.getConnectedEdges(newNode.id).size() -
        this->getConnectedEdges(trackedNode.id).size();

      double penalty = trackedNode.screen_edge ?
        0 :
        this->edgePenalty * std::abs(connectedEdgeDiff);

      double cost = distance + penalty;
      costs[i][j] = cost;
    }
  }

  return costs;
}

void NavigationNode::imageCallback(sensor_msgs::msg::Image::SharedPtr msg)
{
  if (this->get_current_state().id() != lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE) {
    return;
  }

  cv_bridge::CvImageConstPtr cv_ptr;

  try {
    cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
  }

  cv::Mat frame = cv_ptr->image;

  this->currentFrame = frame;
}

void NavigationNode::timerCallback()
{
  switch (this->state) {
    case FOLLOWING: {
        if (this->currentFrame.empty()) {return;}

        switch (this->navigationType) {
          case NavigationType::SIMPLE:
            this->simpleNavigation(this->currentFrame);
            break;
          case NavigationType::ADVANCED:
            this->advancedNavigation(this->currentFrame);
            break;
        }

        break;
      }
    case TOWER_ROTATE_START:
      break;
    case TOWER_MOVE:
      break;
    case TOWER_ROTATE_END:
      break;
    case GREEN_ROTATE:
      break;
    case GREEN_MOVE_FORWARD:
      break;
    case COMPLETE:
      break;
  }
}

void NavigationNode::odomCallback(nav_msgs::msg::Odometry::SharedPtr msg)
{
  this->x = msg->pose.pose.position.x;
  this->y = -msg->pose.pose.position.y;
  this->angle = msg->pose.pose.position.z;
}

void NavigationNode::publishError(double error)
{
  std_msgs::msg::Float64 msg = std_msgs::msg::Float64();
  msg.data = error * 350;
  this->errorPub->publish(msg);
}

// ============================================================================
// Ported line-following vision pipeline
// (from line_follower_vision.py's LineFollowerVision.process() and its
// step-by-step helpers). Used exclusively by simpleError() below.
// ============================================================================

// STEP 1: BINARIZE  (LineFollowerVision._binarize)
cv::Mat NavigationNode::visionBinarize(const cv::Mat & frameBgr)
{
  cv::Mat gray, blurred, binary;
  cv::cvtColor(frameBgr, gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

  if (VISION_USE_OTSU) {
    cv::threshold(blurred, binary, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);
  } else {
    cv::threshold(blurred, binary, VISION_FIXED_BINARY_THRESH, 255, cv::THRESH_BINARY_INV);
  }

  return binary;  // line pixels == 255
}

// LineFollowerVision._find_segments_in_row
std::vector<StripSegment> NavigationNode::visionFindSegmentsInRow(
  const std::vector<uint8_t> & rowBool)
{
  std::vector<StripSegment> segments;
  bool inRun = false;
  int start = 0;

  for (int x = 0; x < static_cast<int>(rowBool.size()); x++) {
    bool val = rowBool[x] != 0;
    if (val && !inRun) {
      inRun = true;
      start = x;
    } else if (!val && inRun) {
      inRun = false;
      int width = x - start;
      if (width >= VISION_MIN_SEGMENT_WIDTH_PX) {
        segments.push_back({start, x - 1, (start + x - 1) / 2.0, width});
      }
    }
  }

  if (inRun) {
    int end = static_cast<int>(rowBool.size()) - 1;
    int width = end - start + 1;
    if (width >= VISION_MIN_SEGMENT_WIDTH_PX) {
      segments.push_back({start, end, (start + end) / 2.0, width});
    }
  }

  return segments;
}

// STEP 2: MULTI-STRIP SCAN  (LineFollowerVision._scan_strips)
std::vector<ScanStrip> NavigationNode::visionScanStrips(const cv::Mat & binary, int h, int w)
{
  int yTop = static_cast<int>(h * VISION_ROI_Y_TOP_RATIO);
  int yBottom = static_cast<int>(h * VISION_ROI_Y_BOTTOM_RATIO);

  std::vector<ScanStrip> strips;
  int half = VISION_STRIP_THICKNESS_PX / 2;

  for (int i = 0; i < VISION_NUM_STRIPS; i++) {
    double t = (VISION_NUM_STRIPS == 1) ?
      0.0 : static_cast<double>(i) / (VISION_NUM_STRIPS - 1);
    int y = static_cast<int>(std::round(yTop + t * (yBottom - yTop)));

    int y0 = std::max(0, y - half);
    int y1 = std::min(h, y + half + 1);

    cv::Mat band = binary(cv::Range(y0, y1), cv::Range(0, w));

    // Collapse the band's thickness: a column counts as foreground if the
    // majority of rows in the band are foreground.
    std::vector<uint8_t> collapsed(w, 0);
    for (int col = 0; col < w; col++) {
      double sum = 0.0;
      for (int row = 0; row < band.rows; row++) {
        sum += band.at<uint8_t>(row, col);
      }
      double mean = sum / band.rows;
      collapsed[col] = (mean > 127) ? 1 : 0;
    }

    ScanStrip strip;
    strip.y = y;
    strip.segments = this->visionFindSegmentsInRow(collapsed);
    strip.is_branch = false;
    strip.is_branch_strong = false;
    strips.push_back(strip);
  }

  return strips;
}

// LineFollowerVision._estimate_line_width
double NavigationNode::visionEstimateLineWidth(const std::vector<ScanStrip> & strips)
{
  std::vector<double> widths;
  for (const ScanStrip & s : strips) {
    if (s.segments.size() == 1) {
      widths.push_back(static_cast<double>(s.segments[0].width));
    }
  }

  if (static_cast<int>(widths.size()) >= VISION_BASELINE_STRIP_COUNT) {
    std::sort(widths.begin(), widths.end());
    size_t n = widths.size();
    double median = (n % 2 == 0) ?
      (widths[n / 2 - 1] + widths[n / 2]) / 2.0 :
      widths[n / 2];
    return median;
  }

  return VISION_DEFAULT_LINE_WIDTH_PX;
}

// LineFollowerVision._flag_branch_strips
void NavigationNode::visionFlagBranchStrips(std::vector<ScanStrip> & strips, double baselineWidth)
{
  for (ScanStrip & s : strips) {
    bool tooMany = static_cast<int>(s.segments.size()) >= VISION_BRANCH_MIN_SEGMENTS;

    bool tooWide = false;
    bool veryWide = false;
    for (const StripSegment & seg : s.segments) {
      if (seg.width > baselineWidth * VISION_WIDE_SEGMENT_MULTIPLIER) {
        tooWide = true;
      }
      if (seg.width > baselineWidth * VISION_STRONG_WIDE_SEGMENT_MULTIPLIER) {
        veryWide = true;
      }
    }

    s.is_branch = tooMany || tooWide;      // weak signal (needs corroboration)
    s.is_branch_strong = veryWide;         // strong signal (stands alone)
  }
}

// STEP 2b: JUNCTION DETECTION  (LineFollowerVision._detect_junction)
std::optional<JunctionInfo> NavigationNode::visionDetectJunction(
  std::vector<ScanStrip> & strips, const cv::Mat & binary)
{
  std::vector<int> branchIdx;
  for (int i = 0; i < static_cast<int>(strips.size()); i++) {
    if (strips[i].is_branch) {
      branchIdx.push_back(i);
    }
  }

  if (branchIdx.empty()) {
    return std::nullopt;
  }

  // Cluster branch-flagged strips by index proximity so an isolated,
  // spatially-unrelated false-positive strip doesn't get averaged into a
  // real junction found elsewhere.
  std::vector<std::vector<int>> clusters;
  std::vector<int> current = {branchIdx[0]};
  for (size_t k = 1; k < branchIdx.size(); k++) {
    int idx = branchIdx[k];
    if (idx - current.back() <= VISION_BRANCH_CLUSTER_GAP_STRIPS) {
      current.push_back(idx);
    } else {
      clusters.push_back(current);
      current = {idx};
    }
  }
  clusters.push_back(current);

  // A cluster is a valid junction if it has a strong strip on its own, or
  // enough weak strips corroborate each other.
  std::vector<std::vector<int>> validClusters;
  for (const std::vector<int> & c : clusters) {
    bool hasStrong = false;
    for (int i : c) {
      if (strips[i].is_branch_strong) {
        hasStrong = true;
        break;
      }
    }
    if (hasStrong || static_cast<int>(c.size()) >= VISION_BRANCH_MIN_STRIP_COUNT) {
      validClusters.push_back(c);
    }
  }

  if (validClusters.empty()) {
    return std::nullopt;
  }

  const std::vector<int> * bestCluster = &validClusters[0];
  for (const std::vector<int> & c : validClusters) {
    if (c.size() > bestCluster->size()) {
      bestCluster = &c;
    }
  }

  std::vector<int> clusterYs;
  for (int i : *bestCluster) {
    clusterYs.push_back(strips[i].y);
  }

  int stripSpacing = (strips.size() >= 2) ?
    (strips[1].y - strips[0].y) : VISION_STRIP_THICKNESS_PX;

  int minY = *std::min_element(clusterYs.begin(), clusterYs.end());
  int maxY = *std::max_element(clusterYs.begin(), clusterYs.end());

  // Crop the mask to the cluster's y-span (padded by one strip spacing on
  // each side) and pull out the largest connected foreground blob: a
  // pixel-mass centroid tolerates an angled approach far better than
  // averaging per-row segment centers.
  int y0 = std::max(0, minY - stripSpacing);
  int y1 = std::min(binary.rows, maxY + stripSpacing + 1);

  cv::Mat band = binary(cv::Range(y0, y1), cv::Range(0, binary.cols));

  cv::Mat labels, stats, centroids;
  int numLabels = cv::connectedComponentsWithStats(band, labels, stats, centroids, 8);

  if (numLabels <= 1) {
    return std::nullopt;
  }

  // label 0 is background; pick the largest foreground component.
  int largestLabel = 1;
  int largestArea = stats.at<int>(1, cv::CC_STAT_AREA);
  for (int lbl = 2; lbl < numLabels; lbl++) {
    int area = stats.at<int>(lbl, cv::CC_STAT_AREA);
    if (area > largestArea) {
      largestArea = area;
      largestLabel = lbl;
    }
  }

  double cx = centroids.at<double>(largestLabel, 0);
  double cy = centroids.at<double>(largestLabel, 1) + y0;  // back to full-frame coords

  int bx = stats.at<int>(largestLabel, cv::CC_STAT_LEFT);
  int by = stats.at<int>(largestLabel, cv::CC_STAT_TOP);
  int bw = stats.at<int>(largestLabel, cv::CC_STAT_WIDTH);
  int bh = stats.at<int>(largestLabel, cv::CC_STAT_HEIGHT);

  JunctionInfo junction;
  junction.center = cv::Point2d(cx, cy);
  junction.box = cv::Rect2d(bx, by + y0, bw, bh);

  return junction;
}

// STEP 3: STEERING LINE FIT  (LineFollowerVision._fit_steering_line)
void NavigationNode::visionFitSteeringLine(
  const std::vector<ScanStrip> & strips, bool haveJunction, int w,
  double & m, double & b, std::vector<cv::Point2d> & fitPoints)
{
  (void)haveJunction;  // kept for parity with the python call site; unused directly

  // Fit using the non-branch strips (the clean approach line); fall back to
  // every strip if none qualify.
  std::vector<const ScanStrip *> candidateStrips;
  for (const ScanStrip & s : strips) {
    if (!s.is_branch) {
      candidateStrips.push_back(&s);
    }
  }
  if (candidateStrips.empty()) {
    for (const ScanStrip & s : strips) {
      candidateStrips.push_back(&s);
    }
  }

  // Per strip, pick the segment closest to image-center as the "primary"
  // line segment (a stand-in for frame-to-frame tracking).
  double priorX = w / 2.0;
  std::vector<double> ptsX, ptsY;

  for (const ScanStrip * s : candidateStrips) {
    if (s->segments.empty()) {
      continue;
    }

    const StripSegment * best = &s->segments[0];
    double bestDiff = std::abs(best->x_center - priorX);
    for (const StripSegment & seg : s->segments) {
      double diff = std::abs(seg.x_center - priorX);
      if (diff < bestDiff) {
        bestDiff = diff;
        best = &seg;
      }
    }

    ptsX.push_back(best->x_center);
    ptsY.push_back(static_cast<double>(s->y));
  }

  fitPoints.clear();
  for (size_t i = 0; i < ptsX.size(); i++) {
    fitPoints.push_back(cv::Point2d(ptsX[i], ptsY[i]));
  }

  if (static_cast<int>(ptsX.size()) < VISION_MIN_FIT_POINTS) {
    m = 0.0;
    b = w / 2.0;
    return;
  }

  // Least-squares fit of x = m*y + b (np.polyfit(pts_y, pts_x, 1) equivalent).
  double n = static_cast<double>(ptsX.size());
  double sumY = 0.0, sumX = 0.0, sumYY = 0.0, sumXY = 0.0;
  for (size_t i = 0; i < ptsX.size(); i++) {
    sumY += ptsY[i];
    sumX += ptsX[i];
    sumYY += ptsY[i] * ptsY[i];
    sumXY += ptsX[i] * ptsY[i];
  }

  double denom = n * sumYY - sumY * sumY;
  if (std::abs(denom) < 1e-9) {
    m = 0.0;
    b = sumX / n;
    return;
  }

  m = (n * sumXY - sumY * sumX) / denom;
  b = (sumX - m * sumY) / n;
}

// LineFollowerVision._steering_angle
double NavigationNode::visionSteeringAngle(double m)
{
  // direction of travel vector as y decreases (moving up-frame): (dx, dy) = (-m, -1)
  // angle measured from "straight ahead" (0,-1); positive = should turn right
  return std::atan2(-m, 1.0) * 180.0 / std::numbers::pi;
}

// LineFollowerVision._line_offset
double NavigationNode::visionLineOffset(double m, double b, int h, int w)
{
  double xAtBottom = m * h + b;
  return xAtBottom - w / 2.0;
}

// LineFollowerVision._heading_vectors
void NavigationNode::visionHeadingVectors(double m, cv::Point2d & forward, cv::Point2d & right)
{
  double fx = -m;
  double fy = -1.0;
  double norm = std::hypot(fx, fy);
  if (norm < 1e-9) {
    norm = 1.0;
  }
  forward = cv::Point2d(fx / norm, fy / norm);
  // right_vector = forward rotated -90 degrees (clockwise) in image coords
  right = cv::Point2d(-forward.y, forward.x);
}

// STEP 4a: GREEN BLOB DETECTION  (LineFollowerVision._detect_green_blobs)
std::vector<GreenBlobInfo> NavigationNode::visionDetectGreenBlobs(const cv::Mat & frameBgr)
{
  cv::Mat hsv;
  cv::cvtColor(frameBgr, hsv, cv::COLOR_BGR2HSV);

  cv::Scalar lowerGreen(40, 60, 60);
  cv::Scalar upperGreen(85, 255, 255);

  cv::Mat mask;
  cv::inRange(hsv, lowerGreen, upperGreen, mask);

  cv::Mat kernel = cv::getStructuringElement(
    cv::MORPH_ELLIPSE, cv::Size(VISION_MORPH_KERNEL_SIZE, VISION_MORPH_KERNEL_SIZE));
  cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
  cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  std::vector<GreenBlobInfo> blobs;
  for (const std::vector<cv::Point> & c : contours) {
    double area = cv::contourArea(c);
    if (area < VISION_MIN_GREEN_BLOB_AREA) {
      continue;
    }

    cv::Moments mo = cv::moments(c);
    if (mo.m00 == 0) {
      continue;
    }

    GreenBlobInfo blob;
    blob.center = cv::Point2d(mo.m10 / mo.m00, mo.m01 / mo.m00);
    blob.area = area;
    blob.contour = c;
    blobs.push_back(blob);
  }

  return blobs;
}

// STEP 4b: GREEN BLOB CLASSIFICATION  (LineFollowerVision._classify_blobs)
void NavigationNode::visionClassifyBlobs(
  std::vector<GreenBlobInfo> & blobs, const std::optional<JunctionInfo> & junction,
  const cv::Point2d & forwardVec, const cv::Point2d & rightVec, double baselineWidth)
{
  double nearForwardLimit = baselineWidth * VISION_NEAR_FORWARD_LIMIT_LINE_WIDTHS;
  double maxLateralDist = baselineWidth * VISION_MAX_LATERAL_DIST_LINE_WIDTHS;

  for (GreenBlobInfo & blob : blobs) {
    if (!junction.has_value()) {
      blob.region = "no_junction";
      blob.side = "";
      continue;
    }

    double relX = blob.center.x - junction->center.x;
    double relY = blob.center.y - junction->center.y;

    double forwardProj = relX * forwardVec.x + relY * forwardVec.y;
    double lateralProj = relX * rightVec.x + relY * rightVec.y;

    blob.forward_proj = forwardProj;
    blob.lateral_proj = lateralProj;

    if (std::abs(lateralProj) > maxLateralDist) {
      // Guards against unrelated green blobs elsewhere in frame.
      blob.region = "unrelated";
      blob.side = "";
    } else if (forwardProj > nearForwardLimit) {
      // Sits well past the junction center: a corner that does not touch
      // the arm we're arriving on -> ignore for decision purposes.
      blob.region = "far";
      blob.side = (lateralProj > 0) ? "right" : "left";
    } else {
      blob.region = "near";
      blob.side = (lateralProj > 0) ? "right" : "left";
    }
  }
}

// STEP 5: DECISION FUSION  (LineFollowerVision._decide)
void NavigationNode::visionDecide(
  const std::optional<JunctionInfo> & junction,
  const std::vector<GreenBlobInfo> & classified,
  std::string & decision, std::string & decisionReason)
{
  if (!junction.has_value()) {
    decision = "STRAIGHT";
    decisionReason = "no junction detected, following line";
    return;
  }

  bool left = false;
  bool right = false;
  for (const GreenBlobInfo & b : classified) {
    if (b.region == "near") {
      if (b.side == "left") {left = true;}
      if (b.side == "right") {right = true;}
    }
  }

  if (left && right) {
    decision = "TURN_180";
    decisionReason = "green markers on both sides of the incoming arm";
  } else if (left) {
    decision = "TURN_LEFT";
    decisionReason = "green marker on left side of incoming arm";
  } else if (right) {
    decision = "TURN_RIGHT";
    decisionReason = "green marker on right side of incoming arm";
  } else {
    decision = "STRAIGHT";
    decisionReason = "junction present but no relevant markers on this arm";
  }
}

// ANNOTATION / DEBUG DRAWING  (LineFollowerVision._annotate)
cv::Mat NavigationNode::visionAnnotate(
  const cv::Mat & frameBgr, const std::vector<ScanStrip> & strips, double baselineWidth,
  const std::optional<JunctionInfo> & junction, double m, double b,
  const std::vector<cv::Point2d> & fitPoints, double steeringAngleDeg,
  const std::vector<GreenBlobInfo> & classified, const std::string & decision,
  const std::string & decisionReason, const cv::Point2d & forwardVec)
{
  (void)baselineWidth;  // not needed for drawing; kept for parity with the python signature

  cv::Mat out = frameBgr.clone();
  int h = out.rows;
  int w = out.cols;

  const cv::Scalar colorLineNormal(0, 255, 255);   // yellow: non-branch strip segment
  const cv::Scalar colorLineBranch(0, 128, 255);   // orange: branch strip segment
  const cv::Scalar colorSteer(255, 0, 255);        // magenta: fitted steering line
  const cv::Scalar colorJunction(255, 0, 0);       // blue: junction box
  const cv::Scalar colorBlobNear(0, 255, 0);       // green: near/relevant blob
  const cv::Scalar colorBlobFar(128, 128, 128);    // gray: far/ignored blob
  const cv::Scalar colorText(255, 255, 255);
  const cv::Scalar colorTextBg(0, 0, 0);

  // --- strip scan segments ---
  for (const ScanStrip & s : strips) {
    int y = s.y;
    cv::Scalar color = s.is_branch ? colorLineBranch : colorLineNormal;
    cv::line(out, cv::Point(0, y), cv::Point(w, y), cv::Scalar(60, 60, 60), 1, cv::LINE_AA);
    for (const StripSegment & seg : s.segments) {
      cv::line(out, cv::Point(seg.x_start, y), cv::Point(seg.x_end, y), color, 3);
      cv::circle(out, cv::Point(static_cast<int>(seg.x_center), y), 3, color, -1);
    }
  }

  // --- fitted steering line ---
  int y0 = 0;
  int y1 = h - 1;
  int x0 = static_cast<int>(m * y0 + b);
  int x1 = static_cast<int>(m * y1 + b);
  cv::line(out, cv::Point(x0, y0), cv::Point(x1, y1), colorSteer, 2, cv::LINE_AA);
  for (const cv::Point2d & p : fitPoints) {
    cv::circle(out, cv::Point(static_cast<int>(p.x), static_cast<int>(p.y)), 4, colorSteer, 1);
  }

  // --- junction box + center + forward vector ---
  if (junction.has_value()) {
    double pad = 15;
    cv::rectangle(
      out,
      cv::Point(
        static_cast<int>(junction->box.x - pad), static_cast<int>(junction->box.y - pad)),
      cv::Point(
        static_cast<int>(junction->box.x + junction->box.width + pad),
        static_cast<int>(junction->box.y + junction->box.height + pad)),
      colorJunction, 2);

    cv::Point jCenter(
      static_cast<int>(junction->center.x), static_cast<int>(junction->center.y));
    cv::circle(out, jCenter, 5, colorJunction, -1);

    cv::Point tip(
      static_cast<int>(junction->center.x + forwardVec.x * 40),
      static_cast<int>(junction->center.y + forwardVec.y * 40));
    cv::arrowedLine(out, jCenter, tip, colorJunction, 2, cv::LINE_8, 0, 0.3);
  }

  // --- green blobs ---
  for (const GreenBlobInfo & blob : classified) {
    cv::Scalar color = (blob.region == "near") ? colorBlobNear : colorBlobFar;
    std::vector<std::vector<cv::Point>> contourWrapper = {blob.contour};
    cv::drawContours(out, contourWrapper, -1, color, 2);

    std::string label = (blob.side.empty() ? "?" : blob.side) + "/" + blob.region;
    cv::putText(
      out, label,
      cv::Point(static_cast<int>(blob.center.x) + 8, static_cast<int>(blob.center.y)),
      cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv::LINE_AA);
  }

  // --- text overlay (steering angle, decision) ---
  std::ostringstream angleStream;
  angleStream << std::showpos << std::fixed << std::setprecision(1) <<
    steeringAngleDeg << " deg";

  std::vector<std::string> lines = {
    "steering: " + angleStream.str(),
    std::string("junction: ") + (junction.has_value() ? "YES" : "no"),
    "decision: " + decision,
    "reason: " + decisionReason,
  };

  int textX = 10;
  int textY = 10;
  int lineH = 20;
  int boxW = 0;
  for (const std::string & line : lines) {
    int baseline = 0;
    cv::Size sz = cv::getTextSize(line, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    boxW = std::max(boxW, sz.width);
  }
  boxW += 20;

  cv::rectangle(
    out, cv::Point(textX, textY),
    cv::Point(textX + boxW, textY + lineH * static_cast<int>(lines.size()) + 10),
    colorTextBg, -1);

  for (size_t i = 0; i < lines.size(); i++) {
    int ty = textY + lineH * (static_cast<int>(i) + 1);
    cv::putText(
      out, lines[i], cv::Point(textX + 10, ty), cv::FONT_HERSHEY_SIMPLEX, 0.5,
      colorText, 1, cv::LINE_AA);
  }

  return out;
}

// ============================================================================
// PUBLIC ENTRY POINT for simple navigation
// (mirrors LineFollowerVision.process(), then reduces its high-level
// decision into the scalar steering error the rest of NavigationNode
// expects)
// ============================================================================
double NavigationNode::simpleError(const cv::Mat & frame)
{
  int newWidth = static_cast<int>(frame.cols * 0.8);
  int newHeight = static_cast<int>(frame.rows * 0.6);

  // Center-crop, then resize, exactly as before: this keeps the vision
  // pipeline working on a consistent frame size regardless of the raw
  // camera resolution.
  int x = (frame.cols - newWidth) / 2;
  int y = (frame.rows - newHeight) / 2;
  cv::Rect roi(x, y, newWidth, newHeight);
  cv::Mat croppedImg = frame(roi);

  cv::Mat resized;
  cv::Size dsize(854, 480);
  cv::resize(croppedImg, resized, dsize);

  int h = resized.rows;
  int w = resized.cols;

  // 1. Binarize
  cv::Mat binary = this->visionBinarize(resized);

  // 2. Multi-strip scan -> per-strip segments, doubling as junction detector
  std::vector<ScanStrip> strips = this->visionScanStrips(binary, h, w);
  double baselineWidth = this->visionEstimateLineWidth(strips);
  this->visionFlagBranchStrips(strips, baselineWidth);

  // 2b. Junction detection (byproduct of the strip scan)
  std::optional<JunctionInfo> junction = this->visionDetectJunction(strips, binary);

  // 3. Steering line fit
  double m = 0.0;
  double b = w / 2.0;
  std::vector<cv::Point2d> fitPoints;
  this->visionFitSteeringLine(strips, junction.has_value(), w, m, b, fitPoints);

  double steeringAngleDeg = this->visionSteeringAngle(m);

  cv::Point2d forwardVec, rightVec;
  this->visionHeadingVectors(m, forwardVec, rightVec);

  // 4. Green marker detection + classification relative to the junction
  std::vector<GreenBlobInfo> greenBlobs = this->visionDetectGreenBlobs(resized);
  this->visionClassifyBlobs(greenBlobs, junction, forwardVec, rightVec, baselineWidth);

  // 5. Decision fusion
  std::string decision, decisionReason;
  this->visionDecide(junction, greenBlobs, decision, decisionReason);

  cv::Mat processed = this->visionAnnotate(
    resized, strips, baselineWidth, junction, m, b, fitPoints, steeringAngleDeg,
    greenBlobs, decision, decisionReason, forwardVec);
  this->writer.write(processed);

  // --- Reduce the module's high-level decision into a scalar error ---
  //
  // The python module hands back STRAIGHT / TURN_LEFT / TURN_RIGHT /
  // TURN_180 plus a continuous steering angle; simpleNavigation() only
  // wants one steering error to publish, so:
  //   - TURN_180 (green markers on both sides of the incoming arm) is
  //     handled exactly like the old "double green" behaviour: kick off
  //     a rotate-in-place maneuver and let the state machine take over.
  //   - otherwise, the error follows the fitted line's steering angle,
  //     nudged by how far off-center the line sits at the bottom row
  //     (mirrors line_offset_px), and gets a hard bias toward whichever
  //     side a single green marker calls for at a junction.
  // The bias magnitude and offset weighting are new tuning knobs (they
  // don't exist in the python module, which only reports a decision) and
  // will likely need retuning against the real 0.01 gain in publishError.
  if (decision == "TURN_180") {
    RCLCPP_INFO(this->get_logger(), "Starting double green (180)");
    this->sendMovementGoal(0, 100, 3.3);
    this->state = GREEN_ROTATE;
    RCLCPP_INFO(this->get_logger(), "Sent double green start message");
    return 0.0;
  }

  double lineOffsetPx = this->visionLineOffset(m, b, h, w);
  double error = steeringAngleDeg + lineOffsetPx * 0.05;

  const double turnBiasDeg = 45.0;
  if (decision == "TURN_LEFT") {
    error -= turnBiasDeg;
  } else if (decision == "TURN_RIGHT") {
    error += turnBiasDeg;
  }

  return error;
}

void NavigationNode::simpleNavigation(cv::Mat & frame)
{
  double error = this->simpleError(frame);

  this->publishError(0.01 * error);
}

void NavigationNode::advancedNavigation(cv::Mat & frame)
{
  cv::Mat processed = this->processImage(frame); \
  this->skeletonizedImage = processed;
  this->extractNodes();
  this->extractEdges();
  this->removeShortEdges(this->graph.edges);
  this->removeUnconnectedNodes();
  this->updateGraph();
  this->findNextTarget(this->currentTarget, &this->currentEdge);
}

cv::Mat NavigationNode::applyThreshold(cv::Mat & image, uint32_t threshSize, uint32_t kernelSize)
{
  cv::Mat gray, binary;
  cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  GaussianBlur(gray, gray, cv::Size(kernelSize, kernelSize), 0);

  // cv::adaptiveThreshold(gray, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
  //                       cv::THRESH_BINARY_INV, threshSize, 4);

  cv::threshold(gray, binary, 120, 255, cv::THRESH_BINARY_INV);
  // binary = this->applySmoothVariableThreshold(gray);

  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));

  cv::Mat opened_image;
  cv::morphologyEx(binary, opened_image, cv::MORPH_OPEN, kernel);

  cv::Mat closed_image;
  cv::morphologyEx(opened_image, closed_image, cv::MORPH_CLOSE, kernel);

  return closed_image;
}

cv::Mat NavigationNode::processImage(cv::Mat & image)
{
  cv::Mat resized;
  cv::Size dsize(200, 100);
  cv::resize(image, resized, dsize);

  cv::Mat binary = this->applyThreshold(resized, 55, 5);

  cv::Mat skeleton;
  cv::ximgproc::thinning(binary, skeleton,
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

cv::Point NavigationNode::localToGlobalFrame(cv::Point point)
{
  cv::Point relative = point - this->frameCentre;
  float rotatedX = relative.x * std::cos(this->angle) - relative.y * std::sin(this->angle);
  float rotatedY = relative.x * std::sin(this->angle) + relative.y * std::cos(this->angle);

  cv::Point newCenter = cv::Point2f(this->x, this->y) / this->pixelSize;
  return newCenter + cv::Point(rotatedX, rotatedY);
}

cv::Point NavigationNode::cvtPoint(
  cv::Mat & src, cv::Mat & dst,
  cv::Point point)
{
  double sx = static_cast<double>(dst.cols) / static_cast<double>(src.cols);
  double sy = static_cast<double>(dst.rows) / static_cast<double>(src.rows);

  return cv::Point(cv::saturate_cast<int>(point.x * sx),
                   cv::saturate_cast<int>(point.y * sy));
}

Node * Graph::nodeFromID(int id)
{
  for (Node & node : this->nodes) {
    if (node.id == id) {
      return &node;
    }
  }

  return nullptr;
}

TrackedNode * TrackedGraph::nodeFromID(int id)
{
  for (TrackedNode & node : this->nodes) {
    if (node.id == id) {
      return &node;
    }
  }

  return nullptr;
}

void NavigationNode::extractNodes()
{
  cv::Mat image = this->skeletonizedImage;
  std::vector<cv::Point> whitePixels;
  std::vector<Node> foundNodes;

  cv::findNonZero(image, whitePixels);

  for (const auto & point : whitePixels) {
    std::vector<cv::Point> surroundingPoints =
      this->getSurroundingPoints(point, 3);

    if (surroundingPoints.size() == 3) {
      continue;
    }

    Node node;
    node.pos = point;
    node.id = this->graph.nextID++;

    if (surroundingPoints.size() > 3) {
      node.is_endpoint = false;
    } else {
      node.is_endpoint = true;
    }

    if (node.pos.x <= 1 || node.pos.y <= 1 || node.pos.x >= image.cols - 2 ||
      node.pos.y >= image.rows - 2)
    {
      node.screen_edge = true;
    } else {
      node.screen_edge = false;
    }

    foundNodes.push_back(node);
  }

  this->graph.nodes = foundNodes;
}

std::vector<cv::Point> NavigationNode::getSurroundingPoints(
  cv::Point centre,
  int radius)
{
  cv::Mat image = this->skeletonizedImage;
  cv::Rect roi(centre.x - 1, centre.y - 1, radius, radius);
  std::vector<cv::Point> surroundingPoints;

  if (centre.x <= 0 || centre.y <= 0 || centre.x >= image.cols - 1 ||
    centre.y >= image.rows - 1)
  {
    return surroundingPoints;
  }

  cv::Mat cropped;
  cropped = image(roi).clone();

  cv::findNonZero(cropped, surroundingPoints);

  for (auto & point : surroundingPoints) {
    point += centre + cv::Point(-1, -1);
  }

  return surroundingPoints;
}

void NavigationNode::extractEdges()
{
  if (this->graph.nodes.size() == 0) {
    return;
  }

  std::vector<Edge> edges;

  for (const auto & node : this->graph.nodes) {
    // unoptimised—Should check if node path exists on edge before tracing
    std::vector<Edge> connectedEdges = this->traceConnectedEdges(node);

    for (const auto & edge : connectedEdges) {
      bool exists = false;

      for (const auto & existingEdge : edges) {
        if (edge == existingEdge) {
          exists = true;
          break;
        }
      }

      if (!exists) {
        edges.push_back(edge);
      }
    }
  }

  this->graph.edges = edges;
}

std::vector<Edge *> Graph::getConnectedEdges(int nodeID)
{
  std::vector<Edge *> result;

  for (Edge & edge : this->edges) {
    if (edge.src == nodeID || edge.dst == nodeID) {
      result.push_back(&edge);
    }
  }

  return result;
}

std::vector<TrackedEdge *> TrackedGraph::getConnectedEdges(int nodeID)
{
  std::vector<TrackedEdge *> result;

  for (TrackedEdge & edge : this->edges) {
    if (edge.src == nodeID || edge.dst == nodeID) {
      result.push_back(&edge);
    }
  }

  return result;
}

void NavigationNode::removeShortEdges(std::vector<Edge> & edges)
{
  for (uint32_t i = 0; i < edges.size(); i++) {
    // If the edge is long enough, do nothing.
    if (edges[i].path.size() >= this->minEdgeSize) {
      continue;
    }

    Node *src = this->graph.nodeFromID(edges[i].src);
    Node *dst = this->graph.nodeFromID(edges[i].dst);

    if (!src || !dst) {
      RCLCPP_ERROR(this->get_logger(), "Node does not exist!");
      continue;
    }

    // If either of the ends of an edge are endpoints, delete it.
    if (src->is_endpoint || dst->is_endpoint) {
      edges.erase(edges.begin() + i);
      i--;
      continue;
    }

    // Merge close intersections
    for (Edge *connectedEdge : this->graph.getConnectedEdges(edges[i].src)) {
      if (!connectedEdge) {
        RCLCPP_ERROR(this->get_logger(), "Edge does not exist!");
        continue;
      }

      *connectedEdge = this->mergeEdges(*connectedEdge, edges[i]);
    }

    edges.erase(edges.begin() + i);
    i--;
  }
}

Edge NavigationNode::mergeEdges(Edge edge1, Edge edge2)
{
  if (edge1.dst == edge2.src) {
    edge1.path.insert(edge1.path.end(), edge2.path.begin() + 1,
                      edge2.path.end());
    edge1.dst = edge2.dst;
  } else if (edge1.src == edge2.dst) {
    edge1.path.insert(edge1.path.begin(), edge2.path.begin() + 1,
                      edge2.path.end());
    edge1.src = edge2.src;
  } else if (edge1.dst == edge2.dst) {
    std::reverse(edge2.path.begin(), edge2.path.end());

    edge1.path.insert(edge1.path.end(), edge2.path.begin() + 1,
                      edge2.path.end());
    edge1.dst = edge2.src;
  } else if (edge1.src == edge2.src) {
    std::reverse(edge2.path.begin(), edge2.path.end());
    edge1.path.insert(edge1.path.begin(), edge2.path.begin() + 1,
                      edge2.path.end());
    edge1.src = edge2.dst;
  }

  edge1.length = edge1.path.size();
  return edge1;
}

void NavigationNode::removeUnconnectedNodes()
{
  for (uint32_t i = 0; i < this->graph.nodes.size(); i++) {
    Node node = this->graph.nodes[i];

    bool connected = false;
    for (uint32_t j = 0; j < this->graph.edges.size() && !connected; j++) {
      Edge edge = this->graph.edges[j];

      if (edge.src == node.id || edge.dst == node.id) {
        connected = true;
      }
    }

    if (!connected) {
      this->graph.nodes.erase(this->graph.nodes.begin() + i);
      i--;
    }
  }
}

std::vector<Edge> NavigationNode::traceConnectedEdges(Node node)
{
  std::vector<Edge> connectedEdges;
  std::vector<cv::Point> surroundingPoints =
    this->getSurroundingPoints(node.pos, 3);

  for (const auto & point : surroundingPoints) {
    if (point == node.pos) {
      continue;
    }

    Edge edge;
    edge.src = node.id;

    edge.path.push_back(node.pos);
    edge.path.push_back(point);

    edge.dst = this->followToNode(edge.path).id;
    edge.length = edge.path.size();

    connectedEdges.push_back(edge);
  }

  return connectedEdges;
}

double NavigationNode::calculateAngle(cv::Point point1, cv::Point point2)
{
  int rise = point2.y - point1.y;
  int run = point2.x - point1.x;

  double angle = std::atan2(rise, run);
  return angle;
}

double NavigationNode::calculateDist(cv::Point point1, cv::Point point2)
{
  return std::sqrt(std::pow(point1.x - point2.x, 2) +
                   std::pow(point1.y - point2.y, 2));
}

Node NavigationNode::followToNode(std::vector<cv::Point> & path)
{
  cv::Point current = path[path.size() - 1];
  cv::Point previous;

  if (path.size() > 1) {
    previous = path[path.size() - 2];
  }

  auto it =
    std::find_if(this->graph.nodes.begin(), this->graph.nodes.end(),
      [current](const Node & node) {return node.pos == current;});

  if (it != this->graph.nodes.end()) {
    return *it;
  }

  std::vector<cv::Point> surroundingPoints =
    this->getSurroundingPoints(current, 3);

  auto it1 =
    std::find(surroundingPoints.begin(), surroundingPoints.end(), current);

  if (it1 != surroundingPoints.end()) {
    surroundingPoints.erase(it1);
  }

  auto it2 =
    std::find(surroundingPoints.begin(), surroundingPoints.end(), previous);

  if (it2 != surroundingPoints.end()) {
    surroundingPoints.erase(it2);
  }

  if (surroundingPoints.size() != 1) {
    throw std::runtime_error("Line does not end in node!");
  }

  path.push_back(surroundingPoints[0]);

  return this->followToNode(path);
}

void NavigationNode::findNextNode(std::vector<Node> & path)
{
  Node current = path[path.size() - 1];
  Node previous = path[path.size() - 2];

  std::vector<Edge *> connected = this->graph.getConnectedEdges(current.id);
  std::vector<int> connectedNodes;

  for (const Edge *edge : connected) {
    if (edge->dst == current.id) {
      connectedNodes.push_back(edge->src);
    } else {
      connectedNodes.push_back(edge->dst);
    }
  }

  if (connected.size() == 0 || path.size() > this->pathLimit) {
    return;
  }

  std::vector<double> connectedDirs =
    this->getEdgeDirections(current, connected);

  double previousAngle = 0;

  for (uint32_t i = 0; i < connected.size(); i++) {
    if (connected[i]->src == previous.id || connected[i]->dst == previous.id) {
      previousAngle = connectedDirs[i];
    }
  }

  double targetAngle = fmod(previousAngle + M_PI, 2 * M_PI);
  double closestAngle = connectedDirs[0];
  int closestNode = connectedNodes[0];

  for (uint32_t i = 0; i < connected.size(); i++) {
    double angle = connectedDirs[i];
    if (abs(angle - targetAngle) < abs(closestAngle - targetAngle)) {
      closestAngle = angle;
      closestNode = connectedNodes[i];
    }
  }

  Node next = *this->graph.nodeFromID(closestNode);
  path.push_back(next);
  this->findNextNode(path);
}

void NavigationNode::updateGraph()
{
  for (TrackedNode & node : this->trackedGraph.nodes) {
    cv::Mat prediction = node.kf.predict();

    // int x = prediction.at<float>(0);
    // int y = prediction.at<float>(1);

    // node.pos = cv::Point(x, y);
  }

  // If there are no nodes, add all nodes currently observec
  if (this->trackedGraph.nodes.size() == 0) {
    graph.nextID = 0;

    std::vector<int> newIDs;

    for (const Node & node : this->graph.nodes) {
      TrackedNode newNode(this->localToGlobalFrame(node.pos));

      newNode.id = graph.nextID++;
      newNode.is_endpoint = node.is_endpoint;
      newNode.screen_edge = node.screen_edge;

      this->trackedGraph.nodes.push_back(newNode);
      newIDs.push_back(newNode.id);
    }

    for (uint32_t i = 0; i < this->graph.edges.size(); i++) {
      Edge edge = this->graph.edges[i];
      TrackedEdge trackedEdge;
      this->edgeToTracked(edge, trackedEdge);

      // TODO FIX
      int src = edge.src;
      int dst = edge.dst;

      auto src_it =
        std::find_if(this->graph.nodes.begin(), this->graph.nodes.end(),
          [&src](const Node & node) {return node.id == src;});

      auto dst_it =
        std::find_if(this->graph.nodes.begin(), this->graph.nodes.end(),
          [&dst](const Node & node) {return node.id == dst;});

      if (src_it != this->graph.nodes.end() &&
        dst_it != this->graph.nodes.end())
      {
        continue;
      }

      int srcIndex = newIDs[std::distance(this->graph.nodes.begin(), src_it)];
      int dstIndex = newIDs[std::distance(this->graph.nodes.begin(), dst_it)];

      trackedEdge.src = newIDs[srcIndex];
      trackedEdge.dst = newIDs[dstIndex];

      this->trackedGraph.edges.push_back(trackedEdge);
    }

    return;
  }

  // Match observed nodes to tracked nodes
  std::vector<std::vector<double>> costMatrix =
    this->trackedGraph.getCostMatrix(this->graph);
  std::vector<int> assignment;

  HungarianAlgorithm().Solve(costMatrix, assignment);
  std::vector<bool> matched(this->graph.nodes.size(), false);
  std::vector<int> newIDs(this->graph.nodes.size(), 0);

  // Update matched nodes
  for (uint32_t i = 0; i < this->trackedGraph.nodes.size(); i++) {
    int assigned = assignment[i];

    if (assigned > 0 && static_cast<uint32_t>(assigned) < matched.size() &&
      costMatrix[i][assigned] <= this->gatingThreshold)
    {
      cv::Mat measurement =
        (cv::Mat_<float>(2, 1) << this->graph.nodes[assigned].pos.x,
        this->graph.nodes[assigned].pos.y);

      // this->trackedGraph.nodes[i].kf.correct(measurement);
      this->trackedGraph.nodes[i].missedFrames = 0;
      this->trackedGraph.nodes[i].age++;
      this->trackedGraph.nodes[i].pos = this->graph.nodes[assigned].pos;
      this->trackedGraph.nodes[i].is_endpoint =
        this->graph.nodes[assigned].is_endpoint;
      this->trackedGraph.nodes[i].screen_edge =
        this->graph.nodes[assigned].screen_edge;
      matched[assigned] = true;
      newIDs[assigned] = this->trackedGraph.nodes[i].id;
    } else {
      this->trackedGraph.nodes[i].missedFrames++;
    }
  }

  // Add nodes that weren't matched
  for (uint32_t i = 0; i < matched.size(); i++) {
    if (matched[i]) {
      continue;
    }

    Node detectedNode = this->graph.nodes[assignment[i]];

    TrackedNode newNode(detectedNode.pos);
    newNode.id = this->trackedGraph.nextID++;
    newNode.screen_edge = detectedNode.screen_edge;
    newNode.is_endpoint = detectedNode.is_endpoint;

    this->trackedGraph.nodes.push_back(newNode);
    newIDs[i] = newNode.id;
  }

  // Remove nodes that haven't been seen in 5 frames
  this->trackedGraph.nodes.erase(
      std::remove_if(
          this->trackedGraph.nodes.begin(), this->trackedGraph.nodes.end(),
      [](const TrackedNode & node) {return node.missedFrames > 5;}),
      this->trackedGraph.nodes.end());

  this->trackedGraph.edges.clear();

  // Add edges to tracked graph
  for (uint32_t i = 0; i < this->graph.nodes.size(); i++) {
    Node & node = this->graph.nodes[i];
    for (const Edge & edge : this->graph.edges) {
      if (edge.src != node.id && edge.dst != node.id) {
        continue;
      }

      int connectedID = edge.src == node.id ? edge.dst : edge.src;

      auto connectedIt =
        std::find_if(this->graph.nodes.begin(), this->graph.nodes.end(),
          [&connectedID](const Node & connected) {
            return connected.id == connectedID;
                       });

      if (connectedIt == this->graph.nodes.end()) {
        RCLCPP_ERROR(this->get_logger(),
          "Couldn't find the other node??? (This should never happen)");
        return;
      }

      int connectedIndex =
        std::distance(this->graph.nodes.begin(), connectedIt);

      int trackedSrcIndex = edge.src == node.id ? i : connectedIndex;
      int trackedDstIndex = edge.dst == node.id ? i : connectedIndex;

      int trackedSrc = newIDs[trackedSrcIndex];
      int trackedDst = newIDs[trackedDstIndex];

      TrackedEdge tracked;
      this->edgeToTracked(edge, tracked);

      tracked.src = trackedSrc;
      tracked.dst = trackedDst;

      bool exists = false;
      for (const TrackedEdge & existingTracked : this->trackedGraph.edges) {
        if ((tracked.src == existingTracked.src &&
          tracked.dst == existingTracked.dst) ||
          (tracked.src == existingTracked.dst &&
          tracked.dst == existingTracked.src))
        {
          exists = true;
          break;
        }
      }

      if (exists) {
        continue;
      }

      this->trackedGraph.edges.push_back(tracked);
    }
  }

  // Remove unconnected edges
  for (uint32_t i = 0; i < this->trackedGraph.edges.size(); i++) {
    const TrackedEdge & edge = this->trackedGraph.edges[i];

    int src = edge.src;
    int dst = edge.dst;

    auto src_it = std::find_if(
        this->trackedGraph.nodes.begin(), this->trackedGraph.nodes.end(),
      [&src](const TrackedNode & node) {return node.id == src;});

    auto dst_it = std::find_if(
        this->trackedGraph.nodes.begin(), this->trackedGraph.nodes.end(),
      [&dst](const TrackedNode & node) {return node.id == dst;});

    if (src_it != this->trackedGraph.nodes.end() &&
      dst_it != this->trackedGraph.nodes.end())
    {
      continue;
    }

    this->trackedGraph.edges.erase(this->trackedGraph.edges.begin() + i);
    i--;
  }

}

std::vector<double>
NavigationNode::getEdgeDirections(Node origin, std::vector<Edge *> edges)
{
  std::vector<double> results;

  for (const Edge *edge : edges) {
    cv::Point p;

    if (edge->src == origin.id) {
      p = edge->path[this->minEdgeSize - 1];
    } else {
      p = edge->path[edge->path.size() - this->minEdgeSize];
    }

    double dy = p.y - origin.pos.y;
    double dx = p.x - origin.pos.x;

    double angle = std::atan2(dy, dx);

    results.push_back(angle);
  }

  return results;
}

void NavigationNode::edgeToTracked(const Edge & edge, TrackedEdge & tracked)
{
  tracked.length = edge.length;
  tracked.age = 0;

  tracked.angleFromSrc = this->calculateAngle(
      this->graph.nodeFromID(edge.src)->pos, edge.path[this->minEdgeSize - 1]) + this->angle;
  tracked.angleFromDst =
    this->calculateAngle(this->graph.nodeFromID(edge.dst)->pos,
                           edge.path[edge.path.size() - this->minEdgeSize]) + this->angle;
  tracked.path = edge.path;
}

void NavigationNode::findStartingEdge(
  int & trackingID,
  TrackedEdge **currentEdge)
{
  trackingID = -1;

  if (this->trackedGraph.edges.size() == 0) {
    return;
  }

  int largest = 0;
  for (const TrackedEdge & edge : this->trackedGraph.edges) {
    if (edge.length > largest) {
      **currentEdge = edge;
      largest = edge.length;
    }
  }


  TrackedNode *src = this->trackedGraph.nodeFromID((*currentEdge)->src);
  TrackedNode *dst = this->trackedGraph.nodeFromID((*currentEdge)->dst);

  trackingID =
    src->pos.y > dst->pos.y ? (*currentEdge)->dst : (*currentEdge)->src;
}

double NavigationNode::wrapAngle(double angle)
{
  constexpr double two_pi = 2.0 * std::numbers::pi;
  double wrapped = std::remainder(angle, two_pi);

  if (wrapped == -std::numbers::pi) {
    wrapped = std::numbers::pi;
  }

  return wrapped;
}

double NavigationNode::addAngles(double angle1, double angle2)
{
  double sum = angle1 + angle2;
  double wrapped = this->wrapAngle(sum);
  return wrapped;
}

void NavigationNode::findNextTarget(
  int & trackingID,
  TrackedEdge **currentEdge)
{
  if (this->searchLineBreak) {
    if (this->trackedGraph.nodes.size() == 0) {
      return;
    }

    std::vector<TrackedNode> nodesInRange;

    for (TrackedNode & node : this->trackedGraph.nodes) {
      double dist = this->searchDistance(node.pos);
      if (dist > this->searchMinDist) {
        continue;
      }

      nodesInRange.push_back(node);
    }

    if (nodesInRange.size() == 0) {
      return;
    }

    TrackedNode closestNode = nodesInRange[0];
    double closestDist =
      this->calculateDist(this->searchLastPoint, closestNode.pos);

    for (TrackedNode & node : nodesInRange) {
      double dist = this->calculateDist(this->searchLastPoint, node.pos);

      if (dist > closestDist) {
        continue;
      }

      closestDist = dist;
      closestNode = node;
    }

    std::vector<TrackedEdge *> surroundingEdges;

    for (TrackedEdge & edge : this->trackedGraph.edges) {
      if (edge.src == closestNode.id || edge.dst == closestNode.id) {
        surroundingEdges.push_back(&edge);
      }
    }

    TrackedEdge *newEdge = this->closestToAngle(
        closestNode.id, surroundingEdges, this->searchDirection);
    int newTarget =
      closestNode.id == newEdge->src ? newEdge->dst : newEdge->src;

    this->currentTarget = newTarget;
    this->currentEdge = newEdge;
    this->searchLineBreak = false;
  }

  if (!currentEdge || trackingID < 0) {
    *currentEdge = new TrackedEdge;
    this->findStartingEdge(trackingID, currentEdge);
    return;
  }

  TrackedNode *currentNodePointer = this->trackedGraph.nodeFromID(trackingID);

  if (!currentNodePointer) {
    this->currentTarget = -1;
    this->findStartingEdge(trackingID, currentEdge);
    return;
  }

  TrackedNode currentNode = *currentNodePointer;

  if (currentNode.screen_edge) {
    return;
  } else if (currentNode.is_endpoint) {
    this->searchLineBreak = true;
    this->searchLastNode = currentNode.id;
    this->searchLastPoint = currentNode.pos;
    double currentDir = trackingID == (*currentEdge)->src ?
      (*currentEdge)->angleFromSrc :
      (*currentEdge)->angleFromDst;
    this->searchDirection = this->addAngles(currentDir, std::numbers::pi);


    return;
  }

  std::vector<TrackedEdge *> surroundingEdges;

  for (TrackedEdge & edge : this->trackedGraph.edges) {
    if (edge.src == trackingID || edge.dst == trackingID) {
      surroundingEdges.push_back(&edge);
    }
  }


  if (surroundingEdges.size() == 0) {
    RCLCPP_ERROR(this->get_logger(), "No surrounding edges to node (This should never happen)");
    return;
  }

  double currentAngle = trackingID == (*currentEdge)->src ?
    (*currentEdge)->angleFromSrc :
    (*currentEdge)->angleFromDst;

  if (surroundingEdges.size() < 3) {

    double targetAngle = this->addAngles(currentAngle, std::numbers::pi);

    TrackedEdge *closestEdge =
      this->closestToAngle(trackingID, surroundingEdges, targetAngle);

    *this->currentEdge = *closestEdge;
    this->currentTarget =
      trackingID == closestEdge->src ? closestEdge->dst : closestEdge->src;

    return;
  }

  std::vector<double> surroundingGreen;
  double minGreenDist = 40;

  for (cv::Point greenPos : this->green) {
    double dist = this->calculateDist(greenPos, currentNode.pos);
    if (dist > minGreenDist) {
      continue;
    }

    double angle = this->calculateAngle(currentNode.pos, greenPos);
    surroundingGreen.push_back(angle);
  }

  double targetLeft = this->addAngles(currentAngle, -std::numbers::pi / 2);
  double targetRight = this->addAngles(currentAngle, std::numbers::pi / 2);
  double targetStraight = this->addAngles(currentAngle, std::numbers::pi);

  TrackedEdge *leftEdge =
    this->closestToAngle(trackingID, surroundingEdges, targetLeft);
  TrackedEdge *rightEdge =
    this->closestToAngle(trackingID, surroundingEdges, targetRight);
  TrackedEdge *straightEdge =
    this->closestToAngle(trackingID, surroundingEdges, targetStraight);

  bool greenLeft = false;
  bool greenRight = false;

  for (double & greenAngle : surroundingGreen) {
    double diff = this->addAngles(currentAngle, -greenAngle);

    if (diff < 0 && diff > -std::numbers::pi / 2) {
      greenLeft = true;
    }
    if (diff > 0 && diff < std::numbers::pi / 2) {
      greenRight = true;
    }
  }

  if (greenRight && greenLeft) {
    trackingID = trackingID == (*currentEdge)->src ? (*currentEdge)->dst :
      (*currentEdge)->src;
    return;
  } else if (greenRight) {
    **currentEdge = *rightEdge;
  } else if (greenLeft) {
    **currentEdge = *leftEdge;
  } else {
    **currentEdge = *straightEdge;
  }

  trackingID = trackingID == (*currentEdge)->src ? (*currentEdge)->dst :
    (*currentEdge)->src;
}

double NavigationNode::searchDistance(cv::Point point)
{
  double sinTheta = std::sin(this->searchDirection);
  double cosTheta = std::cos(this->searchDirection);

  // Vector from startPoint to targetPoint
  double dx = point.x - this->searchLastPoint.x;
  double dy = point.y - this->searchLastPoint.y;

  // Project the target point onto the line's direction vector (Dot Product)
  double projection = dx * cosTheta + dy * sinTheta;

  if (projection < 0.0) {
    // The point is "behind" the starting point.
    // Return the straight-line Euclidean distance to the startPoint.
    return 50 * std::sqrt(dx * dx + dy * dy);
  }

  return std::abs(dx * sinTheta - dy * cosTheta);
}

TrackedEdge *
NavigationNode::closestToAngle(
  int currentNode,
  std::vector<TrackedEdge *> currentEdges,
  double targetAngle)
{
  TrackedEdge *closestEdge = currentEdges[0];
  double closestAngle = currentNode == currentEdge->src ?
    closestEdge->angleFromSrc :
    closestEdge->angleFromDst;

  for (TrackedEdge *edge : currentEdges) {
    double angle =
      currentNode == edge->src ? edge->angleFromSrc : edge->angleFromDst;

    double closestDiff = std::abs(this->addAngles(targetAngle, -closestAngle));
    double diff = std::abs(this->addAngles(targetAngle, -angle));

    if (diff < closestDiff) {
      closestAngle = angle;
      closestEdge = edge;
    }
  }

  return closestEdge;
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<NavigationNode>(rclcpp::NodeOptions());
  rclcpp::spin(node->get_node_base_interface());
  rclcpp::shutdown();
  return 0;
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(NavigationNode);

