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

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/ximgproc.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include "nav_msgs/msg/odometry.hpp"
#include <lifecycle_msgs/msg/state.hpp>
#include <std_msgs/msg/float64.hpp>

#include <Hungarian.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <numbers>
#include <unordered_map>

using std::placeholders::_1;
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
    {"simple", NavigationType::SIMPLE}, {"advanced", NavigationType::ADVANCED}};

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

  this->writer = cv::VideoWriter("/videos/output.mp4", fourcc, fps, frameSize);

  if (!this->writer.isOpened()) {
    RCLCPP_ERROR(this->get_logger(), "CRITICAL ERROR: VideoWriter failed to initiate!");
    RCLCPP_ERROR(
      this->get_logger(), "Check 1: Does OpenCV have FFMPEG? Build Info: %s",
      cv::getBuildInformation().c_str());
  }
}

CallbackReturn NavigationNode::on_configure(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(this->get_logger(), "Configuring...");
  auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();

  this->errorPub = this->create_publisher<std_msgs::msg::Float64>("line_error", 10);
  this->lineCompletePub = this->create_publisher<std_msgs::msg::Bool>("rescue_active", 10);
  this->imageSub = this->create_subscription<sensor_msgs::msg::Image>(
    "/down_camera/camera_node/image_raw", qos, std::bind(&NavigationNode::imageCallback, this, _1));
  this->odomSub = this->create_subscription<nav_msgs::msg::Odometry>(
    "/odom", 10, std::bind(&NavigationNode::odomCallback, this, _1));

  timer_ = this->create_wall_timer(100ms, std::bind(&NavigationNode::timerCallback, this));
  timer_->cancel();

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
  kf.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, dt, 0, 0, 1, 0, dt, 0, 0, 1, 0, 0, 0, 0, 1);

  kf.measurementMatrix = (cv::Mat_<float>(2, 4) << 1, 0, 0, 0, 0, 1, 0, 0);

  cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-4));

  cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-2));

  cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1));

  kf.statePost.at<float>(1) = pos.x;  // Initial X
  kf.statePost.at<float>(1) = pos.y;  // Initial Y
  kf.statePost.at<float>(2) = 0.0f;   // Initial velocity X
  kf.statePost.at<float>(3) = 0.0f;   // Initial velocity Y

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
  sendGoalOptions.result_callback = std::bind(&NavigationNode::goalResultCallback, this, _1);

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
      this->sendMovementGoal(100, 0, 5);
      this->state = GREEN_MOVE_FORWARD;
      break;
    case GREEN_MOVE_FORWARD:
      this->sendMovementGoal(100, 0, 5);
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
  int size = this->nodes.size() > graph.nodes.size() ? this->nodes.size() : graph.nodes.size();

  std::vector<std::vector<double>> costs(size, std::vector(size, 0.0));

  for (uint32_t i = 0; i < this->nodes.size(); i++) {
    for (uint32_t j = 0; j < graph.nodes.size(); j++) {
      Node & newNode = graph.nodes[j];
      Node & trackedNode = this->nodes[i];

      double distance = cv::norm(trackedNode.pos - newNode.pos);
      int connectedEdgeDiff =
        graph.getConnectedEdges(newNode.id).size() - this->getConnectedEdges(trackedNode.id).size();

      double penalty =
        trackedNode.screen_edge ? 0 : this->edgePenalty * std::abs(connectedEdgeDiff);

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
    case FOLLOWING:
      switch (this->navigationType) {
        case NavigationType::SIMPLE:
          this->simpleNavigation(this->currentFrame);
          break;
        case NavigationType::ADVANCED:
          this->advancedNavigation(this->currentFrame);
          break;
      }

      break;
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
  msg.data = error * 500;
  this->errorPub->publish(msg);
}

// ---------------------------------------------------------------------------
// Simple line follower.
//
// Instead of thresholding + taking the centroid of the largest blob (which
// gets dragged around by whatever else is in frame, and has no real notion
// of "the line ahead" vs "the line behind"), this:
//
//   1. Thins the binary mask down to a 1px skeleton of the line.
//   2. Starts from the skeleton pixel closest to the robot and walks
//      forward along the skeleton, one connected pixel at a time -- this is
//      the actual path-finding step. The walk stops either because it ran
//      out of lookahead, hit a dead end, or hit a junction (a skeleton
//      pixel with more than one place to go next -- i.e. an intersection).
//   3. Tracks the walk's end position and heading direction from frame to
//      frame, so "which way is forward" stays consistent even when the
//      line is only visible at an odd angle (e.g. approaching a junction
//      off-axis), rather than being recomputed from scratch every frame.
//   4. If a junction was found, hands off to evaluateIntersectionGreen()
//      to look at the green markers relative to that junction and decide
//      whether to go straight, turn, or reverse.
// ---------------------------------------------------------------------------
double NavigationNode::simpleError(const cv::Mat & frame)
{
  int newWidth = static_cast<int>(frame.cols * 0.8);
  int newHeight = static_cast<int>(frame.rows * 0.6);

  int cropX = (frame.cols - newWidth) / 2;
  int cropY = (frame.rows - newHeight) / 2;

  cv::Rect roi(cropX, cropY, newWidth, newHeight);
  cv::Mat croppedImg = frame(roi);

  cv::Mat resized;
  cv::Size dsize(854, 480);
  cv::resize(croppedImg, resized, dsize);

  cv::Mat thresh = this->applyThreshold(resized, 255, 35);

  // --- Skeletonize down to a 1px-wide line for path-finding ---
  cv::Mat skeleton;
  cv::ximgproc::thinning(thresh, skeleton, cv::ximgproc::THINNING_GUOHALL);

  // Blank the very edge so that a skeleton pixel touching the border reads
  // as a genuine "leaves the frame" endpoint (one fewer neighbour), the
  // same trick used for the advanced-mode graph skeleton.
  int skelRows = skeleton.rows;
  int skelCols = skeleton.cols;
  skeleton(cv::Rect(0, 0, skelCols, 1)).setTo(0);
  skeleton(cv::Rect(0, skelRows - 1, skelCols, 1)).setTo(0);
  skeleton(cv::Rect(0, 1, 1, skelRows - 2)).setTo(0);
  skeleton(cv::Rect(skelCols - 1, 1, 1, skelRows - 2)).setTo(0);

  // getSurroundingPoints() (shared with advanced mode) reads from this
  // member -- simple and advanced never run in the same frame, so reusing
  // it here is safe.
  this->skeletonizedImage = skeleton;

  cv::Mat processed = resized.clone();

  std::vector<cv::Point> skeletonPoints;
  cv::findNonZero(skeleton, skeletonPoints);

  if (skeletonPoints.empty()) {
    this->writer.write(processed);
    this->havePrevForwardEndpoint = false;
    this->havePrevHeading = false;
    return 0.0;
  }

  // Robot's reference point: bottom-centre of the (resized) crop.
  cv::Point stationaryPoint(resized.cols / 2, resized.rows - 1);

  // --- Pick where to start walking from ---
  // Default: the skeleton pixel nearest the robot.
  cv::Point walkStart = skeletonPoints[0];
  double bestDist = this->calculateDist(walkStart, stationaryPoint);

  for (const cv::Point & p : skeletonPoints) {
    double dist = this->calculateDist(p, stationaryPoint);
    if (dist < bestDist) {
      bestDist = dist;
      walkStart = p;
    }
  }

  // If last frame ended its walk somewhere, and that point (or something
  // near it) is still visible, prefer picking up the walk from there --
  // keeps us on the same physical strand of line frame-to-frame instead of
  // occasionally locking onto a noise fragment closer to the robot.
  if (this->havePrevForwardEndpoint) {
    double trackedDist = this->calculateDist(walkStart, this->prevForwardEndpoint);
    for (const cv::Point & p : skeletonPoints) {
      double dist = this->calculateDist(p, this->prevForwardEndpoint);
      if (dist < trackedDist) {
        trackedDist = dist;
        walkStart = p;
      }
    }
  }

  // --- Walk forward along the skeleton until a junction or a dead end ---
  const int lookaheadSteps = 260;

  cv::Point current = walkStart;
  cv::Point previous(-1, -1);  // sentinel: never matches a real pixel

  std::vector<cv::Point> path{current};
  bool hitJunction = false;
  cv::Point junctionCentre = current;

  for (int step = 0; step < lookaheadSteps; step++) {
    std::vector<cv::Point> neighbours = this->getSurroundingPoints(current, 3);

    std::vector<cv::Point> forwardCandidates;
    for (const cv::Point & n : neighbours) {
      if (n == current || n == previous) {
        continue;
      }
      forwardCandidates.push_back(n);
    }

    if (forwardCandidates.empty()) {
      break;  // dead end / true endpoint of the line
    }

    if (forwardCandidates.size() > 1) {
      hitJunction = true;
      junctionCentre = current;
      break;
    }

    previous = current;
    current = forwardCandidates[0];
    path.push_back(current);
  }

  // --- Heading direction at the end of the walk ---
  cv::Point2d headingDir = this->havePrevHeading ? this->prevHeadingDir : cv::Point2d(0, -1);

  if (path.size() >= 2) {
    size_t lookBack = std::min<size_t>(path.size() - 1, 15);
    cv::Point tail = path.back();
    cv::Point head = path[path.size() - 1 - lookBack];

    cv::Point2d dir(tail.x - head.x, tail.y - head.y);
    double norm = std::hypot(dir.x, dir.y);
    if (norm > 1e-3) {
      headingDir = dir / norm;
    }
  }

  this->prevForwardEndpoint = path.back();
  this->havePrevForwardEndpoint = true;
  this->prevHeadingDir = headingDir;
  this->havePrevHeading = true;

  // --- If we reached a junction, decide what to do about it ---
  if (hitJunction) {
    std::vector<std::vector<cv::Point>> greenContours;
    std::vector<cv::Point> greenCenters = this->extractGreen(resized, &greenContours);
    cv::drawContours(processed, greenContours, -1, cv::Scalar(0, 255, 0), 2);

    std::vector<cv::Point> usedGreen;
    GreenTurnDirection decision =
      this->evaluateIntersectionGreen(junctionCentre, headingDir, greenCenters, &usedGreen);

    for (const cv::Point & g : greenCenters) {
      bool used = std::find(usedGreen.begin(), usedGreen.end(), g) != usedGreen.end();
      cv::circle(processed, g, 6, used ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 0, 255), -1);
    }

    cv::drawMarker(processed, junctionCentre, cv::Scalar(255, 0, 255), cv::MARKER_DIAMOND, 16, 2);

    if (decision != GreenTurnDirection::NONE) {
      this->triggerGreenTurn(decision);
      this->writer.write(processed);
      // The turn is about to change the geometry completely; drop the
      // frame-to-frame tracking so we don't try to "continue" the old line.
      this->havePrevForwardEndpoint = false;
      this->havePrevHeading = false;
      return 0.0;
    }
    // decision == NONE: no green here (or an unrecognised marker layout),
    // fall through and just keep following the line through the junction.
  }

  // --- Steering target: pure-pursuit style point a fixed distance ahead
  //     along the traced path ---
  const double targetLookaheadDist = 140.0;
  cv::Point targetPoint = path.back();

  for (const cv::Point & p : path) {
    if (this->calculateDist(p, walkStart) >= targetLookaheadDist) {
      targetPoint = p;
      break;
    }
  }

  cv::polylines(processed, path, false, cv::Scalar(0, 140, 255), 2);
  cv::circle(processed, walkStart, 6, cv::Scalar(255, 0, 0), -1);
  cv::circle(processed, targetPoint, 10, cv::Scalar(255, 255, 0), 2);
  cv::drawMarker(processed, targetPoint, cv::Scalar(255, 255, 0), cv::MARKER_CROSS, 20, 2);

  double dx = targetPoint.x - stationaryPoint.x;
  double dy = stationaryPoint.y - targetPoint.y;  // flip so "straight ahead" is positive dy

  double error = std::atan2(dx, dy);

  this->writer.write(processed);
  cv::imshow("b", processed);
  cv::waitKey(1);

  return error;
}

// Looks at the green markers found near a junction and decides whether the
// robot should carry straight on, turn left/right, or reverse.
//
//  - Markers further than maxGreenRange from the junction are ignored
//    (irrelevant clutter elsewhere in frame).
//  - Markers that project *ahead* of the junction along headingDir --
//    i.e. sit beyond the crossing line, on the far/exit side of the
//    intersection -- are ignored, per spec ("indicators above the
//    horizontal line that it approaches should be ignored").
//  - 0 valid markers -> NONE (go straight through).
//  - 1 valid marker  -> LEFT or RIGHT, whichever side it's on.
//  - 2+ valid markers -> REVERSE (180).
GreenTurnDirection NavigationNode::evaluateIntersectionGreen(
  const cv::Point & junctionCentre, const cv::Point2d & headingDir,
  const std::vector<cv::Point> & greenPoints, std::vector<cv::Point> * usedGreenOut)
{
  // Perpendicular to the heading -- the "horizontal" axis of the junction.
  // Negative = left of the direction of travel, positive = right.
  cv::Point2d perpDir(-headingDir.y, headingDir.x);

  const double maxGreenRange = 220.0;

  std::vector<double> sideProjections;

  for (const cv::Point & g : greenPoints) {
    cv::Point2d rel(g.x - junctionCentre.x, g.y - junctionCentre.y);
    double dist = std::hypot(rel.x, rel.y);

    if (dist > maxGreenRange) {
      continue;
    }

    double forwardProjection = rel.x * headingDir.x + rel.y * headingDir.y;
    if (forwardProjection > 0) {
      // Beyond the crossing line -- on the far/exit side. Ignore.
      continue;
    }

    if (usedGreenOut) {
      usedGreenOut->push_back(g);
    }

    sideProjections.push_back(rel.x * perpDir.x + rel.y * perpDir.y);
  }

  if (sideProjections.size() >= 2) {
    return GreenTurnDirection::REVERSE;
  } else if (sideProjections.size() == 1) {
    return sideProjections[0] < 0 ? GreenTurnDirection::LEFT : GreenTurnDirection::RIGHT;
  }

  return GreenTurnDirection::NONE;
}

// Kicks off the physical turn for a decision from evaluateIntersectionGreen().
// This reuses the existing GREEN_ROTATE / GREEN_MOVE_FORWARD state machine:
// goalResultCallback() picks up once this first rotate goal completes and
// drives the "move forward past the junction" phase, same as before.
void NavigationNode::triggerGreenTurn(GreenTurnDirection direction)
{
  const double baseAngularVel = 100.0;
  const double quarterTurnTime = 5.0;

  switch (direction) {
    case GreenTurnDirection::LEFT:
      this->turnAngularVel = baseAngularVel;
      this->turnDuration = quarterTurnTime;
      RCLCPP_INFO(this->get_logger(), "Green marker turn: LEFT");
      break;
    case GreenTurnDirection::RIGHT:
      this->turnAngularVel = -baseAngularVel;
      this->turnDuration = quarterTurnTime;
      RCLCPP_INFO(this->get_logger(), "Green marker turn: RIGHT");
      break;
    case GreenTurnDirection::REVERSE:
      this->turnAngularVel = baseAngularVel;
      this->turnDuration = quarterTurnTime * 2.0;
      RCLCPP_INFO(this->get_logger(), "Green marker turn: REVERSE (180)");
      break;
    case GreenTurnDirection::NONE:
    default:
      return;
  }

  this->state = GREEN_ROTATE;
  this->sendMovementGoal(0, this->turnAngularVel, this->turnDuration);
}

void NavigationNode::simpleNavigation(cv::Mat & frame)
{
  double error = this->simpleError(frame);

  this->publishError(error);
}

void NavigationNode::advancedNavigation(cv::Mat & frame)
{
  cv::Mat processed = this->processImage(frame);
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

  cv::adaptiveThreshold(
    gray, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, threshSize, 4);

  // cv::threshold(gray, binary, 120, 255, cv::THRESH_BINARY_INV);
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
  cv::ximgproc::thinning(binary, skeleton, cv::ximgproc::THINNING_GUOHALL);

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

std::vector<cv::Point> NavigationNode::extractGreen(
  cv::Mat & image, std::vector<std::vector<cv::Point>> * outContours)
{
  cv::Mat hsv;
  cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

  cv::Scalar lower_green(40, 40, 30);
  cv::Scalar upper_green(100, 255, 255);

  cv::Mat mask;
  cv::inRange(hsv, lower_green, upper_green, mask);

  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
  cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  std::vector<cv::Point> centers;

  for (const auto & contour : contours) {
    double area = cv::contourArea(contour);

    if (area <= 200) {
      continue;
    }

    cv::Moments m = cv::moments(contour);

    if (m.m00 == 0) {
      continue;
    }

    int cX = static_cast<int>(m.m10 / m.m00);
    int cY = static_cast<int>(m.m01 / m.m00);

    centers.push_back(cv::Point(cX, cY));

    if (outContours) {
      outContours->push_back(contour);
    }
  }

  return centers;
}

cv::Point NavigationNode::cvtPoint(cv::Mat & src, cv::Mat & dst, cv::Point point)
{
  double sx = static_cast<double>(dst.cols) / static_cast<double>(src.cols);
  double sy = static_cast<double>(dst.rows) / static_cast<double>(src.rows);

  return cv::Point(cv::saturate_cast<int>(point.x * sx), cv::saturate_cast<int>(point.y * sy));
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
    std::vector<cv::Point> surroundingPoints = this->getSurroundingPoints(point, 3);

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

    if (
      node.pos.x <= 1 || node.pos.y <= 1 || node.pos.x >= image.cols - 2 ||
      node.pos.y >= image.rows - 2) {
      node.screen_edge = true;
    } else {
      node.screen_edge = false;
    }

    foundNodes.push_back(node);
  }

  this->graph.nodes = foundNodes;
}

std::vector<cv::Point> NavigationNode::getSurroundingPoints(cv::Point centre, int radius)
{
  cv::Mat image = this->skeletonizedImage;
  cv::Rect roi(centre.x - 1, centre.y - 1, radius, radius);
  std::vector<cv::Point> surroundingPoints;

  if (centre.x <= 0 || centre.y <= 0 || centre.x >= image.cols - 1 || centre.y >= image.rows - 1) {
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

    Node * src = this->graph.nodeFromID(edges[i].src);
    Node * dst = this->graph.nodeFromID(edges[i].dst);

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
    for (Edge * connectedEdge : this->graph.getConnectedEdges(edges[i].src)) {
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
    edge1.path.insert(edge1.path.end(), edge2.path.begin() + 1, edge2.path.end());
    edge1.dst = edge2.dst;
  } else if (edge1.src == edge2.dst) {
    edge1.path.insert(edge1.path.begin(), edge2.path.begin() + 1, edge2.path.end());
    edge1.src = edge2.src;
  } else if (edge1.dst == edge2.dst) {
    std::reverse(edge2.path.begin(), edge2.path.end());

    edge1.path.insert(edge1.path.end(), edge2.path.begin() + 1, edge2.path.end());
    edge1.dst = edge2.src;
  } else if (edge1.src == edge2.src) {
    std::reverse(edge2.path.begin(), edge2.path.end());
    edge1.path.insert(edge1.path.begin(), edge2.path.begin() + 1, edge2.path.end());
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
  std::vector<cv::Point> surroundingPoints = this->getSurroundingPoints(node.pos, 3);

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
  return std::sqrt(std::pow(point1.x - point2.x, 2) + std::pow(point1.y - point2.y, 2));
}

Node NavigationNode::followToNode(std::vector<cv::Point> & path)
{
  cv::Point current = path[path.size() - 1];
  cv::Point previous;

  if (path.size() > 1) {
    previous = path[path.size() - 2];
  }

  auto it = std::find_if(
    this->graph.nodes.begin(), this->graph.nodes.end(),
    [current](const Node & node) { return node.pos == current; });

  if (it != this->graph.nodes.end()) {
    return *it;
  }

  std::vector<cv::Point> surroundingPoints = this->getSurroundingPoints(current, 3);

  auto it1 = std::find(surroundingPoints.begin(), surroundingPoints.end(), current);

  if (it1 != surroundingPoints.end()) {
    surroundingPoints.erase(it1);
  }

  auto it2 = std::find(surroundingPoints.begin(), surroundingPoints.end(), previous);

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

  for (const Edge * edge : connected) {
    if (edge->dst == current.id) {
      connectedNodes.push_back(edge->src);
    } else {
      connectedNodes.push_back(edge->dst);
    }
  }

  if (connected.size() == 0 || path.size() > this->pathLimit) {
    return;
  }

  std::vector<double> connectedDirs = this->getEdgeDirections(current, connected);

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

      auto src_it = std::find_if(
        this->graph.nodes.begin(), this->graph.nodes.end(),
        [&src](const Node & node) { return node.id == src; });

      auto dst_it = std::find_if(
        this->graph.nodes.begin(), this->graph.nodes.end(),
        [&dst](const Node & node) { return node.id == dst; });

      if (src_it != this->graph.nodes.end() && dst_it != this->graph.nodes.end()) {
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
  std::vector<std::vector<double>> costMatrix = this->trackedGraph.getCostMatrix(this->graph);
  std::vector<int> assignment;

  HungarianAlgorithm().Solve(costMatrix, assignment);
  std::vector<bool> matched(this->graph.nodes.size(), false);
  std::vector<int> newIDs(this->graph.nodes.size(), 0);

  // Update matched nodes
  for (uint32_t i = 0; i < this->trackedGraph.nodes.size(); i++) {
    int assigned = assignment[i];

    if (
      assigned > 0 && static_cast<uint32_t>(assigned) < matched.size() &&
      costMatrix[i][assigned] <= this->gatingThreshold) {
      cv::Mat measurement =
        (cv::Mat_<float>(2, 1) << this->graph.nodes[assigned].pos.x,
         this->graph.nodes[assigned].pos.y);

      // this->trackedGraph.nodes[i].kf.correct(measurement);
      this->trackedGraph.nodes[i].missedFrames = 0;
      this->trackedGraph.nodes[i].age++;
      this->trackedGraph.nodes[i].pos = this->graph.nodes[assigned].pos;
      this->trackedGraph.nodes[i].is_endpoint = this->graph.nodes[assigned].is_endpoint;
      this->trackedGraph.nodes[i].screen_edge = this->graph.nodes[assigned].screen_edge;
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
      [](const TrackedNode & node) { return node.missedFrames > 5; }),
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

      auto connectedIt = std::find_if(
        this->graph.nodes.begin(), this->graph.nodes.end(),
        [&connectedID](const Node & connected) { return connected.id == connectedID; });

      if (connectedIt == this->graph.nodes.end()) {
        RCLCPP_ERROR(
          this->get_logger(), "Couldn't find the other node??? (This should never happen)");
        return;
      }

      int connectedIndex = std::distance(this->graph.nodes.begin(), connectedIt);

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
        if (
          (tracked.src == existingTracked.src && tracked.dst == existingTracked.dst) ||
          (tracked.src == existingTracked.dst && tracked.dst == existingTracked.src)) {
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
      [&src](const TrackedNode & node) { return node.id == src; });

    auto dst_it = std::find_if(
      this->trackedGraph.nodes.begin(), this->trackedGraph.nodes.end(),
      [&dst](const TrackedNode & node) { return node.id == dst; });

    if (src_it != this->trackedGraph.nodes.end() && dst_it != this->trackedGraph.nodes.end()) {
      continue;
    }

    this->trackedGraph.edges.erase(this->trackedGraph.edges.begin() + i);
    i--;
  }
}

std::vector<double> NavigationNode::getEdgeDirections(Node origin, std::vector<Edge *> edges)
{
  std::vector<double> results;

  for (const Edge * edge : edges) {
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

  tracked.angleFromSrc =
    this->calculateAngle(this->graph.nodeFromID(edge.src)->pos, edge.path[this->minEdgeSize - 1]) +
    this->angle;
  tracked.angleFromDst =
    this->calculateAngle(
      this->graph.nodeFromID(edge.dst)->pos, edge.path[edge.path.size() - this->minEdgeSize]) +
    this->angle;
  tracked.path = edge.path;
}

void NavigationNode::findStartingEdge(int & trackingID, TrackedEdge ** currentEdge)
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

  TrackedNode * src = this->trackedGraph.nodeFromID((*currentEdge)->src);
  TrackedNode * dst = this->trackedGraph.nodeFromID((*currentEdge)->dst);

  trackingID = src->pos.y > dst->pos.y ? (*currentEdge)->dst : (*currentEdge)->src;
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

void NavigationNode::findNextTarget(int & trackingID, TrackedEdge ** currentEdge)
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
    double closestDist = this->calculateDist(this->searchLastPoint, closestNode.pos);

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

    TrackedEdge * newEdge =
      this->closestToAngle(closestNode.id, surroundingEdges, this->searchDirection);
    int newTarget = closestNode.id == newEdge->src ? newEdge->dst : newEdge->src;

    this->currentTarget = newTarget;
    this->currentEdge = newEdge;
    this->searchLineBreak = false;
  }

  if (!currentEdge || trackingID < 0) {
    *currentEdge = new TrackedEdge;
    this->findStartingEdge(trackingID, currentEdge);
    return;
  }

  TrackedNode * currentNodePointer = this->trackedGraph.nodeFromID(trackingID);

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
    double currentDir = trackingID == (*currentEdge)->src ? (*currentEdge)->angleFromSrc
                                                          : (*currentEdge)->angleFromDst;
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

  double currentAngle =
    trackingID == (*currentEdge)->src ? (*currentEdge)->angleFromSrc : (*currentEdge)->angleFromDst;

  if (surroundingEdges.size() < 3) {
    double targetAngle = this->addAngles(currentAngle, std::numbers::pi);

    TrackedEdge * closestEdge = this->closestToAngle(trackingID, surroundingEdges, targetAngle);

    *this->currentEdge = *closestEdge;
    this->currentTarget = trackingID == closestEdge->src ? closestEdge->dst : closestEdge->src;

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

  TrackedEdge * leftEdge = this->closestToAngle(trackingID, surroundingEdges, targetLeft);
  TrackedEdge * rightEdge = this->closestToAngle(trackingID, surroundingEdges, targetRight);
  TrackedEdge * straightEdge = this->closestToAngle(trackingID, surroundingEdges, targetStraight);

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
    trackingID = trackingID == (*currentEdge)->src ? (*currentEdge)->dst : (*currentEdge)->src;
    return;
  } else if (greenRight) {
    **currentEdge = *rightEdge;
  } else if (greenLeft) {
    **currentEdge = *leftEdge;
  } else {
    **currentEdge = *straightEdge;
  }

  trackingID = trackingID == (*currentEdge)->src ? (*currentEdge)->dst : (*currentEdge)->src;
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

TrackedEdge * NavigationNode::closestToAngle(
  int currentNode, std::vector<TrackedEdge *> currentEdges, double targetAngle)
{
  TrackedEdge * closestEdge = currentEdges[0];
  double closestAngle =
    currentNode == closestEdge->src ? closestEdge->angleFromSrc : closestEdge->angleFromDst;

  for (TrackedEdge * edge : currentEdges) {
    double angle = currentNode == edge->src ? edge->angleFromSrc : edge->angleFromDst;

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