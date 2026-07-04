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

#include <rclcpp/publisher.hpp>
#include <rclcpp/subscription.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include "nav_msgs/msg/odometry.hpp"
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64.hpp>
#include <opencv2/opencv.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <robot_msgs/action/move_time.hpp>
#include <optional>
#include <string>

// ============================================================================
// Vision pipeline types, ported from line_follower_vision.py
// (LineFollowerVision). These back NavigationNode's "simple" navigation
// mode only; the graph-based "advanced" mode below is untouched.
// ============================================================================

// A single foreground run found within one horizontal scan strip.
struct StripSegment
{
  int x_start;
  int x_end;
  double x_center;
  int width;
};

// One horizontal sample band of the strip scan, at row `y`.
struct ScanStrip
{
  int y;
  std::vector<StripSegment> segments;
  bool is_branch = false;         // weak or strong branch/junction signal
  bool is_branch_strong = false;  // strong signal alone, no corroboration needed
};

// A detected line junction (byproduct of the strip scan + connected
// components on the branch-flagged band).
struct JunctionInfo
{
  cv::Point2d center;
  cv::Rect2d box;
};

// A detected green marker blob, classified relative to the junction.
struct GreenBlobInfo
{
  cv::Point2d center;
  double area = 0.0;
  std::vector<cv::Point> contour;
  double forward_proj = 0.0;
  double lateral_proj = 0.0;
  std::string region;  // "near", "far", "unrelated", "no_junction"
  std::string side;    // "left", "right", or "" if not applicable
};

struct Node
{
  int id;
  cv::Point pos; // averaged position after merging
  bool is_endpoint;
  bool screen_edge;
};

struct Edge
{
  int src, dst;                // node IDs
  std::vector<cv::Point> path; // pixel chain along the skeleton
  double length;               // Euclidean arc length

  bool operator==(const Edge & other) const
  {
    return (src == other.src && dst == other.dst) ||
           (src == other.dst && dst == other.src);
  }
};

class Graph {
public:
  std::vector<Node> nodes;
  std::vector<Edge> edges;

  int nextID = 0;

  Node * nodeFromID(int id);
  std::vector<Edge *> getConnectedEdges(int nodeID);
};

struct LocalEdge : Edge {};

struct TrackedNode : Node
{
  TrackedNode(cv::Point pos);
  int age = 0;
  int missedFrames = 0;
  cv::KalmanFilter kf = cv::KalmanFilter(4, 2, 0);
};

struct TrackedEdge : Edge
{
  uint32_t age = 0;
  double angleFromSrc;
  double angleFromDst;
};

class TrackedGraph {
public:
  std::vector<TrackedNode> nodes;
  std::vector<TrackedEdge> edges;

  int nextID = 0;
  int edgePenalty = 0; // TODO Change later once edge detection exists

  TrackedNode * nodeFromID(int id);
  std::vector<TrackedEdge *> getConnectedEdges(int nodeID);

  std::vector<std::vector<double>> getCostMatrix(Graph & graph);
};

enum NavigationType
{
  SIMPLE,
  ADVANCED
};

enum LineFollowState
{
  FOLLOWING,
  TOWER_ROTATE_START,
  TOWER_MOVE,
  TOWER_ROTATE_END,
  GREEN_ROTATE,
  GREEN_MOVE_FORWARD,
  COMPLETE
};

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

class NavigationNode : public rclcpp_lifecycle::LifecycleNode {
public:
  explicit NavigationNode(const rclcpp::NodeOptions & options);

  CallbackReturn on_configure(const rclcpp_lifecycle::State &) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State &) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State &) override;
  CallbackReturn on_cleanup(const rclcpp_lifecycle::State &) override;
  CallbackReturn on_shutdown(const rclcpp_lifecycle::State &) override;

  uint32_t pathLimit;
  uint32_t minEdgeSize;
  uint32_t gatingThreshold;
  double pixelSize;
  cv::Point frameCentre;
  NavigationType navigationType;

private:
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr imageSub;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odomSub;
  std::shared_ptr<rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Float64>> errorPub;
  std::shared_ptr<rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Bool>> lineCompletePub;
  rclcpp::TimerBase::SharedPtr timer_;

  rclcpp_action::Client<robot_msgs::action::MoveTime>::SharedPtr actionClient;

  LineFollowState state;
  cv::Mat currentFrame;

  cv::Point cvtPoint(cv::Mat & src, cv::Mat & dst, cv::Point point);

  void simpleNavigation(cv::Mat & frame);
  void advancedNavigation(cv::Mat & frame);

  void imageCallback(sensor_msgs::msg::Image::SharedPtr msg);
  void odomCallback(nav_msgs::msg::Odometry::SharedPtr msg);
  void goalResponseCallback(
    const rclcpp_action::ClientGoalHandle<robot_msgs::action::MoveTime>::SharedPtr & goalHandle);
  void goalFeedbackCallback(
    rclcpp_action::ClientGoalHandle<robot_msgs::action::MoveTime>::SharedPtr,
    const std::shared_ptr<const robot_msgs::action::MoveTime::Feedback> feedback);
  void goalResultCallback(
    const rclcpp_action::ClientGoalHandle<robot_msgs::action::MoveTime>::WrappedResult & result);
  double simpleError(const cv::Mat & frame);
  void publishError(double error);
  void timerCallback();

  void sendMovementGoal(double vel, double angular_vel, double time);

  cv::Mat processImage(cv::Mat & image);
  cv::Mat applyThreshold(cv::Mat & image, uint32_t threshSize, uint32_t kernelSize);
  cv::Point localToGlobalFrame(cv::Point point);

  // --------------------------------------------------------------------
  // Ported line-following vision pipeline (from line_follower_vision.py)
  // Used exclusively by simpleError() / simpleNavigation().
  // --------------------------------------------------------------------
  cv::Mat visionBinarize(const cv::Mat & frameBgr);
  std::vector<ScanStrip> visionScanStrips(const cv::Mat & binary, int h, int w);
  std::vector<StripSegment> visionFindSegmentsInRow(const std::vector<uint8_t> & rowBool);
  double visionEstimateLineWidth(const std::vector<ScanStrip> & strips);
  void visionFlagBranchStrips(std::vector<ScanStrip> & strips, double baselineWidth);
  std::optional<JunctionInfo> visionDetectJunction(
    std::vector<ScanStrip> & strips, const cv::Mat & binary);
  void visionFitSteeringLine(
    const std::vector<ScanStrip> & strips, bool haveJunction, int w,
    double & m, double & b, std::vector<cv::Point2d> & fitPoints);
  double visionSteeringAngle(double m);
  double visionLineOffset(double m, double b, int h, int w);
  void visionHeadingVectors(double m, cv::Point2d & forward, cv::Point2d & right);
  std::vector<GreenBlobInfo> visionDetectGreenBlobs(const cv::Mat & frameBgr);
  void visionClassifyBlobs(
    std::vector<GreenBlobInfo> & blobs, const std::optional<JunctionInfo> & junction,
    const cv::Point2d & forwardVec, const cv::Point2d & rightVec, double baselineWidth);
  void visionDecide(
    const std::optional<JunctionInfo> & junction,
    const std::vector<GreenBlobInfo> & classified,
    std::string & decision, std::string & decisionReason);
  cv::Mat visionAnnotate(
    const cv::Mat & frameBgr, const std::vector<ScanStrip> & strips, double baselineWidth,
    const std::optional<JunctionInfo> & junction, double m, double b,
    const std::vector<cv::Point2d> & fitPoints, double steeringAngleDeg,
    const std::vector<GreenBlobInfo> & classified, const std::string & decision,
    const std::string & decisionReason, const cv::Point2d & forwardVec);

  void extractNodes();
  void extractEdges();

  std::vector<cv::Point> getSurroundingPoints(cv::Point centre, int radius);
  std::vector<Edge> traceConnectedEdges(Node node);
  Node followToNode(std::vector<cv::Point> & path);

  void removeShortEdges(std::vector<Edge> & edges);
  Edge mergeEdges(Edge edge1, Edge edge2);
  void removeUnconnectedNodes();

  void findNextNode(std::vector<Node> & path);
  double calculateAngle(cv::Point point1, cv::Point point2);
  double calculateDist(cv::Point point1, cv::Point point2);

  void updateGraph();

  std::vector<double> getEdgeDirections(Node origin, std::vector<Edge *> edges);

  void edgeToTracked(const Edge & edge, TrackedEdge & trackedEdge);

  double wrapAngle(double angle);
  double addAngles(double angle1, double angle2);

  void findStartingEdge(int & trackingID, TrackedEdge **currentEdge);
  void findNextTarget(int & trackingID, TrackedEdge **currentEdge);
  TrackedEdge * closestToAngle(
    int currentNode,
    std::vector<TrackedEdge *> currentEdges,
    double targetAngle);

  double searchDistance(cv::Point point);

  cv::Mat rawImage;
  cv::Mat skeletonizedImage;

  Graph graph;
  TrackedGraph trackedGraph;
  std::vector<cv::Point> green;

  int currentTarget = -1;
  TrackedEdge *currentEdge = nullptr;

  double x = 0;
  double y = 0;
  double angle = 0;

  bool searchLineBreak = false;
  int searchLastNode;
  cv::Point searchLastPoint;
  double searchDirection;
  double searchMinDist;

  cv::VideoWriter writer;

  // --------------------------------------------------------------------
  // Vision tuning constants, ported 1:1 from the class-level constants in
  // line_follower_vision.py's LineFollowerVision. See that module's
  // comments for the reasoning behind each value.
  // --------------------------------------------------------------------

  // --- Binarization (line vs background) ---
  static constexpr bool VISION_USE_OTSU = true;
  static constexpr int VISION_FIXED_BINARY_THRESH = 90;  // used only if VISION_USE_OTSU is false

  // --- Region of interest for the strip scan, as a fraction of image height ---
  static constexpr double VISION_ROI_Y_TOP_RATIO = 0.05;
  static constexpr double VISION_ROI_Y_BOTTOM_RATIO = 0.95;

  // --- Multi-strip scan ---
  static constexpr int VISION_NUM_STRIPS = 12;
  static constexpr int VISION_STRIP_THICKNESS_PX = 12;
  static constexpr int VISION_MIN_SEGMENT_WIDTH_PX = 4;

  // --- Branch / junction detection (reuses the strip scan) ---
  static constexpr int VISION_BRANCH_MIN_SEGMENTS = 2;
  static constexpr double VISION_WIDE_SEGMENT_MULTIPLIER = 2.3;
  static constexpr double VISION_STRONG_WIDE_SEGMENT_MULTIPLIER = 3;
  static constexpr int VISION_BRANCH_MIN_STRIP_COUNT = 2;
  static constexpr int VISION_BRANCH_CLUSTER_GAP_STRIPS = 2;
  static constexpr int VISION_BASELINE_STRIP_COUNT = 3;
  static constexpr double VISION_DEFAULT_LINE_WIDTH_PX = 20.0;

  // --- Steering line fit ---
  static constexpr int VISION_MIN_FIT_POINTS = 3;

  // --- Green marker detection (HSV) ---
  static constexpr int VISION_MIN_GREEN_BLOB_AREA = 15;
  static constexpr int VISION_MORPH_KERNEL_SIZE = 3;

  // --- Green blob classification relative to the junction ---
  static constexpr double VISION_NEAR_FORWARD_LIMIT_LINE_WIDTHS = 1.0;
  static constexpr double VISION_MAX_LATERAL_DIST_LINE_WIDTHS = 4.5;
};