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
#include "nav_msgs/msg/odometry.hpp"
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64.hpp>
#include <opencv2/opencv.hpp>
#include <nav_msgs/msg/odometry.hpp>

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
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr silverSub;
  std::shared_ptr<rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Float64>> errorPub;
  std::shared_ptr<rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Bool>> lineCompletePub;

  LineFollowState state;

  cv::Point cvtPoint(cv::Mat & src, cv::Mat & dst, cv::Point point);

  void simpleNavigation(cv::Mat & frame);
  void advancedNavigation(cv::Mat & frame);

  void imageCallback(sensor_msgs::msg::Image::SharedPtr msg);
  void odomCallback(nav_msgs::msg::Odometry::SharedPtr msg);
  void silverCallback(std_msgs::msg::Bool::SharedPtr msg);
  double simpleError(const cv::Mat & frame);
  void publishError(double error);

  cv::Mat processImage(cv::Mat & image);
  cv::Mat applyThreshold(cv::Mat & image, uint32_t threshSize, uint32_t kernelSize);
  cv::Point localToGlobalFrame(cv::Point point);

  std::vector<cv::Point> extractGreen(cv::Mat & image);
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
};
