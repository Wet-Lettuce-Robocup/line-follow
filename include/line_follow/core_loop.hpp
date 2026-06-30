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

#include <rclcpp/client.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <lifecycle_msgs/srv/change_state.hpp>
#include <memory>

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

class CoreLoop : public rclcpp_lifecycle::LifecycleNode
{
public:
  explicit CoreLoop(const rclcpp::NodeOptions & options);

  CallbackReturn on_configure(const rclcpp_lifecycle::State &) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State &) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State &) override;
  CallbackReturn on_cleanup(const rclcpp_lifecycle::State &) override;
  CallbackReturn on_shutdown(const rclcpp_lifecycle::State &) override;

private:
  void transitionClient(
    std::shared_ptr<rclcpp::Client<lifecycle_msgs::srv::ChangeState>> client,
    uint8_t transition);
  std::shared_ptr<rclcpp::Client<lifecycle_msgs::srv::ChangeState>> navigation_client;
  std::shared_ptr<rclcpp::Client<lifecycle_msgs::srv::ChangeState>> pid_loop_client;
};
