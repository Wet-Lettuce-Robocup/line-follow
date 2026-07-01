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

#include "line_follow/core_loop.hpp"

#include <rclcpp/executors.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/utilities.hpp>
#include "lifecycle_msgs/msg/transition.hpp"
#include "lifecycle_msgs/srv/change_state.hpp"

#include <memory>

CoreLoop::CoreLoop(const rclcpp::NodeOptions & options)
: rclcpp_lifecycle::LifecycleNode("core_loop", options)
{
  RCLCPP_INFO(this->get_logger(), "test");

  this->navigation_client = nullptr;
  this->pid_loop_client = nullptr;
}

void CoreLoop::transitionClient(
  std::shared_ptr<rclcpp::Client<lifecycle_msgs::srv::ChangeState>> client,
  uint8_t transition)
{
  auto request = std::make_shared<lifecycle_msgs::srv::ChangeState::Request>();
  request->transition.id = transition;

  client->async_send_request(request,
    [this](rclcpp::Client<lifecycle_msgs::srv::ChangeState>::SharedFuture future) {
      auto response = future.get();
      if (response->success) {
        RCLCPP_INFO(this->get_logger(), "Transition successful");
      }
        });
}

CallbackReturn CoreLoop::on_configure(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(this->get_logger(), "configuring");

  this->navigation_client =
    this->create_client<lifecycle_msgs::srv::ChangeState>("navigation/change_state");

  if (!this->navigation_client->wait_for_service(std::chrono::seconds(-1))) {
    RCLCPP_ERROR(this->get_logger(), "Navigation service not available");
    return CallbackReturn::FAILURE;
  }

  this->pid_loop_client =
    this->create_client<lifecycle_msgs::srv::ChangeState>("pid_loop/change_state");

  if (!this->pid_loop_client->wait_for_service(std::chrono::seconds(-1))) {
    RCLCPP_ERROR(this->get_logger(), "PID manager service not available");
    return CallbackReturn::FAILURE;
  }

  this->transitionClient(this->navigation_client,
    lifecycle_msgs::msg::Transition::TRANSITION_CONFIGURE);
  this->transitionClient(this->pid_loop_client,
    lifecycle_msgs::msg::Transition::TRANSITION_CONFIGURE);

  return CallbackReturn::SUCCESS;
}

CallbackReturn CoreLoop::on_activate(const rclcpp_lifecycle::State &)
{
  this->transitionClient(this->navigation_client,
    lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE);
  this->transitionClient(this->pid_loop_client,
    lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE);

  RCLCPP_INFO(this->get_logger(), "activated");

  return CallbackReturn::SUCCESS;
}

CallbackReturn CoreLoop::on_deactivate(const rclcpp_lifecycle::State &)
{
  this->transitionClient(this->navigation_client,
    lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE);
  this->transitionClient(this->pid_loop_client,
    lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE);

  RCLCPP_INFO(this->get_logger(), "deactivated");

  return CallbackReturn::SUCCESS;
}

CallbackReturn CoreLoop::on_cleanup(const rclcpp_lifecycle::State &)
{
  this->transitionClient(this->navigation_client,
    lifecycle_msgs::msg::Transition::TRANSITION_CLEANUP);
  this->transitionClient(this->pid_loop_client,
    lifecycle_msgs::msg::Transition::TRANSITION_CLEANUP);

  RCLCPP_INFO(this->get_logger(), "cleanup");

  return CallbackReturn::SUCCESS;
}

CallbackReturn CoreLoop::on_shutdown(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(this->get_logger(), "shutdown");

  return CallbackReturn::SUCCESS;
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CoreLoop>(rclcpp::NodeOptions());
  rclcpp::spin(node->get_node_base_interface());
  rclcpp::shutdown();
  return 0;
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(CoreLoop);
