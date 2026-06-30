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

#include "line_follow/pid_loop.hpp"
#include <rclcpp/rclcpp.hpp>
#include <lifecycle_msgs/msg/state.hpp>
#include <cstdint>

using std::placeholders::_1;

PIDLoop::PIDLoop(const rclcpp::NodeOptions & options)
: rclcpp_lifecycle::LifecycleNode("pid_loop", options)
{
  this->declare_parameter<double>("kp", 0.1);
  this->declare_parameter<double>("ki", 0.0);
  this->declare_parameter<double>("kd", 0.0);
  this->declare_parameter<int>("default_speed", 50);

  this->kp = this->get_parameter("kp").as_double();
  this->ki = this->get_parameter("ki").as_double();
  this->kd = this->get_parameter("kd").as_double();
  this->defaultSpeed = this->get_parameter("default_speed").as_int();
}

CallbackReturn PIDLoop::on_configure(const rclcpp_lifecycle::State &)
{
  this->twistPub = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
  this->errorSub = this->create_subscription<std_msgs::msg::Float64>("line_error", 10,
    std::bind(&PIDLoop::errorCallback, this, _1));

  return CallbackReturn::SUCCESS;
}

CallbackReturn PIDLoop::on_activate(const rclcpp_lifecycle::State &)
{
  this->twistPub->on_activate();

  this->integral = 0;
  this->lastError = 0;
  this->lastTime = this->get_clock()->now();

  return CallbackReturn::SUCCESS;
}

CallbackReturn PIDLoop::on_deactivate(const rclcpp_lifecycle::State &)
{
  this->twistPub->on_deactivate();

  return CallbackReturn::SUCCESS;
}

CallbackReturn PIDLoop::on_cleanup(const rclcpp_lifecycle::State &)
{
  this->twistPub.reset();
  this->errorSub.reset();

  return CallbackReturn::SUCCESS;
}

CallbackReturn PIDLoop::on_shutdown(const rclcpp_lifecycle::State &)
{
  return CallbackReturn::SUCCESS;
}

void PIDLoop::errorCallback(std_msgs::msg::Float64::SharedPtr msg)
{
  if (this->get_current_state().id() != lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE) {
    return;
  }

  rclcpp::Time currentTime = this->get_clock()->now();
  int64_t dt = (currentTime - this->lastTime).nanoseconds();
  if (dt <= 0) {
    return;
  }

  double error = msg->data;
  this->integral += error * dt;
  double derivative = (error - this->lastError) / dt;

  this->lastError = error;
  this->lastTime = currentTime;

  double sum = error * this->kp + this->integral * this->ki + derivative * this->kd;
  this->sendManualI2C(static_cast<int32_t>(sum));
  return;

  auto twist_msg = geometry_msgs::msg::Twist();
  twist_msg.linear.x = this->defaultSpeed;
  twist_msg.angular.z = error * this->kp + this->integral * this->ki + derivative * this->kd;

  this->twistPub->publish(twist_msg);
}

void PIDLoop::sendManualI2C(int32_t error)
{
  const char * device = "/dev/i2c-1";
  int i2c_fd = open(device, O_RDWR);

  if (i2c_fd < 0) {
    RCLCPP_ERROR(this->get_logger(), "Failed to open the I2C bus: %s", device);
    return;
  }

  int slave_address = 0x67;
  if (ioctl(i2c_fd, I2C_SLAVE, slave_address) < 0) {
    RCLCPP_ERROR(this->get_logger(), "Failed to acquire bus access/talk to slave.");
    close(i2c_fd);
    return;
  }

  uint32_t speed = this->defaultSpeed - 0.3 * std::abs(error);

  uint8_t buffer[13];

  buffer[0] = 0x01;

  buffer[1] = (speed >> 24) & 0xFF;
  buffer[2] = (speed >> 16) & 0xFF;
  buffer[3] = (speed >> 8) & 0xFF;
  buffer[4] = speed & 0xFF;

  buffer[5] = 0;
  buffer[6] = 0;
  buffer[7] = 0;
  buffer[8] = 0;

  buffer[9] = (error >> 24) & 0xFF;
  buffer[10] = (error >> 16) & 0xFF;
  buffer[11] = (error >> 8) & 0xFF;
  buffer[12] = error & 0xFF;

  ssize_t bytes_written = write(i2c_fd, buffer, sizeof(buffer));

  if (bytes_written != sizeof(buffer)) {
    RCLCPP_ERROR(this->get_logger(), "Failed to write to the I2C bus.");
    close(i2c_fd);
    return;
  }

  RCLCPP_INFO(this->get_logger(), "Successfully sent %ld bytes over I2C!", bytes_written);

  close(i2c_fd);
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PIDLoop>(rclcpp::NodeOptions());
  rclcpp::spin(node->get_node_base_interface());
  rclcpp::shutdown();
  return 0;
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(PIDLoop);
