# Copyright (C) 2026  William D'Olier
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('line_follow'), 'config', 'params.yaml'
    )

    core_loop = ComposableNode(
        package='line_follow',
        plugin='core_loop::CoreLoop',
        name='core_loop',
        namespace='line_follow',
        parameters=[config],
    )
    navigation = ComposableNode(
        package='line_follow',
        plugin='navigation::NavigationNode',
        name='navigation',
        namespace='line_follow',
        parameters=[config],
    )
    pid_loop = ComposableNode(
        package='line_follow',
        plugin='pid_loop::PIDLoop',
        name='pid_loop',
        namespace='line_follow',
        parameters=[config],
    )

    container = ComposableNodeContainer(
        name='line_follow_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[core_loop, navigation, pid_loop],
        output='screen',
    )

    return LaunchDescription([container])
