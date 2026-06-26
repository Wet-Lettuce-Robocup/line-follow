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
from launch_ros.actions import LifecycleNode


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('line_follow'), 'config', 'params.yaml'
    )

    return LaunchDescription(
        [
            LifecycleNode(
                package='line_follow',
                executable='core_loop',
                name='core_loop',
                namespace='line_follow',
                output='screen',
                parameters=[config],
            ),
            LifecycleNode(
                package='line_follow',
                executable='navigation',
                name='navigation',
                namespace='line_follow',
                output='screen',
                parameters=[config],
            ),
            LifecycleNode(
                package='line_follow',
                executable='pid_loop',
                name='pid_loop',
                namespace='line_follow',
                output='screen',
                parameters=[config],
            ),
        ]
    )
