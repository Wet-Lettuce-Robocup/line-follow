cmake_minimum_required(VERSION 3.20)
project(line_follow)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_library(hungarian-algorithm-cpp STATIC
  extern/hungarian-algorithm-cpp/Hungarian.cpp
  extern/hungarian-algorithm-cpp/Hungarian.h
)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(lifecycle_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# Plugins

# =================================================================
# PLUGIN 1: NAVIGATION
# =================================================================
add_library(navigation_plugin SHARED src/navigation.cpp)

target_link_libraries(navigation_plugin PUBLIC
  ${sensor_msgs_TARGETS}
  ${nav_msgs_TARGETS}
  rclcpp::rclcpp
  rclcpp_lifecycle::rclcpp_lifecycle
  rclcpp_components::component
  ${std_msgs_TARGETS}
  cv_bridge::cv_bridge
  ${OpenCV_LIBRARIES}
  hungarian-algorithm-cpp
)

target_include_directories(navigation_plugin PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/hungarian-algorithm-cpp
)

rclcpp_components_register_nodes(navigation_plugin
  "line_follow::NavigationNode"
)

# =================================================================
# PLUGIN 2: CORE LOOP
# =================================================================
add_library(core_loop_plugin SHARED src/core_loop.cpp)

target_link_libraries(core_loop_plugin PUBLIC
  ${lifecycle_msgs_TARGETS}
  rclcpp::rclcpp
  rclcpp_lifecycle::rclcpp_lifecycle
  rclcpp_components::component
  ${std_msgs_TARGETS}
)

target_include_directories(core_loop_plugin PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

rclcpp_components_register_nodes(core_loop_plugin
  "core_loop::CoreLoop"
)

# =================================================================
# PLUGIN 3: PID LOOP
# =================================================================
add_library(pid_loop_plugin SHARED src/pid_loop.cpp)

target_link_libraries(pid_loop_plugin PUBLIC
  ${lifecycle_msgs_TARGETS}
  ${geometry_msgs_TARGETS}
  rclcpp::rclcpp
  rclcpp_lifecycle::rclcpp_lifecycle
  rclcpp_components::component
  ${std_msgs_TARGETS}
)

target_include_directories(pid_loop_plugin PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

rclcpp_components_register_nodes(pid_loop_plugin
  "pid_loop::PIDLoop"
)

# =================================================================
# INSTALLATION
# =================================================================
install(TARGETS
  navigation_plugin
  core_loop_plugin
  pid_loop_plugin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install executable permissions so your original `main()` blocks work standalone
# via `ros2 run line_follow navigation_plugin` if desired
install(PROGRAMS
  ${CMAKE_CURRENT_BINARY_DIR}/libnavigation_plugin.so
  ${CMAKE_CURRENT_BINARY_DIR}/libcore_loop_plugin.so
  ${CMAKE_CURRENT_BINARY_DIR}/libpid_loop_plugin.so
  DESTINATION lib/${PROJECT_NAME}
)

# Executables

add_executable(core_loop src/core_loop.cpp)

target_include_directories(core_loop PRIVATE
  include)

target_link_libraries(core_loop PRIVATE
  ${lifecycle_msgs_TARGETS}
  rclcpp::rclcpp
  rclcpp_lifecycle::rclcpp_lifecycle
  rclcpp_components::component
  ${std_msgs_TARGETS}
)

add_executable(navigation src/navigation.cpp)

target_include_directories(navigation PRIVATE
  include
  ${OpenCV_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/hungarian-algorithm-cpp
  avdevice
)

target_link_libraries(navigation PRIVATE
  ${sensor_msgs_TARGETS}
  ${nav_msgs_TARGETS}
  rclcpp::rclcpp
  rclcpp_lifecycle::rclcpp_lifecycle
  rclcpp_components::component
  ${std_msgs_TARGETS}
  cv_bridge::cv_bridge
  ${OpenCV_LIBRARIES}
  hungarian-algorithm-cpp
)

add_executable(pid_loop src/pid_loop.cpp)

target_include_directories(pid_loop PRIVATE
  include)

target_link_libraries(pid_loop PRIVATE
  ${lifecycle_msgs_TARGETS}
  ${geometry_msgs_TARGETS}
  rclcpp::rclcpp
  rclcpp_lifecycle::rclcpp_lifecycle
  rclcpp_components::component
  ${std_msgs_TARGETS}
)

install(TARGETS
  core_loop
  navigation
  pid_loop
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/ DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  file(GLOB_RECURSE EXCLUDED_EXTERN_FILES
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "extern/*"
  )

  # 2. Feed the explicit file list directly into the global linter exclusion variable
  list(APPEND AMENT_LINT_AUTO_FILE_EXCLUDE ${EXCLUDED_EXTERN_FILES})

  set(ament_cmake_uncrustify_ARGS EXCLUDE "extern")

  ament_lint_auto_find_test_dependencies()
endif()

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
