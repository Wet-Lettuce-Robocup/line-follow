cmake_minimum_required(VERSION 3.20)
project(line_follow)

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
find_package(lifecycle_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(core_loop src/core_loop.cpp)

target_include_directories(core_loop PRIVATE
  include)

target_link_libraries(core_loop PRIVATE
  ${lifecycle_msgs_TARGETS}
  rclcpp::rclcpp
  rclcpp_lifecycle::rclcpp_lifecycle
  ${std_msgs_TARGETS}
)

add_executable(navigation src/navigation.cpp)

target_include_directories(navigation PRIVATE
  include
  ${OpenCV_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/hungarian-algorithm-cpp
)

target_link_libraries(navigation PRIVATE
  ${sensor_msgs_TARGETS}
  rclcpp::rclcpp
  rclcpp_lifecycle::rclcpp_lifecycle
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
  # set(ament_cmake_copyright_FOUND TRUE)
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

  set(ament_cmake_copyright_ARGS
    EXCLUDE
    "extern"
    "CONTRIBUTING.md"
    "LICENSE"
    "README.md"
  )

  ament_lint_auto_find_test_dependencies()
endif()

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
