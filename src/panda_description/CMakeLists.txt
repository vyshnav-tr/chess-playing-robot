cmake_minimum_required(VERSION 3.10.2)
project(ros_arm_panda_description)
find_package(ament_cmake REQUIRED)

ament_package()

install(DIRECTORY meshes urdf config launch worlds DESTINATION share/${PROJECT_NAME})
