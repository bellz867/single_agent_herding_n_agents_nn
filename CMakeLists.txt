cmake_minimum_required(VERSION 2.8.3)
project(single_agent_herding_n_agents_nn)

find_package(catkin REQUIRED)

catkin_package()

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  bebop_msgs
  geometry_msgs
  roscpp
  sensor_msgs
  std_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(herding_nn src/herding.cpp)
add_dependencies(herding_nn ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(herding_nn ${catkin_LIBRARIES})

add_executable(herding_nn_des_odom src/herding_des_odom.cpp)
add_dependencies(herding_nn_des_odom ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(herding_nn_des_odom ${catkin_LIBRARIES})

add_executable(herding_arbiter_nn src/arbiter.cpp)
add_dependencies(herding_arbiter_nn ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(herding_arbiter_nn ${catkin_LIBRARIES})

