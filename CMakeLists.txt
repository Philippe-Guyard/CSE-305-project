cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 17)
project(main)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -pthread -std=c++17")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(SOURCES main.cpp
            graph.hpp)

add_executable(main ${SOURCES})