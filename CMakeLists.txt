cmake_minimum_required(VERSION 4.0)
project(GPU_Programming)

set(CMAKE_CXX_STANDARD 14)

add_executable(GPU_Programming
        Chapters/chapter-1/timeTest-for-newton.cpp
        Chapters/chapter-1/Timer.h)
