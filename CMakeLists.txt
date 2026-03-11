cmake_minimum_required(VERSION 3.12.0)
project(wave_eq_2D LANGUAGES CXX C)

include(../common/cmake-common.cmake)

add_executable(exercise-wave src/exercise-wave.cpp src/WaveEquation.cpp)
deal_ii_setup_target(exercise-wave)
