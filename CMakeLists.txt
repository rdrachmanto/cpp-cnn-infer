cmake_minimum_required(VERSION 3.18)
project(libtorch_opencv_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(test_app main.cpp)

target_link_libraries(test_app
  PRIVATE
    "${TORCH_LIBRARIES}"
    ${OpenCV_LIBS}
)

target_compile_options(test_app PRIVATE ${TORCH_CXX_FLAGS})
