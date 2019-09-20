cmake_minimum_required(VERSION 3.0)

project(test)
set(CMAKE_PREFIX_PATH /home/yxhu/work_nfs3/workspace/torch_inference/enhance_for_wechat/third_party/libtorch/)
find_package(Torch REQUIRED)

add_subdirectory(lib)

add_executable(example-app test.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
target_link_libraries(example-app -Wl,--no-as-needed sru_cpu_infer)
set_property(TARGET example-app PROPERTY CXX_STANDARD 11)
