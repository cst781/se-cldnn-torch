
add_library(sru_cpu_infer SHARED sru_cpu_infer.cpp)
target_link_libraries(sru_cpu_infer PRIVATE "${TORCH_LIBRARIES}")
target_compile_features(sru_cpu_infer PRIVATE cxx_range_for)
