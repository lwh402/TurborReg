# turboreg lib
add_library(turboreg SHARED
    src/turboreg.cpp
    src/rigid_transform.cpp
    src/core_turboreg_gpu.cpp
    src/utils_pcr.cpp
    src/model_selection.cpp
)
target_link_libraries(turboreg "${TORCH_LIBRARIES}")

