#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <iostream>
#include <turboreg/turboreg.hpp>

namespace py = pybind11;
using namespace turboreg;

void bind_turboreg_gpu(py::module &m)
{
     py::class_<TurboRegGPU>(m, "TurboRegGPU")
         .def(py::init<int, float, int, float, float, const std::string &>(),
              py::arg("max_N"),
              py::arg("tau_length_consis"),
              py::arg("num_pivot"),
              py::arg("radiu_nms"),
              py::arg("tau_inlier"),
              py::arg("metric_str"))
         .def("run_reg", &TurboRegGPU::runRegCXXReturnTensor,
              py::arg("kpts_src_all"), py::arg("kpts_dst_all"),
              "Perform registration and return RigidTransform")
         // 新增：绑定获取索引的函数
         .def("get_pivots", &TurboRegGPU::get_pivots, "Get pivot edges indices (num_pivot, 2)")
         .def("get_topk_K2", &TurboRegGPU::get_topk_K2, "Get top-2 third match indices for each pivot edge (num_pivot, 2)")
         .def("get_cliques_tensor", &TurboRegGPU::get_cliques_tensor, "Get all candidate cliques indices (num_pivot*2, 3)")
         .def("get_idx_best_guess", &TurboRegGPU::get_idx_best_guess, "Get Top1 clique index (scalar)");
}

PYBIND11_MODULE(turboreg_gpu, m)
{
     m.doc() = "Python bindings for TurboRegGPU class using pybind11 and LibTorch";
     bind_turboreg_gpu(m);
}