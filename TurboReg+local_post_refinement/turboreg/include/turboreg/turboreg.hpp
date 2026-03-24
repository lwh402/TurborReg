#ifndef TURBOREG_HPP
#define TURBOREG_HPP

#include <torch/torch.h>
#include <vector>
#include <Eigen/Dense>
#include "rigid_transform.hpp"
#include "model_selection.hpp"
#include <unordered_map>

namespace turboreg
{

    class TurboRegGPU
    {
    public:
        TurboRegGPU(int max_N, float tau_length_consis, int num_pivot, float radiu_nms, float tau_inlier, const std::string &metric_str);
        RigidTransform runRegCXX(torch::Tensor &kpts_src_all, torch::Tensor &kpts_dst_all);
        torch::Tensor runRegCXXReturnTensor(torch::Tensor &kpts_src_all, torch::Tensor &kpts_dst_all);

        // 新增：暴露候选cliques的枢轴边、第三匹配对索引，以及Top1 clique索引
        torch::Tensor get_pivots() const { return pivots; }
        torch::Tensor get_topk_K2() const { return topk_K2; }
        torch::Tensor get_cliques_tensor() const { return cliques_tensor; }
        torch::Tensor get_idx_best_guess() const { return idx_best_guess; }

    private:
        int max_N;               // Maximum number of correspondences
        float tau_length_consis; // \tau
        float radiu_nms;  // Radius for avoiding the instability of the solution
        int num_pivot;    // Number of pivot points, K_1
        float tau_inlier; // Threshold for inlier points. NOTE: just for post-refinement (REF@PointDSC/SC2PCR/MAC)
        bool hard = true; // Flag for hard thresholding. NOTE: just using hard compatibility graph.
        MetricType eval_metric;


        // 新增：记录候选clique相关索引
        torch::Tensor pivots;          // 枢轴边索引 (num_pivot, 2)
        torch::Tensor topk_K2;         // 每个枢轴边的Top2第三匹配对索引 (num_pivot, 2)
        torch::Tensor cliques_tensor;  // 所有候选cliques索引 (num_pivot*2, 3)
        torch::Tensor idx_best_guess;  // Top1 clique的索引 (scalar)
    };

}

#endif // TURBOREG_HPP
