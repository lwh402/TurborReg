#include <turboreg/turboreg.hpp>
#include <turboreg/utils_debug.hpp>
#include <turboreg/core_turboreg_gpu.hpp>

namespace py = pybind11;
using namespace turboreg;

TurboRegGPU::TurboRegGPU(int max_N, float tau_length_consis, int num_pivot, float radiu_nms, float tau_inlier, const std::string &metric_str)
    : max_N(max_N), tau_length_consis(tau_length_consis), radiu_nms(radiu_nms), num_pivot(num_pivot), tau_inlier(tau_inlier)
{
    this->eval_metric = string_to_metric_type(metric_str);
}

RigidTransform TurboRegGPU::runRegCXX(torch::Tensor &kpts_src, torch::Tensor &kpts_dst)
{
    // Control the number of keypoints
    auto N_node = std::min(int(kpts_src.size(0)), max_N);
    if (N_node < kpts_src.size(0))
    {
        kpts_src = kpts_src.slice(0, 0, N_node);
        kpts_dst = kpts_dst.slice(0, 0, N_node);
    }

    // Compute C2
    auto src_dist = torch::norm(kpts_src.unsqueeze(1) - kpts_src.unsqueeze(0), 2, -1);
    auto target_dist = torch::norm(kpts_dst.unsqueeze(1) - kpts_dst.unsqueeze(0), 2, -1);
    auto cross_dist = torch::abs(src_dist - target_dist);
    torch::Tensor C2;

    if (!hard)
    {
        C2 = torch::relu(1 - torch::pow(cross_dist / tau_length_consis, 2));
    }
    else
    {
        C2 = (cross_dist < tau_length_consis).to(torch::kFloat32);
    }

    // Apply mask based on distance threshold
    auto mask = (src_dist + target_dist) <= 0.15;
    C2.masked_fill_(mask, 0);

    auto SC2 = torch::matmul(C2, C2) * C2;

    // Select pivots
    auto SC2_up = torch::triu(SC2, 1);                      // Upper triangular matrix, remove diagonal
    auto flat_SC2_up = SC2_up.flatten();                    // Flatten matrix
    auto topk_result = torch::topk(flat_SC2_up, num_pivot); // Select top-K elements
    auto scores_topk = std::get<0>(topk_result);            // Top-K scores
    auto idx_topk = std::get<1>(topk_result);               // Top-K indices

    auto pivots = torch::stack({(idx_topk / N_node).to(torch::kLong),
                                (idx_topk % N_node).to(torch::kLong)},
                               1);

    // Calculate threshold
    auto SC2_for_search = SC2_up.clone();

    // =========================================================
    // 阶段 III：无截断局部一阶度量聚合 (Truncation-Free Local Degree)
    // =========================================================
    
    // 1. 邻域掩码张量构建 (Neighborhood Mask Construction)
    auto pivot_i_indices = pivots.select(/*dim=*/1, /*index=*/0); // [K1]
    auto pivot_j_indices = pivots.select(/*dim=*/1, /*index=*/1); // [K1]

    auto C2_i = C2.index_select(/*dim=*/0, pivot_i_indices); // [K1, N]
    auto C2_j = C2.index_select(/*dim=*/0, pivot_j_indices); // [K1, N]

    // 提取严格隶属于局部子集 \mathcal{N}(i, j) 的拓扑节点
    auto N_ij_mask = C2_i * C2_j; // [K1, N]

    // 2. 全局矩阵乘法降维 (Dimensionality Reduction via GEMM)
    // 利用全局图 C2 隐式计算每个候选点在局部子集中的一阶 Degree
    auto local_degrees = torch::matmul(N_ij_mask, C2); // [K1, N]

    // 3. 拓扑硬约束惩罚 (Topological Hard Constraint Penalty)
    // 彻底切断外部空间节点 (Outliers) 侵入 TurboClique 的代数路径
    local_degrees.masked_fill_(N_ij_mask == 0, -1e9);

    // 4. 并行张量排序 (Parallel Tensor Sorting)
    int K2 = 2; 
    K2 = std::min(K2, (int)local_degrees.size(1));

    auto topk_result_row = torch::topk(local_degrees, K2, /*dim=*/1);
    auto topk_K2 = std::get<1>(topk_result_row); // [K1, K2]

    // =========================================================

    // ---------------------------------------------------------
    // 阶段 IV：团结构组装与模型选择 (Clique Assembly & Selection)
    // ---------------------------------------------------------
    int num_pivots = pivots.size(0);
    auto cliques_tensor = torch::zeros({num_pivots * 2, 3}, torch::kInt32).to(torch::kCUDA);

    // 组装上半部分 Cliques
    cliques_tensor.index_put_({torch::indexing::Slice(0, num_pivots), torch::indexing::Slice(0, 2)}, pivots);
    cliques_tensor.index_put_({torch::indexing::Slice(0, num_pivots), 2}, topk_K2.index({torch::indexing::Slice(), 0}));

    // 组装下半部分 Cliques
    cliques_tensor.index_put_({torch::indexing::Slice(num_pivots, 2 * num_pivots), torch::indexing::Slice(0, 2)}, pivots);
    cliques_tensor.index_put_({torch::indexing::Slice(num_pivots, 2 * num_pivots), 2}, topk_K2.index({torch::indexing::Slice(), 1}));

    torch::Tensor best_in_num, best_trans, res, cliques_wise_trans, cliquewise_in_num;
    ModelSelection model_selector(this->eval_metric, this->tau_inlier);
    verificationV2Metric(cliques_tensor, kpts_src, kpts_dst, model_selector,
                         best_in_num, best_trans, res,
                         cliques_wise_trans);

    // 刚体变换矩阵的非线性优化 (Post-Refinement)
    torch::Tensor refined_trans = post_refinement(best_trans, kpts_src, kpts_dst, 20, this->tau_inlier);
    RigidTransform trans_final(refined_trans);
    
    return trans_final;
    

}

torch::Tensor TurboRegGPU::runRegCXXReturnTensor(torch::Tensor &kpts_src_all, torch::Tensor &kpts_dst_all)
{
    return runRegCXX(kpts_src_all, kpts_dst_all).getTransformation();
}