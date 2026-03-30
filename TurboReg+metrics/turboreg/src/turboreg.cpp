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
    auto mask = (src_dist + target_dist) <= radiu_nms;
    C2.masked_fill_(mask, 0);

    auto SC2 = torch::matmul(C2, C2) * C2;
    
    /*
    // =========================================================================
    // GF-NMS 双空间联合筛选逻辑替换区
    // =========================================================================
    
    // 阶段 I：关系矩阵实例化。将 SC2 视作全局邻接矩阵 A
    auto A = SC2; 

    // 阶段 II：图拉普拉斯滤波，计算结构重要性分数 s^{GF}
    auto D_A = A.sum(1);                        // 度向量 D(A), shape: [N]
    auto D_A_sq = D_A * D_A;                            // D_A 乘度向量
    auto A_x = torch::matmul(A, D_A.unsqueeze(1)).squeeze(1); // 邻接矩阵乘度向量
    
    auto s_raw = D_A_sq - A_x;                          // (D_A - A) * D(A)
    auto s_min = s_raw.min();
    auto s_max = s_raw.max();
    auto s_GF = (s_raw - s_min) / (s_max - s_min + 1e-8); // MinMax 归一化

    // 阶段 III：物理空间与拓扑空间的联合抑制
    auto sort_res = torch::sort(s_GF, -1, true);
    auto sorted_indices = std::get<1>(sort_res);

    // 将数据转移至 CPU 以便进行高效的序列化屏蔽操作
    auto sorted_indices_cpu = sorted_indices.cpu();
    auto src_dist_cpu = src_dist.cpu();
    auto sorted_indices_acc = sorted_indices_cpu.accessor<int64_t, 1>();
    auto src_dist_acc = src_dist_cpu.accessor<float, 2>();

    std::vector<int64_t> seed_indices;
    std::vector<bool> valid_mask(N_node, true);
    std::vector<bool> is_picked(N_node, false);

    // 步骤 1：利用 s^{GF} 在物理 3D 空间进行标准 NMS (保证空间骨架分布)
    for (int i = 0; i < N_node; ++i) {
        int64_t idx = sorted_indices_acc[i];
        if (valid_mask[idx]) {
            seed_indices.push_back(idx);
            is_picked[idx] = true;
            if (seed_indices.size() >= num_pivot) break;

            // 根据物理半径消除过密聚集簇
            for (int j = 0; j < N_node; ++j) {
                if (src_dist_acc[idx][j] < radiu_nms) {
                    valid_mask[j] = false;
                }
            }
        }
    }

    // 步骤 2：若达到物理瓶颈种子点仍不足，提取纯拓扑空间的高阶内点 (补充高密区域)
    if (seed_indices.size() < num_pivot) {
        for (int i = 0; i < N_node; ++i) {
            int64_t idx = sorted_indices_acc[i];
            if (!is_picked[idx]) {
                seed_indices.push_back(idx);
                is_picked[idx] = true;
                if (seed_indices.size() >= num_pivot) break;
            }
        }
    }

    // 将挑选出的高优种子点 (Nodes) 转化为 TurboReg 期望的支撑边 (Pivots)
    auto seeds_tensor = torch::tensor(seed_indices, torch::dtype(torch::kLong).device(SC2.device()));
    auto SC2_seeds = SC2.index_select(0, seeds_tensor);                   // [num_pivot, N]
    auto best_neighbors = std::get<1>(torch::max(SC2_seeds, 1));  // 选取连通性最强的邻居点 [num_pivot]
    
    // 生成二维支撑边，维度: [num_pivot, 2]
    auto pivots = torch::stack({seeds_tensor, best_neighbors}, 1);

    // =========================================================================
    // 恢复 TurboReg 原版 Clique 搜寻逻辑
    // =========================================================================

    auto SC2_for_search = torch::triu(SC2, 1);


    //Find 3-cliques
    */


    // Select pivots
    auto SC2_up = torch::triu(SC2, 1);                      // Upper triangular matrix, remove diagonal
    auto flat_SC2_up = SC2_up.flatten();                    // Flatten matrix
    auto topk_result = torch::topk(flat_SC2_up, num_pivot); // Select top-K elements
    auto scores_topk = std::get<0>(topk_result);            // Top-K scores
    auto idx_topk = std::get<1>(topk_result);               // Top-K indices

    auto pivots = torch::stack({(idx_topk / N_node).to(torch::kLong),
                                (idx_topk % N_node).to(torch::kLong)},
                               1);
    // 新增：赋值成员变量 
    this->pivots = pivots;    
    
    // Calculate threshold
    auto SC2_for_search = SC2_up.clone();

    // Find 3-cliques
    auto SC2_pivot_0 = SC2_for_search.index_select(0, pivots.select(1, 0)) > 0;
    auto SC2_pivot_1 = SC2_for_search.index_select(0, pivots.select(1, 1)) > 0;
    auto indic_c3_torch = SC2_pivot_0 & SC2_pivot_1;

    auto SC2_pivots = SC2_for_search.index({pivots.select(1, 0), pivots.select(1, 1)});

    // Calculate scores for each 3-clique using broadcasting
    auto SC2_ADD_C3 = SC2_pivots.unsqueeze(1) +
                      (SC2_for_search.index_select(0, pivots.select(1, 0)) +
                       SC2_for_search.index_select(0, pivots.select(1, 1)));

    // Mask the C3 scores
    auto SC2_C3 = SC2_ADD_C3 * indic_c3_torch.to(torch::kFloat32);

    /*
    // 原始代码：计算全局一阶打分 SC2_C3
    // ---------------------------------------------------------
    auto SC2_ADD_C3 = SC2_pivots.unsqueeze(1) +
                      (SC2_for_search.index_select(0, pivots.select(1, 0)) +
                       SC2_for_search.index_select(0, pivots.select(1, 1)));

    // Mask the C3 scores
    auto SC2_C3 = SC2_ADD_C3 * indic_c3_torch.to(torch::kFloat32);

    // =========================================================
    // 方案 A 注入点：局部固定尺寸 SC2 聚合 (Local SC2 Aggregation)
    // =========================================================

    // Step 1: Hardware-Aligned Parameterization
    const int WARP_SIZE = 32;          // Hardware-level Warp size for coalesced memory access
    const int NUM_PIVOTS_IN_CLIQUE = 2; // Fixed dimension occupied by pivot pair (i, j)
    int M = WARP_SIZE - NUM_PIVOTS_IN_CLIQUE; // M = 30
    M = std::min(M, (int)SC2_C3.size(1));     // Boundary safety check

    // Step 2: Coarse Selection
    // Extract Top-M candidate nodes based on global 1st-order compatibility scores
    auto topM_result = torch::topk(SC2_C3, M, 1);
    auto Z = std::get<1>(topM_result);        // [K1, M]
    auto Z_scores = std::get<0>(topM_result); // [K1, M]

    // Step 3: Topological Completion
    // Construct the 32-dimensional local subspace tensor by concatenating pivot nodes and candidates
    auto pivot_i = pivots.select(1, 0).unsqueeze(1); // [K1, 1]
    auto pivot_j = pivots.select(1, 1).unsqueeze(1); // [K1, 1]
    auto Z_ext = torch::cat({pivot_i, pivot_j, Z}, 1); // [K1, 32]

    // Step 4: Dynamic Local Tensor Extraction
    // Broadcast indices to construct adjacency sub-matrices mapping O(1) concurrent reads
    auto Z_row = Z_ext.unsqueeze(2).expand({-1, -1, WARP_SIZE}); // [K1, 32, 32]
    auto Z_col = Z_ext.unsqueeze(1).expand({-1, WARP_SIZE, -1}); // [K1, 32, 32]
    auto G_local = C2.index({Z_row, Z_col}); // [K1, 32, 32]

    // Step 5: Local SC^2 via BMM
    // Compute localized 2nd-order spatial compatibility in strict O(K1 * 32^3) time complexity
    auto local_SC2 = torch::bmm(G_local, G_local) * G_local; // [K1, 32, 32]

    // Step 6: Edge-Centric Local Feature Aggregation
    // Calculate Score(z) = G_sub(z, i) + G_sub(z, j) + G_sub(i, j)
    // 6.1 Extract SC2(z, i): Slice rows [2, 32), select column 0
    auto SC2_zi = local_SC2.slice(1, 2, WARP_SIZE).select(2, 0); // [K1, 30]
    // 6.2 Extract SC2(z, j): Slice rows [2, 32), select column 1
    auto SC2_zj = local_SC2.slice(1, 2, WARP_SIZE).select(2, 1); // [K1, 30]
    // 6.3 Extract SC2(i, j): Select row 0, select column 1 (new dim 1), align dimension
    auto SC2_ij = local_SC2.select(1, 0).select(1, 1).unsqueeze(1); // [K1, 1]
    // 6.4 Parallel Summation
    auto local_scores = SC2_zi + SC2_zj + SC2_ij; // [K1, 30]

    // Step 7: Masking Penalty
    // Filter out topologically incompatible phantom nodes to prevent Top-K overflow
    auto valid_mask = Z_scores > 0;
    local_scores.masked_fill_(~valid_mask, -1e9);

    // Step 8: Fine-Grained Sorting and Index Projection
    int K2 = 2; // Default constraint for TurboClique generation
    K2 = std::min(K2, M);
    // Extract the optimum K2 node indices within the localized topological graph
    auto topk2_res = torch::topk(local_scores, K2, 1);
    auto topk2_local_idx = std::get<1>(topk2_res); // [K1, K2]

    // Project the localized index space back to the global index space
    auto topk_K2 = Z.gather(1, topk2_local_idx); // [K1, K2]   

    // ---------------------------------------------------------
    // 原始代码：构建最终的 Cliques Tensor
    // ---------------------------------------------------------
    int num_pivots = pivots.size(0);
    auto cliques_tensor = torch::zeros({num_pivots * 2, 3}, torch::kInt32).to(torch::kCUDA);
    */

    /*
    //原始代码
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
    
    // ============================================================================
    // 方案B:Truncation-Free Local First-Order Degree Aggregation for TurboClique
    // ============================================================================

    // Step 1: Construct Neighborhood Mask Matrix
    // Extract the indices of pivot nodes i (column 0) and j (column 1)
    auto pivot_i_indices = pivots.select(1, 0); // [K1]
    auto pivot_j_indices = pivots.select(1, 1); // [K1]

    // Extract the corresponding row vectors from the global compatibility graph C2
    auto C2_i = C2.index_select(0, pivot_i_indices); // [K1, N]
    auto C2_j = C2.index_select(0, pivot_j_indices); // [K1, N]

    // Compute the full-scale neighborhood boolean indicator matrix.
    // Since C2 is a 0/1 binary matrix, element-wise multiplication acts as a logical AND,
    // strictly isolating the compatible nodes y \in \mathcal{N}(i, j).
    auto N_ij_mask = C2_i * C2_j; // [K1, N]

    // Step 2: Extract Local Degree via Global MatMul
    // Execute a dense matrix multiplication to calculate local first-order degrees.
    // The operation automatically masks out all edges outside \mathcal{N}(i, j).
    // Element (k, z) perfectly equals the sum of connections of node z within the k-th pivot's local subset.
    auto local_degrees = torch::matmul(N_ij_mask, C2); // [K1, N]

    // Step 3: Masking Penalty
    // Enforce strict graph topology constraints: only candidates strictly belonging 
    // to \mathcal{N}(i, j) are geometrically eligible to form a 3-Clique.
    // Phantom/incompatible nodes are penalized algebraically with -1e9 to sink their scores.
    local_degrees.masked_fill_(N_ij_mask == 0, -1e9);

    // Step 4: Fine-Grained Sorting
    int K2 = 2; // Default number of candidate nodes to complete the 3-Cliques
    K2 = std::min(K2, (int)local_degrees.size(1)); // Boundary safety constraint

    // Extract the top-K2 globally indexed nodes possessing the highest local degrees
    auto topk_result = torch::topk(local_degrees, K2, 1);
    auto topk_K2 = std::get<1>(topk_result); // [K1, K2]
    */





    // Get top-2 indices for each row
    auto topk_result_row = torch::topk(SC2_C3, /*k=*/2, /*dim=*/1);
    auto topk_K2 = std::get<1>(topk_result_row); // Top-K indices
    // 新增：赋值成员变量
    this->topk_K2 = topk_K2; 

    // Initialize cliques tensor, size (num_pivots*2, 3)
    int num_pivots = pivots.size(0);
    auto cliques_tensor = torch::zeros({num_pivots * 2, 3}, torch::kInt32).to(torch::kCUDA);

    // Upper part
    cliques_tensor.index_put_({torch::indexing::Slice(0, num_pivots), torch::indexing::Slice(0, 2)}, pivots);
    cliques_tensor.index_put_({torch::indexing::Slice(0, num_pivots), 2}, topk_K2.index({torch::indexing::Slice(), 0}));

    // Lower part
    cliques_tensor.index_put_({torch::indexing::Slice(num_pivots, 2 * num_pivots), torch::indexing::Slice(0, 2)}, pivots);
    cliques_tensor.index_put_({torch::indexing::Slice(num_pivots, 2 * num_pivots), 2}, topk_K2.index({torch::indexing::Slice(), 1}));
    // 新增：赋值成员变量
    this->cliques_tensor = cliques_tensor; 

    torch::Tensor best_in_num, best_trans, res, cliques_wise_trans, cliquewise_in_num;
    ModelSelection model_selector(this->eval_metric, this->tau_inlier);
    verificationV2Metric(cliques_tensor, kpts_src, kpts_dst, model_selector,
                         best_in_num, best_trans, res,
                         cliques_wise_trans);

    // 新增：记录Top1 clique的索引
    this->idx_best_guess = model_selector.calculate_best_clique(cliques_wise_trans, kpts_src, kpts_dst);

    // Post refinement
    torch::Tensor refined_trans = post_refinement(best_trans, kpts_src, kpts_dst, 20, this->tau_inlier);
    RigidTransform trans_final(refined_trans);
    return trans_final;


}

torch::Tensor TurboRegGPU::runRegCXXReturnTensor(torch::Tensor &kpts_src_all, torch::Tensor &kpts_dst_all)
{
    return runRegCXX(kpts_src_all, kpts_dst_all).getTransformation();
}