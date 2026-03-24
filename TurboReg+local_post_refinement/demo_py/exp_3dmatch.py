import time
from typing import Literal
import tyro
import torch
from dataclasses import dataclass
import numpy as np
import pandas as pd
from datetime import datetime

from .dataset_3dmatch import TDMatchFCGFAndFPFHDataset
from demo_py.utils_pcr import (
    compute_transformation_error, numpy_to_torch32,
    get_clique_pp_tm_indices, judge_clique_pp_inlier, judge_clique_tm_inlier,split_clique_to_pp_tm
)

import turboreg_gpu 


@dataclass
class Args:
    # Dataset path
    dir_dataset: str
    desc: Literal["fpfh", "fcgf"] = "fcgf"
    dataname: Literal["3DMatch", "3DLoMatch"] = "3DLoMatch"

    # TurboRegGPU Initialization Parameters
    max_N: int = 7000
    tau_length_consis: float = 0.012
    num_pivot: int = 2000
    radiu_nms: float = 0.15
    tau_inlier: float = 0.1

    metric_str: Literal["IN", "MSE", "MAE"] = "IN" 
    calc_metrics: bool = False

# 新增：初始化指标统计变量
# 重写：指标统计类（改为每个点云对单独计算）
class MetricStats:
    def __init__(self):
        pass

    def compute_single_sample_metrics(self, cliques_tensor, kpts_src, kpts_dst, trans_gt, inlier_thresh, dataset, top1_idx):
        """
        计算单个点云对的6个指标
        返回：字典，包含6个指标值
        """
        total_candidates = cliques_tensor.shape[0]
        if total_candidates == 0:
            return {
                "All-PP-IR": 0.0,
                "All-TM-IR": 0.0,
                "All-Clique-IR": 0.0,
                "Top1-PP": False,
                "Top1-TM": False,
                "Top1-Clique": False
            }

        # ------------------ 统计全候选指标（1-3） ------------------
        all_pp_inlier_count = 0
        all_tm_inlier_count = 0

        for clique_idx in range(total_candidates):
            clique = cliques_tensor[clique_idx]
            pp_indices, tm_index = split_clique_to_pp_tm(clique)
            
            # 判定枢轴边是否全内点
            is_pp_inlier = judge_clique_pp_inlier(pp_indices, kpts_src, kpts_dst, trans_gt, inlier_thresh, dataset)
            if is_pp_inlier:
                all_pp_inlier_count += 1
                # 判定第三匹配对是否为内点
                is_tm_inlier = judge_clique_tm_inlier(tm_index, kpts_src, kpts_dst, trans_gt, inlier_thresh, dataset)
                if is_tm_inlier:
                    all_tm_inlier_count += 1

        # 计算指标1-3
        all_pp_ir = all_pp_inlier_count / total_candidates
        # 指标2分母改为：当前点云对总候选数
        all_tm_ir = all_tm_inlier_count / total_candidates
        all_clique_ir = all_pp_ir * all_tm_ir

        # ------------------ 统计Top1指标（4-6） ------------------
        top1_clique = cliques_tensor[top1_idx]
        top1_pp_indices, top1_tm_index = split_clique_to_pp_tm(top1_clique)
        
        # 指标4：Top1枢轴边是否全内点
        top1_is_pp = judge_clique_pp_inlier(top1_pp_indices, kpts_src, kpts_dst, trans_gt, inlier_thresh, dataset)
        # 指标5：Top1第三匹配对是否为内点
        top1_is_tm = judge_clique_tm_inlier(top1_tm_index, kpts_src, kpts_dst, trans_gt, inlier_thresh, dataset) if top1_is_pp else False
        # 指标6：Top1团是否全内点
        top1_is_clique = top1_is_pp and top1_is_tm

        return {
            "All-PP-IR": all_pp_ir,
            "All-TM-IR": all_tm_ir,
            "All-Clique-IR": all_clique_ir,
            "Top1-PP": top1_is_pp,
            "Top1-TM": top1_is_tm,
            "Top1-Clique": top1_is_clique
        }
def main():
    args = tyro.cli(Args)
    # 新增：初始化统计
    metric_calculator = None
    results = []
    if args.calc_metrics:
        metric_calculator = MetricStats()

    if args.dataname.lower() == "3dmatch":
        processed_dataname = "3DMatch"
    elif args.dataname.lower() == "3dlomatch":
        processed_dataname = "3DLoMatch"
    else:
        raise ValueError(f"Invalid dataname: {args.dataname}. Expected '3DMatch' or '3DLoMatch'.")

    # TurboReg
    reger = turboreg_gpu.TurboRegGPU(
        args.max_N,
        args.tau_length_consis,
        args.num_pivot,
        args.radiu_nms,
        args.tau_inlier,
        args.metric_str
    )

    ds = TDMatchFCGFAndFPFHDataset(base_dir=args.dir_dataset, dataset_type=processed_dataname, descriptor_type=args.desc)

    num_succ = 0
    total_time=0

    for i in range(len(ds)):
        data = ds[i]
        kpts_src, kpts_dst, trans_gt = data['kpts_src'], data['kpts_dst'], data['trans_gt']
        # 新增：从样本中提取内点阈值
        inlier_thresh = data['inlier_thresh']  
        # Move keypoints to CUDA device
        kpts_src, kpts_dst = numpy_to_torch32(
            torch.device('cuda:0'), kpts_src, kpts_dst
        )

        # Run TurboReg
        t1 = time.time()
        trans_pred_torch = reger.run_reg(kpts_src, kpts_dst)
        T_reg = (time.time() - t1) * 1000
        total_time += T_reg
        trans_pred = trans_pred_torch.cpu().numpy()
        rre, rte = compute_transformation_error(trans_gt, trans_pred)
        is_succ = (rre < 15) & (rte < 0.3)
        num_succ += is_succ
        
        current_metrics = None
        if args.calc_metrics:
            # 获取当前样本的所有候选和Top1索引
            cliques_tensor = reger.get_cliques_tensor()
            top1_idx = reger.get_idx_best_guess().cpu().numpy().astype(int)

            # 计算当前样本的6个指标
            current_metrics = metric_calculator.compute_single_sample_metrics(
                cliques_tensor, kpts_src, kpts_dst, trans_gt, inlier_thresh, ds, top1_idx
            )

            # 整理数据到字典
            sample_result = {
                "sample_id": i + 1,
                "is_succ": is_succ,
                "RRE": rre,
                "RTE": rte,
                "reg_time_ms": T_reg,
                "All-PP-IR": current_metrics["All-PP-IR"],
                "All-TM-IR": current_metrics["All-TM-IR"],
                "All-Clique-IR": current_metrics["All-Clique-IR"],
                "Top1-PP": current_metrics["Top1-PP"],
                "Top1-TM": current_metrics["Top1-TM"],
                "Top1-Clique": current_metrics["Top1-Clique"]
            }
            results.append(sample_result)

        # 终端输出（根据开关动态调整）
        print(f"Processed item {i+1}/{len(ds)}:")
        print(f"  Registration time: {T_reg:.3f} ms, RR= {(num_succ/(i+1))*100:.3f}%")
        
        if args.calc_metrics and current_metrics is not None:
            print(f"  Current 6 Metrics:")
            print(f"    All-PP-IR: {current_metrics['All-PP-IR']*100:.2f}%")
            print(f"    All-TM-IR: {current_metrics['All-TM-IR']*100:.2f}%")
            print(f"    All-Clique-IR: {current_metrics['All-Clique-IR']*100:.2f}%")
            print(f"    Top1-PP: {'Yes' if current_metrics['Top1-PP'] else 'No'}")
            print(f"    Top1-TM: {'Yes' if current_metrics['Top1-TM'] else 'No'}")
            print(f"    Top1-Clique: {'Yes' if current_metrics['Top1-Clique'] else 'No'}")
        print("-"*80)


    if args.calc_metrics:
        df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_data_path = f"turboreg_results_{processed_dataname}_{args.desc}_{timestamp}.csv"
        df.to_csv(raw_data_path, index=False)
        print(f"\n✅ 原始数据已保存至: {raw_data_path}")

    # 原有最终汇总（始终输出）
    print(f"\n===== Final Summary =====")
    print(f"Total Samples: {len(ds)}, Success Rate (RR): {(num_succ/len(ds))*100:.2f}%")
    print(f"Average Registration Time: {total_time/len(ds):.3f} ms")

if __name__ == "__main__":
    main()

"""
python -m demo_py.exp_3dmatch --desc fpfh --dataname 3DMatch --dir_dataset "DIR_3DMATCH_FPFH_FCGF" --max_N 7000 --tau_length_consis 0.012 --num_pivot 2000 --radiu_nms 0.15 --tau_inlier 0.1 --metric_str "IN"
python -m demo_py.exp_3dmatch --desc fcgf --dataname 3DMatch --dir_dataset "DIR_3DMATCH_FPFH_FCGF" --max_N 6000 --tau_length_consis 0.012 --num_pivot 2000 --radiu_nms 0.10 --tau_inlier 0.1 --metric_str "MAE"
"""