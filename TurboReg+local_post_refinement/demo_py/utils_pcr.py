import torch
import numpy as np

def numpy_to_torch32(device, *arrays):
    return [torch.tensor(array, device=device, dtype=torch.float32) for array in arrays]

def compute_transformation_error(trans_gt: np.ndarray, trans_pred: np.ndarray):
    """
    Compute the transformation error between the predicted and ground truth 4x4 rigid transformation matrices.

    This function calculates both the Relative Rotation Error (RRE) and Relative Translation Error (RTE).
    It extracts the rotation and translation components from the 4x4 matrices and computes the respective errors.

    Args:
        trans_gt (np.ndarray): Ground truth 4x4 transformation matrix
        trans_pred (np.ndarray): Predicted 4x4 transformation matrix

    Returns:
        rre (float): Relative Rotation Error in degrees
        rte (float): Relative Translation Error
    """
    
    # Extract rotation and translation components
    gt_rotation = trans_gt[:3, :3]
    est_rotation = trans_pred[:3, :3]
    gt_translation = trans_gt[:3, 3]
    est_translation = trans_pred[:3, 3]
    
    # Compute Relative Rotation Error (RRE)
    rre = compute_relative_rotation_error(gt_rotation, est_rotation)
    
    # Compute Relative Translation Error (RTE)
    rte = compute_relative_translation_error(gt_translation, est_translation)
    
    return rre, rte

def compute_relative_rotation_error(gt_rotation: np.ndarray, est_rotation: np.ndarray):
    r"""Compute the isotropic Relative Rotation Error.

    RRE = acos((trace(R^T \cdot \bar{R}) - 1) / 2)

    Args:
        gt_rotation (array): ground truth rotation matrix (3, 3)
        est_rotation (array): estimated rotation matrix (3, 3)

    Returns:
        rre (float): relative rotation error.
    """
    x = 0.5 * (np.trace(np.matmul(est_rotation.T, gt_rotation)) - 1.0)
    x = np.clip(x, -1.0, 1.0)
    x = np.arccos(x)
    rre = 180.0 * x / np.pi
    return rre


def compute_relative_translation_error(gt_translation: np.ndarray, est_translation: np.ndarray):
    r"""Compute the isotropic Relative Translation Error.

    RTE = \lVert t - \bar{t} \rVert_2

    Args:
        gt_translation (array): ground truth translation vector (3,)
        est_translation (array): estimated translation vector (3,)

    Returns:
        rte (float): relative translation error.
    """
    return np.linalg.norm(gt_translation - est_translation)

# 新增：指标计算相关工具函数
def get_clique_pp_tm_indices(clique_tensor, all_pivots):
    """
    从clique索引中拆分枢轴边(PP)和第三匹配对(TM)的索引
    :param clique_tensor: 单个clique的索引 (3,)，格式[pp0, pp1, tm]
    :param all_pivots: 所有枢轴边索引 (num_pivot, 2)
    :return: pp_indices (2,), tm_index (int)
    """
    pp_indices = clique_tensor[:2].cpu().numpy().astype(int)
    tm_index = clique_tensor[2].cpu().numpy().astype(int)
    return pp_indices, tm_index

def judge_clique_pp_inlier(pp_indices, kpts_src, kpts_dst, trans_gt, inlier_thresh, dataset):
    """
    判定clique的枢轴边是否全为内点
    :param pp_indices: 枢轴边匹配对索引 (2,)
    :param kpts_src: 源关键点 (N,3)
    :param kpts_dst: 目标关键点 (N,3)
    :param trans_gt: 真值位姿 (4,4)
    :param inlier_thresh: 内点阈值
    :param dataset: 数据集实例（提供is_all_matches_inlier函数）
    :return: bool, 枢轴边是否全为内点
    """
    pp_src = kpts_src[pp_indices]
    pp_dst = kpts_dst[pp_indices]
    return dataset.is_all_matches_inlier(pp_src, pp_dst, trans_gt, inlier_thresh)

def judge_clique_tm_inlier(tm_index, kpts_src, kpts_dst, trans_gt, inlier_thresh, dataset):
    """
    判定clique的第三匹配对是否为内点
    :param tm_index: 第三匹配对索引 (int)
    :param kpts_src: 源关键点 (N,3)
    :param kpts_dst: 目标关键点 (N,3)
    :param trans_gt: 真值位姿 (4,4)
    :param inlier_thresh: 内点阈值
    :param dataset: 数据集实例（提供is_single_match_inlier函数）
    :return: bool, 第三匹配对是否为内点
    """
    tm_src = kpts_src[tm_index]
    tm_dst = kpts_dst[tm_index]
    return dataset.is_single_match_inlier(tm_src, tm_dst, trans_gt, inlier_thresh)

def split_clique_to_pp_tm(clique_tensor):
    """
    把单个Clique张量拆分为枢轴边(PP)索引和第三匹配对(TM)索引
    输入：clique_tensor - 形状为 (3,) 的PyTorch张量
    输出：(pp_indices, tm_index) - pp_indices是(2,)数组，tm_index是标量
    """
    pp_indices = clique_tensor[:2].cpu().numpy().astype(int)
    tm_index = clique_tensor[2].cpu().numpy().astype(int)
    return pp_indices, tm_index