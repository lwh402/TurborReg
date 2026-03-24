import os
import numpy as np
import open3d as o3d
import torch

class TDMatchFCGFAndFPFHDataset:
    def __init__(self, base_dir, dataset_type="3DMatch", descriptor_type="fcgf"):
        """
        Initialize the MatchDataset.

        :param base_dir: Base directory containing the datasets.
        :param dataset_type: Type of the dataset ("3DMatch" or "3DLoMatch").
        :param descriptor_type: Descriptor type ("fcgf" or "fpfh").
        """
        self.base_dir = base_dir
        self.dataset_type = dataset_type
        self.descriptor_type = descriptor_type
        self.dataset_path = os.path.join(
            base_dir, dataset_type, f"all_{descriptor_type}")
        
        # 新增：数据集预设内点阈值（3DMatch:0.1m, 3DLoMatch:0.3m）
        # 具体阈值是多少后面再查
        self.inlier_threshold = 0.1 if dataset_type == "3DMatch" else 0.3

        if not os.path.exists(self.dataset_path):
            raise ValueError(
                f"Dataset path {self.dataset_path} does not exist.")

        self.matching_pairs = self._load_matching_pairs()

    def _load_matching_pairs(self):
        """
        Load all matching pairs from the dataset.

        :return: List of matching pair information.
        """
        matching_pairs = []

        # Traverse all scenes in the dataset
        # for scene_dir in tqdm(os.listdir(self.dataset_path), desc="Loading scenes"):
        for scene_dir in os.listdir(self.dataset_path):
            scene_path = os.path.join(self.dataset_path, scene_dir)
            if not os.path.isdir(scene_path):
                continue

            # Collect all matching files in the scene
            target_gt_name = "@corr_fcgf.txt" if self.descriptor_type == "fcgf" else "@corr.txt"
            for file_name in os.listdir(scene_path):
                if file_name.endswith(target_gt_name):
                    base_name = file_name.split("@")[0]
                    if self.descriptor_type == "fpfh":
                        corr_file = os.path.join(scene_path, file_name)
                        gtmat_file = os.path.join(
                            scene_path, f"{base_name}@GTmat.txt")
                    elif self.descriptor_type == "fcgf":
                        corr_file = os.path.join(scene_path, file_name)
                        gtmat_file = os.path.join(
                            scene_path, f"{base_name}@GTmat_{self.descriptor_type}.txt")
                    src_cloud_file = os.path.join(
                        scene_path, f"{base_name.split('+')[0]}.ply")
                    dst_cloud_file = os.path.join(
                        scene_path, f"{base_name.split('+')[1]}.ply")

                    if os.path.exists(gtmat_file) and os.path.exists(src_cloud_file) and os.path.exists(dst_cloud_file):
                        matching_pairs.append({
                            "corr_file": corr_file,
                            "gtmat_file": gtmat_file,
                            "src_cloud_file": src_cloud_file,
                            "dst_cloud_file": dst_cloud_file,
                            # 新增：记录该样本的内点阈值
                            # 为啥要在每个匹配对中记录内点阈值
                            "inlier_thresh": self.inlier_threshold
                        })

        return matching_pairs

    def __len__(self):
        """
        Return the number of matching pairs in the dataset.

        :return: Number of matching pairs.
        """
        return len(self.matching_pairs)

    def __getitem__(self, idx):
        """
        Get a matching pair by index.

        :param idx: Index of the matching pair.
        :return: Dictionary containing kpts_src, kpts_dst, transformation matrix, and point clouds.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range.")

        pair_info = self.matching_pairs[idx]

        # Load correspondences
        correspondences = np.loadtxt(pair_info["corr_file"], delimiter=' ')
        kpts_src = correspondences[:, :3]
        kpts_dst = correspondences[:, 3:]

        # Load ground truth transformation matrix
        trans_gt = np.loadtxt(pair_info["gtmat_file"], delimiter=' ')

        # Load point clouds
        src_cloud = np.asarray(o3d.io.read_point_cloud(
            pair_info["src_cloud_file"]).points)
        dst_cloud = np.asarray(o3d.io.read_point_cloud(
            pair_info["dst_cloud_file"]).points)

        return {
            "kpts_src": kpts_src,
            "kpts_dst": kpts_dst,
            "trans_gt": trans_gt,
            "pts_src": src_cloud,
            "pts_dst": dst_cloud,
             # 新增：返回内点阈值
            "inlier_thresh": pair_info["inlier_thresh"] 
        }

    # 新增：判定单个匹配对是否为内点
    def is_single_match_inlier(self, src_pt, dst_pt, trans_gt, inlier_thresh):
        # ✅ 新增：如果是PyTorch张量，先转回CPU NumPy
        if isinstance(src_pt, torch.Tensor):
            src_pt = src_pt.cpu().numpy()
        if isinstance(dst_pt, torch.Tensor):
            dst_pt = dst_pt.cpu().numpy()
        
        # ✅ 简化：不用转齐次坐标了，直接用NumPy算
        src_pt_transformed = trans_gt[:3, :3] @ src_pt + trans_gt[:3, 3]
        dist = np.linalg.norm(src_pt_transformed - dst_pt)
        return dist <= inlier_thresh


    def is_all_matches_inlier(self, src_pts, dst_pts, trans_gt, inlier_thresh):
        """
        判定一组匹配对是否全为内点
        兼容：NumPy数组 和 PyTorch CUDA张量 两种输入
        """
        # ✅ 新增：如果是PyTorch张量，先转回CPU NumPy
        if isinstance(src_pts, torch.Tensor):
            src_pts = src_pts.cpu().numpy()
        if isinstance(dst_pts, torch.Tensor):
            dst_pts = dst_pts.cpu().numpy()
        
        for src_pt, dst_pt in zip(src_pts, dst_pts):
            if not self.is_single_match_inlier(src_pt, dst_pt, trans_gt, inlier_thresh):
                return False
        return True
