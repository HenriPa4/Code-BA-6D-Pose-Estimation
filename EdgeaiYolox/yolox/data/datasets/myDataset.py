#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
# Adapted for custom COCO-style 6D pose annotations.

import os
import cv2
import numpy as np
import json
from pycocotools.coco import COCO
from loguru import logger
import open3d as o3d

from yolox.data.datasets.datasets_wrapper import Dataset 

def get_object_3d_corners(dim_x, dim_y, dim_z):
    """
    Calculates the 8 corner points of a 3D bounding box centered at origin.
    Args:
        dim_x, dim_y, dim_z: Dimensions of the object.
    Returns:
        np.array: (8, 3) array of corner coordinates.
    """
    w, h, d = dim_x / 2.0, dim_y / 2.0, dim_z / 2.0
    corners = np.array([
        [-w, -h, -d], [w, -h, -d], [w, h, -d], [-w, h, -d],
        [-w, -h,  d], [w, -h,  d], [w, h,  d], [-w, h,  d]
    ], dtype=np.float32)
    return corners

class MyDataset(Dataset):
    """
    Custom Dataset for 6D object pose estimation.
    """

    def __init__(
        self,
        data_dir,
        json_file,
        name,
        img_size=(480, 640),
        preproc=None,
        cache=False,
        object_pose=True,
        obj_dims=(0.118, 0.181, 0.081),
        obj_name="my_tracked_object",
        is_symmetric=False
    ):
        super().__init__(img_size)
        self.data_dir = data_dir
        self.json_file = json_file
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self.object_pose = object_pose
        self.obj_dims = obj_dims
        self.obj_name = obj_name
        self.is_symmetric = is_symmetric
        self.imgs = None

        self.coco_json_path = os.path.join(self.data_dir, self.json_file)
        logger.info(f"Loading COCO annotations from: {self.coco_json_path}")
        self.coco = COCO(self.coco_json_path)
        self.ids = self.coco.getImgIds()
        
        self.class_ids = sorted(self.coco.getCatIds())
        if not self.class_ids:
            raise ValueError("Annotation file has no categories!")
        
        # This is the key part of the fix. We create a mapping.
        # The model will use the 0-based index. The evaluator will use the original COCO ID.
        self.cat_ids_to_cls_idx = {cat_id: i for i, cat_id in enumerate(self.class_ids)}
        
        cats = self.coco.loadCats(self.class_ids)
        
        # The evaluator will look up these dicts using the original COCO category ID (e.g., 1)
        self.class_to_name = {cat_id: cat['name'] for cat_id, cat in zip(self.class_ids, cats)}
        
        calibration_file_path = "calibration_d405.npz"
        try:
            calib_data = np.load(calibration_file_path)
            self.camera_matrix = calib_data['camera_matrix'].flatten()
        except Exception as e:
            logger.error(f"Failed to load camera calibration: {e}")
            raise

        try:
            model_path = os.path.join(self.data_dir, "models/my_object_model.ply")
            mesh = o3d.io.read_triangle_mesh(model_path)
            center = mesh.get_center()
            mesh.translate(-center)
            
            full_model_points = np.asarray(mesh.vertices, dtype=np.float32)
            pcd = mesh.sample_points_uniformly(number_of_points=500)
            sparse_model_points = np.asarray(pcd.points, dtype=np.float32)
            
            # Use the original COCO ID as the key for these dictionaries
            self.class_to_model = {self.class_ids[0]: full_model_points}
            self.class_to_sparse_model = {self.class_ids[0]: sparse_model_points}
        except Exception as e:
            logger.error(f"FATAL: Failed to load/process 3D model: {e}")
            raise
        
        _corners = get_object_3d_corners(self.obj_dims[0], self.obj_dims[1], self.obj_dims[2])
        self.models_corners = {self.class_ids[0]: _corners}
        
        _diameter = np.linalg.norm(np.array(self.obj_dims))
        self.models_diameter = {self.class_ids[0]: _diameter}

        if self.is_symmetric:
            self.symmetric_objects = {self.class_ids[0]: self.obj_name}
        else:
            self.symmetric_objects = {}

        self.annotations = self._load_coco_annotations()
        self.cad_models = self # For evaluator compatibility
        
        if cache:
            self._cache_images()

    def __len__(self):
        return len(self.ids)

    def __del__(self):
        if hasattr(self, 'imgs') and self.imgs is not None:
            del self.imgs

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_id) for _id in self.ids]

    def load_anno_from_ids(self, img_id):
        im_ann = self.coco.loadImgs(img_id)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        
        anno_ids = self.coco.getAnnIds(imgIds=[int(img_id)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)
        res = np.zeros((num_objs, 14 if self.object_pose else 5), dtype=np.float32)

        for ix, obj in enumerate(objs):
            # Convert COCO category_id (e.g., 1) to model's 0-based index
            cls_idx = self.cat_ids_to_cls_idx.get(obj["category_id"])
            if cls_idx is None:
                continue 

            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls_idx # This will be 0

            if self.object_pose:
                pose_info = obj.get("pose_6d_rvec_tvec_cam_object")
                if not pose_info or "rvec" not in pose_info or "tvec" not in pose_info:
                    res[ix, 5:14] = np.nan
                    continue

                rvec = np.array(pose_info["rvec"], dtype=np.float32)
                R_mat, _ = cv2.Rodrigues(rvec)
                res[ix, 5:11] = R_mat[:, :2].T.flatten()
                
                keypoints = obj.get("keypoints", [])
                if len(keypoints) >= 2:
                    res[ix, 11:13] = keypoints[:2]
                else:
                    res[ix, 11:13] = np.nan
                
                res[ix, 13] = pose_info["tvec"][2] # Keep depth in meters
        
        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r
        if self.object_pose:
            valid_pose_mask = ~np.isnan(res[:, 11])
            res[valid_pose_mask, 11:13] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))
        file_name = im_ann["file_name"]
        return (res, img_info, resized_info, file_name)

    def load_anno(self, index):
        # Loads the annotation array 'res' for the given index
        return self.annotations[index][0]

    def load_resized_img(self, index):
        # Loads and resizes an image to fit within self.img_size, maintaining aspect ratio
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        # file_name is the 4th element from load_anno_from_ids
        file_name = self.annotations[index][3]
        
        # self.name is the image subfolder (e.g., "train_images" or "images/train")
        # self.data_dir is the root dataset directory (e.g., "datasets/MyObjectDataset")
        # Path: data_dir / name / file_name.jpg
        img_file_path = os.path.join(self.data_dir, self.name, file_name)
        
        img = cv2.imread(img_file_path)
        if img is None:
            logger.critical(f"Failed to load image at {img_file_path}. Please check path and file integrity.")
            # Depending on strictness, either raise an error or return a placeholder
            raise FileNotFoundError(f"Image not found: {img_file_path}")
        return img

    def pull_item(self, index):
        # Loads image and its annotations for the given index
        id_ = self.ids[index]
        res, img_info, resized_info, _ = self.annotations[index]

        if self.imgs is not None: # Using cached images
            pad_img = self.imgs[index]
            img = pad_img[: resized_info[0], : resized_info[1], :].copy()
        else: # Load image from disk
            img = self.load_resized_img(index)
            
        return img, res.copy(), img_info, id_

    @Dataset.mosaic_getitem # Important decorator for YOLOX training
    def __getitem__(self, index):
        """
        Retrieves an image and its annotations, applies preprocessing.
        Required method for PyTorch DataLoader.
        """
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim) # self.input_dim usually == self.img_size
        
        return img, target, img_info, img_id


    def _cache_images(self):
        logger.warning(
            "\n********************************************************************************\n"
            "You are using cached images in RAM to accelerate training.\n"
            "This requires large system RAM. Make sure you have enough.\n"
            "********************************************************************************\n"
        )
        max_h, max_w = self.img_size[0], self.img_size[1]
        
        # Use a unique name for the cache file, incorporating dataset name/split
        # self.name could be "images/train", replace '/' with '_' for filename
        safe_name_for_file = self.name.replace(os.sep, '_')
        cache_file = os.path.join(self.data_dir, f"img_resized_cache_{safe_name_for_file}.array")
        
        if not os.path.exists(cache_file):
            logger.info(f"Caching images for the first time. This might take a while. Cache file: {cache_file}")
            self.imgs = np.memmap(
                cache_file,
                shape=(len(self.ids), max_h, max_w, 3), # Store as (H,W,C)
                dtype=np.uint8,
                mode="w+",
            )
            from tqdm import tqdm
            from multiprocessing.pool import ThreadPool

            NUM_THREADS = min(8, os.cpu_count())
            with ThreadPool(NUM_THREADS) as pool:
                loaded_images = pool.imap_unordered(lambda i: (i, self.load_resized_img(i)), range(len(self.annotations)))
                pbar = tqdm(loaded_images, total=len(self.annotations), desc="Caching images")
                for k, resized_img in pbar:
                    h, w, _ = resized_img.shape
                    self.imgs[k, :h, :w, :] = resized_img # Fill the memmap array
            self.imgs.flush()
            pbar.close()
        else:
            logger.info(f"Loading cached imgs from {cache_file}. Make sure dataset hasn't changed!")

        logger.info("Loading cached imgs into memory map...")
        self.imgs = np.memmap(
            cache_file,
            shape=(len(self.ids), max_h, max_w, 3),
            dtype=np.uint8,
            mode="r+", # Read-only if already created, 'r+' if might need to write (though typically not after creation)
        )