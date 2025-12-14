#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import cv2
import json
import numpy as np
import open3d as o3d
import torch
from pathlib import Path
from pycocotools.coco import COCO

from yolox.exp import Exp as MyExp
from yolox.data.datasets.datasets_wrapper import Dataset
from yolox.evaluators import ObjectPoseEvaluator

# ==============================================================================
# ===== CUSTOM DATALOADER AND EVALUATOR (Definitive Version) ===================
# ==============================================================================

def safe_decode_rotation_translation(pose, camera_matrix):
    """
    This is a local, corrected, and self-contained version of the utility function.
    It does not depend on any faulty library imports.
    """
    if torch.is_tensor(pose):
        pose = pose.cpu().numpy().copy()
    
    # --- THIS IS THE FIX: Replicate the 6D -> rotation matrix logic locally ---
    # The 6D rotation is the first two columns of the rotation matrix
    r1 = pose[5:8].reshape(3, 1)
    r2 = pose[8:11].reshape(3, 1)
    
    # Compute the third column of the rotation matrix using cross product
    r3 = np.cross(r1.T, r2.T).T
    
    # Combine the columns to get the full 3x3 rotation matrix
    rotation_mat = np.concatenate((r1, r2, r3), axis=1)
    rotation_vec, _ = cv2.Rodrigues(rotation_mat)
    # --- END OF FIX ---

    # Projected 2D center is at indices 11-12
    px, py = pose[11], pose[12]
    
    # Depth is the LAST element of the model's output array.
    tz = pose[-1] * 100.0  # Use index -1 to be robust. Assumes cm->mm conversion.

    # Recalculate Tx and Ty from the projected center and depth
    fx, fy = camera_matrix[0,0], camera_matrix[1,1]
    cx, cy = camera_matrix[0,2], camera_matrix[1,2]
    tx = (px - cx) * tz / fx
    ty = (py - cy) * tz / fy
    
    translation_vec = np.array([tx, ty, tz])
    return rotation_vec, translation_vec

class CustomCADModels:
    # This class is correct.
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        models_info_path = self.data_dir / "models/models_info.json"
        models_dir = self.data_dir / "models"
        with open(models_info_path) as f: self.models_info = json.load(f)
        self.class_to_model_path = {int(k): models_dir / f"obj_{int(k):06d}.ply" for k in self.models_info.keys()}
        self.class_to_model = {cid: o3d.io.read_point_cloud(str(p)) for cid, p in self.class_to_model_path.items() if p.exists()}
        camera_path = self.data_dir / "camera.json"
        with open(camera_path) as f: p = json.load(f)
        cam_matrix = np.array([p['fx'], 0, p['cx'], 0.0, p['fy'], p['cy'], 0.0, 0.0, 1.0]).reshape(3, 3)
        self.camera_matrix = {"camera_uw": cam_matrix, "camera": cam_matrix}
        self.models_corners, self.models_diameter = self.get_models_params()

    def get_models_params(self):
        models_corners_3d, models_diameter = {}, {}
        for model_id_str, model_param in self.models_info.items():
            model_id = int(model_id_str)
            min_x, size_x = model_param['min_x'], model_param['size_x']
            min_y, size_y = model_param['min_y'], model_param['size_y']
            min_z, size_z = model_param['min_z'], model_param['size_z']
            corners_3d = np.array([[min_x, min_y, min_z], [min_x, min_y, min_z + size_z], [min_x, min_y + size_y, min_z + size_z], [min_x, min_y + size_y, min_z], [min_x + size_x, min_y, min_z], [min_x + size_x, min_y, min_z + size_z], [min_x + size_x, min_y + size_y, min_z + size_z], [min_x + size_x, min_y + size_y, min_z],])
            models_corners_3d[model_id] = corners_3d
            models_diameter[model_id] = model_param.get('diameter', 0.0)
        return models_corners_3d, models_diameter

class CustomPoseDataset(Dataset):
    # This data loader is correct.
    def __init__(self, data_dir, json_file, name, img_size, preproc=None, **kwargs):
        super().__init__(img_size)
        self.data_dir = data_dir
        self.json_file = json_file
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self.object_pose = True
        self.json_path = os.path.join(self.data_dir, "annotations", self.json_file)
        self.coco = COCO(self.json_path)
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        self.cad_models = CustomCADModels(data_dir=data_dir)
        self.class_to_model = self.cad_models.class_to_model
        self.models_corners = self.cad_models.models_corners
        self.class_to_name = {cat['id']: cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())}
        self.symmetric_objects = {}
        self.annotations = self._load_coco_annotations()

    def __len__(self): return len(self.ids)
    def _load_coco_annotations(self): return [self.load_anno_from_ids(_id) for _id in self.ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width, height = im_ann["width"], im_ann["height"]
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False))
        res = np.zeros((len(annotations), 14))
        for ix, obj in enumerate(annotations):
            res[ix, 0:4] = obj["bbox"]
            res[ix, 4] = self.class_ids.index(obj["category_id"])
            pose_matrix = np.array(obj["pose"])
            R, tvec = pose_matrix[:3, :3], pose_matrix[:3, 3]
            res[ix, 5:11] = R.T[:,:2].flatten()
            cam_matrix = self.cad_models.camera_matrix['camera_uw']
            obj_center_2d = cam_matrix @ tvec
            if obj_center_2d[2] != 0: res[ix, 11:13] = (obj_center_2d / obj_center_2d[2])[:2]
            res[ix, 13] = tvec[2] 
        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r
        res[:, 11:13] *= r
        return (res, (height, width), (int(height * r), int(width * r)), im_ann["file_name"])

    def load_image(self, index):
        file_name = self.annotations[index][3]
        image_subfolder = "images/train" if self.name == "train" else "images/validation"
        img_file = os.path.join(self.data_dir, image_subfolder, file_name)
        img = cv2.imread(img_file)
        assert img is not None, f"Image not found at {img_file}"
        return img
    
    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        return cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR).astype(np.uint8)

    def pull_item(self, index):
        id_ = self.ids[index]
        res, img_info, resized_info, _ = self.annotations[index]
        img = self.load_resized_img(index)
        return img, res.copy(), img_info, np.array([id_])
    
    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)
        if self.preproc is not None: img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_id

class CustomObjectPoseEvaluator(ObjectPoseEvaluator):
    """ Overrides the original evaluator to call our safe, local utility function. """
    def convert_to_coco_format(self, outputs, targets, info_imgs, ids, camera_matrix):
        frame_data_list, frame_pred_data_list = [], []
        for (output, target, info_img, img_id) in zip(outputs, targets, info_imgs, ids):
            frame_data, pred_data = {'image_id': int(img_id)}, {'image_id': int(img_id)}
            if target is not None:
                gt_rotations, gt_translations = [], []
                for label in target:
                    rotation, translation = safe_decode_rotation_translation(label, camera_matrix)
                    gt_rotations.append(rotation)
                    gt_translations.append(translation)
                frame_data.update({'gt_rotations': gt_rotations, 'gt_translations': gt_translations, 'gt_labels': target[:, 4]})
            if output is not None:
                pred_rotations, pred_translations = [], []
                # Ensure output has detections before proceeding
                if output.shape[0] > 0:
                    for label in np.unique(output[:, 6]):
                        # Filter outputs for the current label
                        label_outputs = output[output[:, 6] == label]
                        if label_outputs.shape[0] > 0:
                            rotation, translation = safe_decode_rotation_translation(label_outputs[0], camera_matrix)
                            pred_rotations.append(rotation)
                            pred_translations.append(translation)
                pred_data.update({'pred_bboxes': output[:, :4], 'pred_scores': output[:, 4] * output[:, 5], 'pred_rotations': pred_rotations, 'pred_translations': pred_translations, 'pred_labels': output[:, 6]})
            frame_data_list.append(frame_data)
            frame_pred_data_list.append(pred_data)
        return frame_data_list, frame_pred_data_list

# ==============================================================================
# ===== EXPERIMENT CONFIGURATION (Final) =======================================
# ==============================================================================

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # ... your settings ...
        self.num_classes = 1
        self.depth = 0.67
        self.width = 0.75
        self.data_dir = "datasets/MyDataset"
        self.train_ann = "images_annotations.json"
        self.val_ann = "validation_annotations.json"
        self.input_size = (480, 640)
        self.test_size = (480, 640)
        self.data_num_workers = 4
        self.object_pose = True
        self.data_set = "ycbv"

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import TrainTransform, YoloBatchSampler, DataLoader, InfiniteSampler, MosaicDetection, worker_init_reset_seed
        dataset = CustomPoseDataset(data_dir=self.data_dir, json_file=self.train_ann, name="train", img_size=self.input_size, preproc=TrainTransform(max_labels=50, object_pose=self.object_pose))
        dataset = MosaicDetection(dataset, mosaic=not no_aug, img_size=self.input_size, preproc=TrainTransform(max_labels=120, object_pose=self.object_pose))
        sampler = InfiniteSampler(len(dataset), seed=self.seed if self.seed else 0)
        batch_sampler = YoloBatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False, mosaic=not no_aug)
        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True, "batch_sampler": batch_sampler, "worker_init_fn": worker_init_reset_seed}
        return DataLoader(dataset, **dataloader_kwargs)

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import ValTransform
        valdataset = CustomPoseDataset(data_dir=self.data_dir, json_file=self.val_ann, name="test", img_size=self.test_size, preproc=ValTransform(legacy=legacy))
        sampler = torch.utils.data.SequentialSampler(valdataset)
        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True, "sampler": sampler, "batch_size": batch_size}
        return torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = CustomObjectPoseEvaluator(
            dataloader=val_loader, img_size=self.test_size, confthre=self.test_conf,
            nmsthre=self.nmsthre, num_classes=self.num_classes,
        )
        return evaluator