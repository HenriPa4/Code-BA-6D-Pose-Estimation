#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import torch

from yolox.exp import Exp as MyExp
import torch.distributed as dist
import torch.nn as nn

import numpy as np

import yolox.utils.logger as LOGGER

from yolox.data.datasets.myDataset import MyDataset


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.67
        self.width = 0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # ---------------- model config ---------------- #
        self.num_classes = 1

        # ---------------- dataloader config ---------------- #
        self.input_size = (480, 640)  # (height, width)
        self.data_dir = "datasets/MyDataset"
        self.train_ann = "annotations/images_annotations.json"
        self.val_ann = "annotations/validation_annotations.json"
        # --------------- transform config ----------------- #
        self.mosaic_prob = 0.0
        self.mixup_prob = 0.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.0
        self.degrees = 10.0
        self.translate = 0.1
        self.mosaic_scale = (0.9, 1.1)
        self.mixup_scale = (1.0, 1.0)
        self.shear = 0.0
        self.perspective = 0.0
        self.enable_mixup = False
        self.shape_loss = False
        # --------------  training config --------------------- #
        self.trn_z_weight = 0.0001
        self.adds_loss_weight = 0.0001
        self.basic_lr_per_img = 1.5625e-06
        self.max_epoch = 300
        self.eval_interval = 10
        # -----------------  testing config ------------------ #
        self.test_size = (480, 640)
        self.test_conf = 0.01
        self.nmsthre = 0.001
        self.data_set = "myDataset" #"ycbv" "lmo"
        self.object_pose  = True
        self.visualize = True
        self.od_weights = "yolox_m.pth"
        try:
            LOGGER.info("Loading camera matrix into experiment file...")
            calib_data = np.load("calibration_d405.npz")
            self.camera_matrix = calib_data['camera_matrix'].flatten()
            LOGGER.success("Camera matrix loaded successfully.")
        except Exception as e:
            raise IOError(f"Failed to load camera matrix from calibration_d405.npz: {e}")

    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXObjectPoseHead
        
        from yolox.data.datasets.myDataset import MyDataset

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels)
            
            temp_dataset = MyDataset(
                data_dir=self.data_dir,
                json_file=self.train_ann,
                name="images/train", 
                img_size=self.input_size,
                preproc=None,
                object_pose=True
            )

            # Pass the dataset object to the head using the 'cad_models' argument.
            head = YOLOXObjectPoseHead(
                self.num_classes, self.width, in_channels=in_channels, 
                dataset=temp_dataset, # This is the key change
                shape_loss=self.shape_loss
            )
            
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)

        # Load pre-trained 2D detection weights
        if self.od_weights is not None:
            LOGGER.info(f"Loading 2D object detection weights from {self.od_weights}")
            # Load pre-trained weights to CPU first
            try:
                # state_dict_object_detect = torch.load(self.od_weights, map_location="cpu")
                # More robustly:
                ckpt = torch.load(self.od_weights, map_location="cpu")
                if "model" in ckpt: # Common practice to save model state_dict inside a 'model' key
                    state_dict_object_detect = ckpt["model"]
                elif "state_dict" in ckpt: # Another common key
                    state_dict_object_detect = ckpt["state_dict"]
                else:
                    state_dict_object_detect = ckpt

                state_dict_object_pose = self.model.state_dict()
                
                # Filter out mismatched keys and keys for parts you don't want to load (e.g. final classification layer if num_classes differs)
                # and also potentially freeze these loaded weights
                model_dict = self.model.state_dict()
                load_dict = {}
                num_frozen_layers = 0
                for k, v in state_dict_object_detect.items():
                    if k in model_dict and model_dict[k].shape == v.shape:
                        load_dict[k] = v
                        # Optionally freeze these layers
                        # if you want to only train the pose-specific parts initially
                        # state_dict_object_pose[k].requires_grad = False # This is incorrect place
                        # Freezing should be done on model.parameters() after loading.
                    else:
                        LOGGER.warning(
                            f"Skipping loading layer {k} from pretrained weights due to mismatch."
                        )
                
                model_dict.update(load_dict)
                self.model.load_state_dict(model_dict)
                LOGGER.info(f"Successfully loaded {len(load_dict)} layers from {self.od_weights}")

                # If you want to freeze the loaded backbone/neck weights:
                # This part should ideally be more granular and configurable
                # For example, freeze all layers except the 'head'.
                # for name, param in self.model.named_parameters():
                #    if name.startswith("backbone.") or name.startswith("neck."): # Adjust based on YOLOPAFPN structure
                #        if name in load_dict: # Only freeze if it was loaded
                #            param.requires_grad = False
                #            num_frozen_layers +=1
                # LOGGER.info(f"Froze {num_frozen_layers} parameters from the loaded 2D detector.")

            except Exception as e:
                LOGGER.error(f"Error loading pre-trained weights from {self.od_weights}: {e}")
        
        return self.model

    def get_data_loader(
            self, batch_size, is_distributed, no_aug=True, cache_img=False
        ):
            from yolox.data import (
                LMODataset,
                YCBVDataset,
                MyDataset,
                TrainTransform,
                YoloBatchSampler,
                DataLoader,
                InfiniteSampler,
                MosaicDetection,
                worker_init_reset_seed,
            )
            from yolox.utils import (
                wait_for_the_master,
                get_local_rank,
            )

            local_rank = get_local_rank()

            with wait_for_the_master(local_rank):
                if self.data_set == "lm" or self.data_set == "lmo":
                    base_dir = "lm" if self.data_set == "lm" else "lmo"
                    dataset = LMODataset(
                            data_dir=self.data_dir,
                            json_file=self.train_ann,
                            img_size=self.input_size,
                            preproc=TrainTransform(
                                max_labels=50,
                                flip_prob=self.flip_prob,
                                hsv_prob=self.hsv_prob,
                                object_pose=self.object_pose),
                            cache=cache_img,
                            object_pose=self.object_pose,
                            base_dir=base_dir
                        )
                elif self.data_set == "ycbv":
                    dataset = YCBVDataset(
                            data_dir=self.data_dir,
                            json_file=self.train_ann,
                            img_size=self.input_size,
                            preproc=TrainTransform(
                                max_labels=50,
                                flip_prob=self.flip_prob,
                                hsv_prob=self.hsv_prob,
                                object_pose=self.object_pose),
                            cache=cache_img,
                            object_pose=self.object_pose
                        )
                elif self.data_set == "myDataset":
                    dataset = MyDataset(
                            data_dir=self.data_dir,
                            json_file=self.train_ann,       # e.g., "train_annotations.json"
                            name="images/train",               # e.g., your training image folder name
                            img_size=self.input_size,
                            preproc=TrainTransform(
                                max_labels=50, # Max objects per image
                                flip_prob=self.flip_prob,
                                hsv_prob=self.hsv_prob,
                                object_pose=self.object_pose), # Crucial
                            cache=cache_img,
                            object_pose=self.object_pose
                    )
                else:
                    raise ValueError(f"Unsupported dataset: {self.data_set}")

            dataset = MosaicDetection(
                dataset,
                mosaic=not no_aug,
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=120,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob,
                    object_pose=self.object_pose),
                degrees=self.degrees,
                translate=self.translate,
                mosaic_scale=self.mosaic_scale,
                mixup_scale=self.mixup_scale,
                shear=self.shear,
                enable_mixup=self.enable_mixup,
                mosaic_prob=self.mosaic_prob,
                mixup_prob=self.mixup_prob,
            )

            self.dataset = dataset

            if is_distributed:
                batch_size = batch_size // dist.get_world_size()

            sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

            batch_sampler = YoloBatchSampler(
                sampler=sampler,
                batch_size=batch_size,
                drop_last=False,
                mosaic=not no_aug,
            )

            dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
            dataloader_kwargs["batch_sampler"] = batch_sampler

            # Make sure each process has different random seed, especially for 'fork' method.
            # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
            dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

            train_loader = DataLoader(self.dataset, **dataloader_kwargs)

            return train_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False, visualize=False):
            if self.val_ann is None:
                return None

            from yolox.evaluators import ObjectPoseEvaluator

            val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
            output_dir = os.path.join(self.output_dir, self.exp_name)
            evaluator = ObjectPoseEvaluator(
                dataloader=val_loader,
                img_size=self.test_size,
                confthre=self.test_conf,
                nmsthre=self.nmsthre,
                num_classes=self.num_classes,
                testdev=testdev,
                visualize=self.visualize,
                output_dir=output_dir
                )

            return evaluator
    
    # In yolox_m_object_pose.py

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import MyDataset, ValTransform
        from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
        import torch

        def eval_collate_fn(batch):
            imgs, targets, img_infos, ids = zip(*batch)
            
            # Stack images into a single tensor
            imgs = torch.stack([torch.from_numpy(img) for img in imgs], 0)
            
            # Convert each target numpy array to a tensor, but keep them in a list
            targets = [torch.from_numpy(target) for target in targets]
            
            # THE FIX: Unzip the list of (h, w) tuples into two separate lists
            img_heights, img_widths = zip(*img_infos)
            
            return imgs, targets, (list(img_heights), list(img_widths)), ids

        val_dataset = MyDataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name="images/validation",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
            object_pose=self.object_pose
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = DistributedSampler(val_dataset, shuffle=False)
        else:
            sampler = SequentialSampler(val_dataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
            "batch_size": batch_size,
            "collate_fn": eval_collate_fn, # Add this line to use our custom function
        }
        val_loader = DataLoader(val_dataset, **dataloader_kwargs)

        return val_loader