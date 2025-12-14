#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os
import random

import json # <-- Add this import
from pathlib import Path # <-- Add this import
from pycocotools.coco import COCO
from yolox.data.datasets.ycbv import YCBVDataset, CADModelsYCBV
from yolox.data.datasets.custom import CustomDataset
from yolox.data.datasets.datasets_wrapper import Dataset
import cv2
import numpy as np
import open3d as o3d

import torch
import torch.distributed as dist
import torch.nn as nn

from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()

        # ---------------- model config ---------------- #
        self.depth = 0.67
        self.width = 0.75
        self.num_classes = 1
        
        # ---------------- dataloader config ---------------- #
        # Define yourself dataset path
        self.input_size = (800, 1280)  # (height, width)
        self.data_dir = "datasets/MyDataset"
        self.train_ann = "train_annotations.json"
        self.val_ann = "validation_annotations.json"
        self.img_folder_names = None
        self.data_num_workers = 4
        # self.multiscale_range = 5
        # You can uncomment this line to specify a multiscale range
        # self.random_size = (14, 26)

        # --------------- transform config ----------------- #
        #stitching images
        self.mosaic_prob = 0.0
        #train on "blended" together images !!!not helpful for 6dpos!!!
        self.mixup_prob = 0.0
        #color deviations for lighting
        self.hsv_prob = 1.0
        #mirror image !!!!DONT USE FOR ASYMMERTICAL OBJECT!!!
        self.flip_prob = 0.0
        #rotation object can be tiltet
        self.degrees = 10.0
        #moves object so its not always in same position
        self.translate = 0.1
        self.mosaic_scale = (0.9, 1.1)
        self.mixup_scale = (1.0, 1.0)
        #shear image into paralellogramm object form different perspective
        self.shear = 0.0
        #add 3d perspective warp !!!not helpful for 6dpos!!!
        self.perspective = 0.0
        self.enable_mixup = False
        #special case for training rotation of object seperatly
        self.shape_loss = False

        # --------------  training config --------------------- #
        #slow start so model doesnt "expolde"
        self.warmup_epochs = 5
        self.warmup_lr = 0
        self.max_epoch = 200
        self.basic_lr_per_img = 0.001 / 64.0
        self.scheduler = "yoloxwarmcos"
        #number of epochs without noise for training at the end
        self.no_aug_epochs = 15
        #learning decreases towards end to this value
        self.min_lr_ratio = 0.05
        #exponential moving average "shadow copy" for comparrison
        self.ema = True
        #prevent weights from growing too large
        self.weight_decay = 5e-4
        #momentum to overcome learning "bumps"
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 10
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        self.test_size = (800, 1280)
        #confidence in tesing images <1% are thrown out immediatly
        self.test_conf = 0.01
        #removes overlapping bboxes
        self.nmsthre = 0.65
        self.data_set = "custom"
        self.object_pose = True
        self.human_pose = False
        #generate output images
        self.visualize = False

        # -----------------  device_type -----------------
        # None, cuda or cpu. None implies cuda
        self.device_type = None


#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead, YOLOXObjectPoseHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels)
            head = YOLOXObjectPoseHead(self.num_classes, self.width, in_channels=in_channels, dataset=self.data_set)
            
            head.trn_xy_loss_weight = 5.0
            head.trn_xyz_loss_weight = 5.0
            
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def get_data_loader(
        self, batch_size, is_distributed, no_aug=False, cache_img=False
    ):
        from yolox.data import (
            COCODataset,
            LMODataset,
            YCBVDataset,
            COCOKPTSDataset,
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
            if self.data_set == "coco":
                dataset = COCODataset(
                    data_dir=self.data_dir,
                    json_file=self.train_ann,
                    img_size=self.input_size,
                    preproc=TrainTransform(
                        max_labels=50,
                        flip_prob=self.flip_prob,
                        hsv_prob=self.hsv_prob),
                    cache=cache_img,
                )
            elif self.data_set == "lmo" or self.data_set == "lm":
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
               dataset = CustomYCBVDataset(
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
            elif self.data_set == "custom":
               dataset = CustomDataset(
                    data_dir=self.data_dir,
                    json_file=self.train_ann,
                    name = "train",
                    img_size=self.input_size,
                    preproc=TrainTransform(
                        max_labels=50,
                        flip_prob=self.flip_prob,
                        hsv_prob=self.hsv_prob,
                        object_pose=self.object_pose),
                    cache=cache_img,
                    object_pose=self.object_pose
                )
            elif self.data_set == "coco_kpts":
                dataset = COCOKPTSDataset(
                    data_dir=self.data_dir,
                    json_file=self.train_ann,
                    img_size=self.input_size,
                    preproc=TrainTransform(
                        max_labels=50,
                        flip_prob=self.flip_prob,
                        hsv_prob=self.hsv_prob),
                    cache=cache_img,
                    human_pose=False,
                )

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

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        tensor = torch.LongTensor(2) if self.device_type == "cpu" else torch.LongTensor(2).cuda()

        if rank == 0:
            size_factor = self.input_size[1] * 1.0 / self.input_size[0]
            if not hasattr(self, 'random_size'):
                min_size = int(self.input_size[0] / 32) - self.multiscale_range
                max_size = int(self.input_size[0] / 32) + self.multiscale_range
                self.random_size = (min_size, max_size)
            size = random.randint(*self.random_size)
            size = (int(32 * size), 32 * int(size * size_factor))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if is_distributed:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size

    def preprocess(self, inputs, targets, tsize):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            inputs = nn.functional.interpolate(
                inputs, size=tsize, mode="bilinear", align_corners=False
            )
            targets[..., 1::2] = targets[..., 1::2] * scale_x
            targets[..., 2::2] = targets[..., 2::2] * scale_y
        return inputs, targets

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay

            optimizer = torch.optim.SGD(
                pg0, lr=lr, momentum=self.momentum, nesterov=True
            )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            self.optimizer = optimizer

        return self.optimizer

    def get_lr_scheduler(self, lr, iters_per_epoch):
        from yolox.utils import LRScheduler

        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            no_aug_epochs=self.no_aug_epochs,
            min_lr_ratio=self.min_lr_ratio,
        )
        return scheduler

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import COCODataset, LMODataset, YCBVDataset, COCOKPTSDataset, ValTransform, MyDataset

        if self.data_set == "coco":
            valdataset = COCODataset(
                data_dir=self.data_dir,
                json_file=self.val_ann if not testdev else "image_info_test-dev2017.json",
                name="val2017" if not testdev else "test2017",
                img_size=self.test_size,
                preproc=ValTransform(legacy=legacy),
            )
        elif self.data_set == "lm" or self.data_set == "lmo":
            base_dir = "lm" if self.data_set == "lm" else "lmo"
            valdataset = LMODataset(
                data_dir=self.data_dir,
                json_file=self.val_ann if not testdev else "image_info_test-dev2017.json",
                name="test", #if not testdev else "test2017",
                img_size=self.test_size,
                preproc=ValTransform(legacy=legacy, visualize=self.visualize),
                object_pose=self.object_pose,
                base_dir=base_dir
            )
        elif self.data_set == "ycbv":
            valdataset = CustomYCBVDataset(
                data_dir=self.data_dir,
                json_file=self.val_ann if not testdev else "image_info_test-dev2017.json",
                name="images/validation", #if not testdev else "test2017",
                img_size=self.test_size,
                preproc=ValTransform(legacy=legacy, visualize=self.visualize),
                object_pose=self.object_pose
            )
        elif self.data_set == "custom":
            valdataset = CustomDataset(
                data_dir=self.data_dir,
                json_file=self.val_ann if not testdev else "image_info_test-dev2017.json",
                name="validation", #if not testdev else "test2017",
                img_size=self.test_size,
                preproc=ValTransform(legacy=legacy, visualize=self.visualize),
                object_pose=self.object_pose
            )
        elif self.data_set ==  "coco_kpts":
            valdataset = COCOKPTSDataset(
                data_dir=self.data_dir,
                json_file=self.val_ann if not testdev else "image_info_test-dev2017.json",
                name="val2017" if not testdev else "test2017",
                img_size=self.test_size,
                preproc=ValTransform(legacy=legacy),
                human_pose = False
            )
        elif self.data_set == "myDataset":
            valdataset = MyDataset(
                data_dir=self.data_dir,
                json_file=self.val_ann,  # e.g., "val_annotations.json"
                name="images/validation",          # e.g., your validation image folder name
                img_size=self.test_size,
                preproc=ValTransform(legacy=legacy, object_pose=self.object_pose), # Crucial
                object_pose=self.object_pose
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.data_set}")


        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        if self.visualize:
            dataloader_kwargs["batch_size"] = 1
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import COCOEvaluator, ObjectPoseEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        if self.object_pose:
            evaluator = ObjectPoseEvaluator(
                dataloader=val_loader,
                img_size=self.test_size,
                confthre=self.test_conf,
                nmsthre=self.nmsthre,
                num_classes=self.num_classes,
                testdev=testdev,  
            )
        else:
            evaluator = COCOEvaluator(
                dataloader=val_loader,
                img_size=self.test_size,
                confthre=self.test_conf,
                nmsthre=self.nmsthre,
                num_classes=self.num_classes,
                testdev=testdev,
            )
        return evaluator

    def eval(self, model, evaluator, is_distributed, half=False):
        return evaluator.evaluate(model, is_distributed, half)