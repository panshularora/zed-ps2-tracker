# ByteTrack/YOLOX experiment: fine-tune YOLOX-X on MOT20 train FIRST HALF only.
# Used by scripts/dgx/finetune_mot20_half.sh. Starts from bytetrack_ablation (CrowdHuman +
# MOT17 half); never sees MOT20 val-half frames, so the resulting cache is clean for val-half.
# Untested on GPU. Data root: $MOT20_ROOT (with annotations/{train_half,val_half}.json).
import os

import torch.distributed as dist
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.num_classes = 1
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = "yolox_x_mot20_half"
        self.data_root = os.environ["MOT20_ROOT"]
        self.train_ann = "train_half.json"
        self.val_ann = "val_half.json"
        self.input_size = (896, 1600)
        self.test_size = (896, 1600)
        self.random_size = (20, 36)
        self.max_epoch = int(os.environ.get("EPOCHS", 10))
        self.no_aug_epochs = 2
        self.eval_interval = 5
        self.print_interval = 20
        self.test_conf = 0.001
        self.nmsthre = 0.7
        self.basic_lr_per_img = 0.001 / 64.0 / 4  # fine-tune: 1/4 of ByteTrack's LR
        self.warmup_epochs = 1

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            MOTDataset,
            TrainTransform,
            YoloBatchSampler,
        )

        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        dataset = MOTDataset(
            data_dir=self.data_root,
            json_file=self.train_ann,
            name="",
            img_size=self.input_size,
            preproc=TrainTransform(rgb_means=mean, std=std, max_labels=600),
        )
        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(rgb_means=mean, std=std, max_labels=1200),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
        )
        self.dataset = dataset
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
        sampler = InfiniteSampler(len(dataset), seed=self.seed if self.seed else 0)
        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
        )
        return DataLoader(
            dataset, num_workers=self.data_num_workers, pin_memory=True, batch_sampler=batch_sampler
        )

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        import torch
        from yolox.data import MOTDataset, ValTransform

        ds = MOTDataset(
            data_dir=self.data_root,
            json_file=self.val_ann,
            name="",
            img_size=self.test_size,
            preproc=ValTransform(rgb_means=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        )
        sampler = (
            torch.utils.data.distributed.DistributedSampler(ds, shuffle=False)
            if is_distributed
            else torch.utils.data.SequentialSampler(ds)
        )
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
        return torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.data_num_workers,
            pin_memory=True,
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import COCOEvaluator

        return COCOEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
