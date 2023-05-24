"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import torch
import nibabel as nib
# from visdom import Visdom
# viz = Visdom(port=8097)
import sys
import random

sys.path.append(".")
import numpy as np
import time
import torch as th
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.lidcloader import LIDCDataset
from guided_diffusion.lidcloader_mose import lidc_Dataloader
from guided_diffusion.msmri_dataset_mose import msmri_Dataloader
from metrics import calc_batched_generalised_energy_distance, batched_hungarian_matching, model_size
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

seed = 10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img


def dice_score(pred, targs):
    pred = (pred > 0).float()
    return 2. * (pred * targs).sum() / (pred + targs).sum()


def expanduservars(path: str) -> str:
    return os.path.expanduser(os.path.expandvars(path))


def main():
    # args = create_argparser().parse_args()
    args = create_argparser().parse_args()

    # world_size = args.ngpu

    # os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    # os.environ["MASTER_PORT"] = str(1028)
    #
    # torch.distributed.init_process_group(
    # 'gloo',
    # init_method='env://',
    # world_size=world_size,
    # rank=args.local_rank,)

    dist_util.setup_dist()

    use_mose_dataset = True
    use_dataset = "lidc"  # "msmri" or "lidc"

    if use_mose_dataset:
        this_run = expanduservars("eval_${NOW}")
        out_dir = "./results/" + use_dataset + "/" + this_run
        logger.configure(dir=out_dir)
        logger.info(f"Log dir: {out_dir}")
    else:
        logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion, prior, posterior = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model_size(model, diffusion, prior, posterior, logger)

    # torch.cuda.set_device(args.local_rank)

    # model.to(dist_util.dev())
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model = torch.nn.parallel.DistributedDataParallel(
    # model,
    # device_ids=[args.local_rank],
    # output_device=args.local_rank,
    # )

    if not use_mose_dataset:
        ds = LIDCDataset(args.data_dir, test_flag=True)
    else:
        if use_dataset == "lidc":
            ds = lidc_Dataloader(
                data_folder="/storage/homefs/lz20w714/git/mose-auseg/data/lidc_npy",
                # data_folder="/home/lukas/git/mose-auseg/data/lidc_npy",
                transform_train=None,
                transform_test=None
            ).test_ds
            # from torch.utils.data.dataset import Subset
            # ds = Subset(ds, [6, 8, 9, 13, 14, 15, 16])

        elif use_dataset == "msmri":
            ds = msmri_Dataloader(
                data_folder="/storage/homefs/lz20w714/git/mose-auseg/data/msmri_npy",
                # data_folder="/home/lukas/git/mose-auseg/data/msmri_npy",
                transform_train=None,
                transform_test=None
            ).test_ds
        else:
            assert False, "unknown dataset"

    logger.info("Arguments: %s" % args.__dict__)

    # sampler = torch.utils.data.distributed.DistributedSampler(
    # ds,
    # num_replicas=args.ngpu,
    # rank=args.local_rank,
    # )

    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False)
    len_dataset = len(ds)
    data = iter(datal)
    all_images = []
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    img_cnt = 0
    geds = 0
    hm_ious = 0
    # assert args.batch_size == 1, f"cf. gaussian_diffusion.py: L579"
    data_len = len(data)
    # while img_cnt < data_len:
    #     img_cnt += 1
    #     print("Test sample ", img_cnt, "/", data_len)
    #
    #     b, label = next(data)  # should return an image from the dataloader "data"

    num_imgs = 0

    logger.log(f"Calculating GED and HM-Iou with N={args.num_ensemble} over {len_dataset} samples..")

    for idx, (b, label) in enumerate(data):
        # print("Test batch ", idx + 1, "/", data_len)

        c = th.randn_like(b[:, :1, ...])
        img = th.cat((b, c), dim=1)  # add a noise channel$
        # slice_ID = path[0].split("/", -1)[3]

        # viz.image(visualize(img[0]), opts=dict(caption="img input0"))
        # viz.image(visualize(img[0,0,...]), opts=dict(caption="image input"))
        # viz.image(visualize(label), opts=dict(caption="gt"))
        # viz.image(visualize(img[0, 2, ...]), opts=dict(caption="img input2"))
        # viz.image(visualize(img[0, 3, ...]), opts=dict(caption="img input3"))
        # viz.image(visualize(img[0, 4, ...]), opts=dict(caption="noise"))

        # logger.log("sampling...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)

        img = img.repeat_interleave(args.num_ensemble, dim=0)

        # for i in range(args.num_ensemble):  # this is for the generation of an ensemble of 5 masks.
        model_kwargs = {}
        start.record()
        sample_fn = (
            diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
        )

        sample, x_noisy, org = sample_fn(
            model,
            (b.shape[0] * args.num_ensemble, -1, args.image_size, args.image_size), img,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )

        end.record()
        th.cuda.synchronize()

        # normalize sample to range [0,1]:
        sample = (sample - sample.min()) / (sample.max() - sample.min())

        # WOLLEB: This mask is thresholded at 0.5 to obtain a binary segmentation
        sample = th.where(sample > 0.5, th.ones_like(sample), th.zeros_like(sample))

        # predictions.append(sample)
        # predictions = th.stack(predictions, dim=1)

        # needed?
        sample = sample.reshape(label.shape[0], -1, *label.shape[2:])  # (2, 5, 2, 128, 128)
        predictions = sample  # eq. to sample[:, :args.num_ensemble]

        label = label

        ged = calc_batched_generalised_energy_distance(label.cpu().numpy(), predictions.cpu().numpy(), NUM_CLASSES)
        geds += np.sum(ged)

        lcm = np.lcm(args.num_ensemble, label.shape[1])
        hm_labels = label.repeat_interleave(lcm // label.shape[1], dim=1).cpu().numpy()
        predictions = predictions.repeat_interleave(lcm // args.num_ensemble, dim=1).cpu().numpy()
        assert all([p in [0, 1] for p in np.unique(predictions)]), "predictions must contain all classes"
        hm_iou = batched_hungarian_matching(hm_labels, predictions, NUM_CLASSES)
        hm_ious += np.sum(hm_iou)
        num_imgs += b.shape[0]
        logger.info("%s | Batch %d/%d (%d/%d) | GED_%d: %.4g, HM-IoU_%d: %.4g" % (time.strftime("%Y-%m-%d %H:%M:%S"),
        idx + 1, data_len, num_imgs, len_dataset, args.num_ensemble, np.sum(ged) / predictions.shape[0], args.num_ensemble, np.sum(hm_iou) / predictions.shape[0]))

    logger.info("\n\n%s | GED_%d: %.4g, HM-IoU_%d: %.4g" % (time.strftime("%Y-%m-%d %H:%M:%S"), args.num_ensemble, geds / len_dataset, args.num_ensemble, hm_ious / len_dataset))


def create_argparser():
    defaults = dict(
        data_dir="./data/testing",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
        num_ensemble=4  # number of samples in the ensemble
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=2)
    parser.add_argument('--ngpu', type=int, default=2)
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
