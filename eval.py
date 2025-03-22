"""
Evaluate an end-to-end compression model on an image dataset.
Based on compressai.utils.eval_model,
improved by LALIC: Linear Attention Modeling for Learned Image Compression.
"""

import os
import re
import sys
import time
import glob
import math
import json
import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image
from pytorch_msssim import ms_ssim

import compressai
from compressai.registry import MODELS
from models import LALIC

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)


def compute_psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return -10 * math.log10(mse)


def compute_msssim_db(a, b):
    return -10 * math.log10(1 - ms_ssim(a, b, data_range=1.0).item())


img_metrics = {
    "psnr": compute_psnr,
    "ms-ssim-db": compute_msssim_db,
}


def torch2img(x: torch.Tensor) -> Image.Image:
    return ToPILImage()(x.clamp_(0, 1).squeeze())


def reglob_collect_images(rootpath):
    file_list = glob.glob(os.path.join(rootpath, "**/*.*"), recursive=True)
    formats = "jpg|jpeg|png|webp"
    pattern = f"(?i)([^\\s]+(\\.({formats}))$)"
    result = filter(re.compile(pattern).match, file_list)
    return sorted(result)


def read_image(filepath: str) -> torch.Tensor:
    assert os.path.isfile(filepath)
    img = Image.open(filepath).convert("RGB")
    return ToTensor()(img)


def pad_centering(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)


def unpad_centering(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )


def load_checkpoint(net_cls, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    net = net_cls.from_state_dict(state_dict)
    return net.eval()


@torch.no_grad()
def inference_real(model, x, fout=""):
    x = x.unsqueeze(0)
    x_padded, padding = pad_centering(x, 128)

    start = time.time()
    out_enc = model.compress(x_padded)
    enc_time = time.time() - start

    start = time.time()
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
    dec_time = time.time() - start

    out_dec["x_hat"].clamp_(0, 1)
    out_dec["x_hat"] = unpad_centering(out_dec["x_hat"], padding)

    if fout:
        img = torch2img(out_dec["x_hat"])
        img.save(fout)

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp_items = [len(s[0]) * 8.0 / num_pixels for s in out_enc["strings"]]
    bpp = sum(bpp_items)

    iqa_result = {key: func(out_dec["x_hat"], x) for key, func in img_metrics.items()}
    org_result = {
        "enc_time": enc_time,
        "dec_time": dec_time,
        "bpp": bpp,
    }
    org_result.update(iqa_result)
    return org_result


@torch.no_grad()
def inference_esti(model, x, fout=""):
    x = x.unsqueeze(0)
    x_padded, padding = pad_centering(x, 128)

    start = time.time()
    out_net = model.forward(x_padded)
    elapsed_time = time.time() - start

    out_net["x_hat"].clamp_(0, 1)
    out_net["x_hat"] = unpad_centering(out_net["x_hat"], padding)

    if fout:
        img = torch2img(out_net["x_hat"])
        img.save(fout)

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp_items = [
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods in out_net["likelihoods"].values()
    ]
    bpp = sum(bpp_items).item()

    iqa_result = {
        key: func(out_net["x_hat"], x).item() for key, func in img_metrics.items()
    }
    org_result = {
        "enc_time": elapsed_time / 2.0,  # broad estimation
        "dec_time": elapsed_time / 2.0,
        "bpp": bpp,
    }
    org_result.update(iqa_result)
    return org_result


def eval_model(model, quality, args):
    device = next(model.parameters()).device
    avg_metrics = defaultdict(float)
    records = []

    filepaths = reglob_collect_images(args.input_dir)
    if len(filepaths) == 0:
        print(f"Error: no images found in {args.input_dir}.", file=sys.stderr)
        raise SystemExit(1)
    if args.output_dir:
        out_sub_dir = f"{args.output_dir}/{quality}"
        os.makedirs(out_sub_dir, exist_ok=True)

    for file in filepaths:
        fout = ""
        if args.output_dir:
            fout = os.path.join(out_sub_dir, os.path.basename(file))
        x = read_image(file).to(device)
        if args.half:
            model = model.half()
            x = x.half()
        if args.real:
            model.update()
            rv = inference_real(model, x, fout)
        else:
            rv = inference_esti(model, x, fout)
        for k, v in rv.items():
            avg_metrics[k] += v

        record = {"file": file, "quality": quality}
        if args.verbose:
            _rv = {key: round(value, 4) for key, value in rv.items()}
            print(file, _rv)
        rv = {key: round(value, 8) for key, value in rv.items()}
        record.update(rv)
        records.append(record)
    for k, v in avg_metrics.items():
        avg_metrics[k] = v / len(filepaths)
        avg_metrics[k] = round(avg_metrics[k], 6)
    return avg_metrics, records


def setup_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="model architecture",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--entropy-coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="entropy coder (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="enable CUDA")
    parser.add_argument(
        "--half",
        action="store_true",
        help="convert model to half floating point (fp16)",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="use evaluated entropy estimation (no entropy coding)",
    )
    parser.add_argument(
        "-p",
        "--checkpoint",
        dest="checkpoints",
        type=str,
        nargs="*",
        help="checkpoint path list",
    )
    parser.add_argument(
        "-q",
        "--quality",
        dest="qualities",
        nargs="*",
        required=True,
        help="quality labels correspoding to the checkpoint path list",
    )
    parser.add_argument("-i", "--input-dir", type=str)
    parser.add_argument("-o", "--output-dir", type=str, default="")
    parser.add_argument("-r", "--result", type=str, help="result file path")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose mode")
    return parser


def main(argv):
    parser = setup_args()
    args = parser.parse_args(argv)
    assert len(args.checkpoints) == len(args.qualities), (
        "checkpoint and quality labels mismatch"
    )

    # model = LALIC(
    #     dims=[96, 144, 256, 320, 256, 192],
    #     depths=[2, 4, 6, 6],
    # ) # fixed model for now
    # net_cls = LALIC
    if args.model in MODELS:
        net_cls = MODELS[args.model]
    else:
        raise ValueError(f"Model {args.model} not found.")

    all_avg_metrics = defaultdict(list)
    all_records = []
    for ckpt, quality in zip(args.checkpoints, args.qualities):
        if args.verbose:
            print(f"\nEvaluating ckpt: {ckpt}")
        model = load_checkpoint(net_cls, ckpt)
        if args.cuda and torch.cuda.is_available():
            model = model.to("cuda")
        # model.update(force=True)
        avg_metrics, records = eval_model(model, quality, args)
        all_records.extend(records)
        for k, v in avg_metrics.items():
            all_avg_metrics[k].append(v)
        mem = torch.cuda.max_memory_allocated(device=None)  # bytes
        print(f"GPU mem: \t{mem / (2**30):.3f} GB")
        del model

    if args.verbose:
        print()

    used_coder = "estimation" if not args.real else args.entropy_coder
    result = {
        "name": f"{args.model}",
        "description": f"{args.model}, coding: {used_coder}, cuda: {args.cuda}, quality: {args.qualities}",
        "results": all_avg_metrics,
    }
    print(json.dumps(result, indent=2))
    result["records"] = all_records
    if args.result:
        with open(args.result, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1:])
    # for example:
    # CUDA_VISIBLE_DEVICES=0 python eval.py -m LALIC -p ckpt/lalic-q1.pth ckpt/lalic-q2.pth  -q 1 2 -i ~/Datasets/Kodak -o eval_out --result result.json --cuda --real --verbose
