import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["PYTHONPATH"] = f"/home/robertob98/GePPETTO-DET-project/external/GePPETTO-CUT:$PYTHONPATH"

from pathlib import Path
from skimage import io as skio, morphology, measure
import numpy as np
import random
from matplotlib import pyplot as plt

import torch.nn.functional as F
from torchvision.transforms import v2 as transforms

from segment_anything import sam_model_registry
from torchvision.ops import sigmoid_focal_loss
from scipy.optimize import linear_sum_assignment
import torch
from matplotlib.patches import Circle

from tqdm.auto import tqdm
import shutil
import torch_fidelity
from PIL import Image

import argparse

from options.test_options import TestOptions
from data.geppetto_dataset import GeppettoDataset
from models.cut_model import CUTModel
from data import create_dataset
from models import create_model
import json

import subprocess

def make_cache(cache_path, images_path):
    cache_path.mkdir(exist_ok=True, parents=True)
    for image_path in images_path.glob("*.tif"):
        Image.fromarray(skio.imread(image_path).astype(np.uint8)).convert("RGB").save(cache_path / f"{image_path.stem}.png")
    # for i in image_path:
    #     shutil.copy(i, cache_path / i.name)
    
def compute_fid(path_1, path_2, epoch, cache_path, verbose=True):
    cache_path_1 = cache_path / f"c1_{epoch}"
    cache_path_2 = cache_path / f"c2_{epoch}"
    make_cache(cache_path_1, path_1)
    make_cache(cache_path_2, path_2)

    n_samples_1 = len(list(cache_path_1.glob("*")))
    n_samples_2 = len(list(cache_path_2.glob("*")))

    fid_params = dict(cuda=True,
        fid=True
    )

    metrics = torch_fidelity.calculate_metrics(
        input1=str(cache_path_1),
        input2=str(cache_path_2),
        kid_subset_size=min(1000, min(n_samples_1, n_samples_2)),
        verbose=verbose,
        **fid_params)
    
    shutil.rmtree(cache_path_1)

    return metrics


class GePPETTOOptions(TestOptions):
    def initialize(self, parser):
        parser = TestOptions.initialize(self, parser)  # define shared options

        parser.add_argument('--cache_dir', type=str, default='./cache/', help='Dataset cache path.')
        parser.add_argument('--masks_path', type=str, default='./masks/', help='Generated masks path.')
        parser.add_argument('--gen_path', type=str, default='./gen_images/', help='Real images path.')
        parser.add_argument('--example_dir', type=str, default=None, help='Path for examples.')
        
        parser.add_argument("--sam_path",  default=Path("../sam_vit_b_01ec64.pth"), type=Path, help="Path to SAMv1 model weights")
        parser.add_argument('--output_dir', default=Path("/mnt/experiments/robertob98/CUT_evaluation"), type=Path, help='Evaluation output')
        parser.add_argument("--seed", default=42, type=int, help="Seed for reproducibility")
        
        parser.add_argument("--geppetto_python", default="python", type=str, help="Python for running GePPETTO-DET")
        parser.add_argument("--geppetto_home", default=Path("../../../GePPETTO-DET/"), type=Path, help="Path to GePPETTO-DET folder")
        parser.add_argument("--geppetto_config_path", default=Path("./geppetto_evaluation.json"), type=Path, help="Path to GePPETTO-DET folder")
        parser.add_argument("--geppetto_output", default=Path("./geppetto_output/"), type=Path, help="Path to save GePPETTO-DET output")

        return parser

if __name__ == '__main__':
    opt = GePPETTOOptions().parse()

    opt.masks_path = Path("/mnt/experiments/robertob98/GePPETTO_DET/outputs_conv/test_sam_gen/synth_BBBC039_split_norm_tiled_allimages_100maskedtiles_256_0/train/masks")
    opt.gen_path = Path("/home/robertob98/GePPETTO-DET-project/external/GePPETTO-CUT/cache_fid_gen")
    opt.geppetto_python = "/home/robertob98/envs/geppetto_env/bin/python"
    opt.geppetto_home = Path("/home/robertob98/GePPETTO-DET/")
    opt.geppetto_output = Path("/mnt/experiments/robertob98/GePPETTO_DET/outputs_eval/")
    opt.dataroot = "/mnt/experiments/robertob98/GePPETTO_DET/outputs_conv/test_sam/cutBBBC039_split_norm_tiled_allimages_100maskedtiles_256_0"
    opt.dataset_mode = "geppetto"

    opt.CUT_checkpoints_dir = "/mnt/experiments/robertob98/CUT/"
    opt.CUT_name = "CUT_dicefocal_nospaced"
    
    random.seed(opt.seed)
    np.random.seed(opt.seed)

    opt.output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Saving results in", opt.output_dir.absolute())
    evaluation_output_path = opt.output_dir / "evaluation_results.json"

    device = "cuda:0"

    geppetto_env = {
        'PYTHONUNBUFFERED': '1', 
        'CUDA_VISIBLE_DEVICES': '1',
        'PYTHONPATH': f'{str(opt.geppetto_home)}:$PYTHONPATH'
    }

    experiment_name = "cut_evaluation"

    geppetto_command = [
        opt.geppetto_python, 
        str((opt.geppetto_home / "src" / "generate_dataset.py").absolute()),
        "-c", str(opt.geppetto_config_path.absolute()),
        "-x", experiment_name,
        "--output_path", str(opt.geppetto_output),
        "--seed", str(opt.seed),
        "--show_examples"
    ]

    opt.isTrain = True
    opt.phase = 'train'
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.gan_mode = 'vanilla'
    opt.lr = 1e-4
    opt.beta1 = 0.5
    opt.beta2 = 0.5
    dataset_obj = create_dataset(opt).dataset
    model_obj = create_model(opt)

    transform_list = [transforms.ToImage(), 
                  transforms.ToDtype(torch.float32, scale=True), 
                  transforms.Normalize((0.5,), (0.5,))]
    to_tensor = transforms.Compose(transform_list)

    sam_loss_history = {}
    sam_tp_loss_history = {}
    sam_fp_loss_history = {}
    n_false_positives_history = {}
    fid_history = {}
    for epoch in range(5, 101, 5):
        print(">>> Epoch", epoch)
        # Update the configuration to use the current epoch
        with open(opt.geppetto_config_path, "r") as f:
            geppetto_conf = json.load(f)
        geppetto_conf["texture_synthesis"]["epoch"] = epoch
        geppetto_conf["texture_synthesis"]["checkpoints_dir"] = str(opt.CUT_checkpoints_dir)
        geppetto_conf["texture_synthesis"]["name"] = opt.CUT_name

        real_images_path = Path(geppetto_conf["real_samples_path"])

        with open(opt.geppetto_config_path, "w") as f:
            geppetto_conf = json.dump(geppetto_conf, f)     

        # Call and wait for GePPETTO-DET
        subprocess.Popen(geppetto_command, 
                         cwd=opt.geppetto_home, 
                         env=geppetto_env).wait()

        base_path = list((opt.geppetto_output / experiment_name).glob("synth_*"))[0]
        images_path = base_path / "train" / "images"
        masks_path = base_path / "train" / "masks"

        sam_loss_history[epoch] = []
        sam_tp_loss_history[epoch] = []
        sam_fp_loss_history[epoch] = []
        n_false_positives_history[epoch] = []
        for sample_i in tqdm(range(0, 30), desc=f"Computing SAM loss (e={epoch})"):
            image = skio.imread(images_path / f"synth_sample_{sample_i}.tif")
            mask = skio.imread(masks_path / f"synth_sample_{sample_i}.tif")

            # SAM loss computation

            uniques, counts = np.unique(mask[None, ...], return_counts=True)
            uniques = np.delete(uniques, np.argmax(counts))
            A_exploded = mask == uniques[:, None, None]
            A_centroids = dataset_obj.get_centroids(A_exploded)

            fake_tensor = to_tensor(image).to(device)[None, ...]
            exploded_tensor = torch.Tensor(A_exploded).to(device)[None, ...]
            centroids_tensor = torch.Tensor(A_centroids).to(device)[None, ...]

            fp_centroids = model_obj.get_fp_centroids(fake_tensor, targets=exploded_tensor)

            if fp_centroids.shape[1] > 0:
                # Append false positive centroids to real centroids
                all_centroids = torch.cat([centroids_tensor.to(device), fp_centroids.to(device)], dim=1)
                # Append fake targets (all zeros) to match the FP centroids
                all_targets = torch.cat([exploded_tensor.to(device), torch.zeros((1, fp_centroids.shape[1], *exploded_tensor.shape[2:])).to(device)], dim=1)  # batch_size, 
            else:
                all_centroids = centroids_tensor.to(device)
                all_targets = exploded_tensor

            n_fp = fp_centroids.shape[1]

            sam_total_loss = model_obj.compute_sam_loss(fake_tensor, all_centroids, all_targets, n_fp=n_fp, lambda_sam=10)

            sam_loss_history[epoch].append(float(sam_total_loss.detach().cpu().numpy()))
            sam_tp_loss_history[epoch].append(float(model_obj.loss_tp_SAM.detach().cpu().numpy()))
            sam_fp_loss_history[epoch].append(float(model_obj.loss_fp_SAM.detach().cpu().numpy() if model_obj.loss_fp_SAM > 0 else 0))
            n_false_positives_history[epoch].append(n_fp)

        # FID computation
        print(f"Computing FID epoch {epoch}...")
        fid_history[epoch] = compute_fid(images_path, 
                                         real_images_path, 
                                         epoch, 
                                         cache_path=opt.geppetto_output / experiment_name / "cache/fid", 
                                         verbose=False)['frechet_inception_distance']
    
        all_results = {
            "sam_loss_history": sam_loss_history,
            "sam_tp_loss_history": sam_tp_loss_history,
            "sam_fp_loss_history": sam_fp_loss_history,
            "n_false_positives_history": n_false_positives_history,
            "fid_history": fid_history
        }

        with open(evaluation_output_path, "w") as f:
            json.dump(all_results, f)

        print("Intermediate results saved at", evaluation_output_path.absolute())
