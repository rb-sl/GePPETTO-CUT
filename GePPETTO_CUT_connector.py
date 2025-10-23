"""
Script adapted from https://github.com/taesungp/contrastive-unpaired-translation/blob/master/test.py
"""
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from skimage import io as skio
from pathlib import Path
import numpy as np


class GePPETTOOptions(TestOptions):
    def initialize(self, parser):
        parser = TestOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--cache_dir', type=str, default='./cache/', help='Dataset cache path.')
        parser.add_argument('--input_masks_dir', type=str, default='./input_masks/', help='Generated masks path.')
        parser.add_argument('--real_images_dir', type=str, default='./real_images/', help='Real images path.')
        parser.add_argument('--example_dir', type=str, default=None, help='Path for examples.')
        parser.add_argument('--output_dir', type=str, default='./outputs/', help='Dataset output path for images.')
        
        return parser


def extract_image(image):
    np_image = image.clamp(-1.0, 1.0).detach().cpu().numpy()
    np_image = (np_image[0].transpose(1, 2, 0) + 1) / 2 * 255
    return np_image.astype(np.uint8)

if __name__ == '__main__':
    opt = GePPETTOOptions().parse()  # get test options
    base_cache_dir = Path(opt.cache_dir)

    input_masks_dir = Path(opt.input_masks_dir)
    real_images_dir = Path(opt.real_images_dir)

    # Converts the masks and saves them in the cache folder
    cache_dir_A = base_cache_dir / "testA"
    fake_cache_trainA = base_cache_dir / "trainA"
    if cache_dir_A.is_symlink():
        cache_dir_A.unlink()
        fake_cache_trainA.unlink()
    cache_dir_A.symlink_to(input_masks_dir)
    fake_cache_trainA.symlink_to(input_masks_dir)
   
    # Saves the images in the cache folder
    cache_dir_B = base_cache_dir / "testB"
    fake_cache_trainB = base_cache_dir / "trainB"
    if cache_dir_B.is_symlink():
        cache_dir_B.unlink()
        fake_cache_trainB.unlink()
    cache_dir_B.symlink_to(real_images_dir)
    fake_cache_trainB.symlink_to(real_images_dir)

    output_dir = Path(opt.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    example_dir = Path(opt.example_dir)
    show_example = example_dir is not None
    if show_example:
        example_dir.mkdir(exist_ok=True, parents=True)

    opt.dataroot = str(base_cache_dir)

    # CUT inference code
    # hard-code some parameters for test
    opt.isTrain = False
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    # train_dataset = create_dataset(util.copyconf(opt, phase="train"))
    model = create_model(opt)      # create a model given opt.model and other options

    for i, data in enumerate(dataset):
        if i == 0:
            model.data_dependent_initialize(data)
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            model.parallelize()
            if opt.eval:
                model.eval()
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results

        image_id = Path(data["A_paths"][0]).stem.split("_")[-1]

        if show_example:
            for image_name, image in visuals.items():
                skio.imsave(example_dir / f"{image_id}_{image_name}.tif", extract_image(image))
            show_example = False
        
        skio.imsave(output_dir / f"synth_sample_{image_id}.tif", extract_image(visuals["fake_B"]).squeeze())

    # fake_cache_trainA.unlink()
    # shutil.rmtree(cache_dir_A)
    # fake_cache_trainB.unlink()
    # shutil.rmtree(cache_dir_B)
