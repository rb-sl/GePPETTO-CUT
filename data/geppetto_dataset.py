import os.path
from data.unaligned_dataset import UnalignedDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util
import numpy as np
# import torch.nn.functional as F
# import torch
from pathlib import Path
from skimage import io as skio
# from torchvision import io as tvio
import torchvision.transforms.v2.functional as TF


# import matplotlib.pyplot as plt
import numpy as np
# from matplotlib.patches import Circle

# def overlay_centroids_and_save(image, centroids, output_filename='overlay_output.png', show=False):
#     """
#     Overlays centroids onto an image using matplotlib and saves the result.

#     Args:
#         image (np.ndarray): The base image. Can be grayscale (H, W) or color (H, W, 3).
#         centroids (np.ndarray): An array of centroid coordinates of shape (N, 2).
#                                  Assumed to be in [x, y] order, matching the coordinate
#                                  system used by matplotlib.
#         output_filename (str): The name and path of the file to save the image to.
#         show (bool): If True, also displays the plot window.
#     """
#     # 1. Create a figure and axis with a controlled size.
#     # 'figsize' is in inches. We set it based on the image size and 100 DPI
#     # to maintain a reasonable image resolution.
#     dpi = 100
#     h, w = image.shape[:2]
#     figsize = w / dpi, h / dpi
#     fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

#     # 2. Display the base image.
#     # We use 'interpolation=nearest' to prevent matplotlib from smoothing the pixels,
#     # and 'cmap=gray' to correctly display grayscale images if provided.
#     ax.imshow(image, cmap='gray', interpolation='nearest')

#     # 3. Add the centroids as overlays.
#     # We loop through each (x, y) coordinate pair.
#     for centroid in centroids:
#         # Check if the centroid is within image boundaries to prevent errors.
#         if 0 <= centroid[0] < w and 0 <= centroid[1] < h:
#             # We create a simple Circle patch and add it to the axes.
#             # You can change the size (radius), color, and outline (linewidth).
#             circle = Circle((centroid[1], centroid[0]), 
#                              radius=3, # Adjust radius as needed
#                              color='red', # Marker color
#                              linewidth=2, # Outline width
#                              fill=True) # Fill the circle
#             ax.add_patch(circle)

#     # 4. Clean up the plot before saving.
#     # This ensures that axes lines, ticks, and labels are removed, leaving only
#     # the original image content and our overlay.
#     ax.axis('off')

#     # 5. Save the image to the specified filename.
#     # We use tight layout to remove unnecessary white margins around the figure.
#     plt.savefig(output_filename, pad_inches=0)

#     print(f"Overlay image successfully saved to: {output_filename}")

#     # 7. Close the figure to free up system memory.
#     plt.close(fig)


class GeppettoDataset(UnalignedDataset):
    def __init__(self, opt):
        super().__init__(opt)

        self.dir_paired =  Path(os.path.join(opt.dataroot, "paired_dataset")) # opt.dir_paired)
        self.dir_labeled =  Path(os.path.join(opt.dataroot, "labeled_masks")) #  opt.dir_labeled)

        self.paired_A_paths = list((self.dir_paired / "masks").glob("*"))
        self.paired_B_paths = list((self.dir_paired / "images").glob("*"))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = np.array(Image.open(A_path).convert('L'))
        B_img = np.array(Image.open(B_path).convert('L'))
        
        A_img = Image.fromarray(np.clip(A_img, 0, np.max(B_img)).astype(np.uint8)).convert("RGB")
        B_img = Image.fromarray(B_img).convert("RGB")
        # Apply image transformation
        # For CUT/FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        transform = get_transform(modified_opt, grayscale=True)

        # sample_path = Path(A_path)
        # labeled_path = self.dir_labeled / f"synth_sample_{sample_path.stem.split("_")[-1]}.tif"
        # base_mask = Image.fromarray(skio.imread(labeled_path)).convert("RGB")

        # A, base_mask = transform(A_img, base_mask)  # Applies the same transformation
        A = transform(A_img)
        B = transform(B_img)

        # SAM GT
        # uniques, counts = np.unique(base_mask, return_counts=True)
        # uniques = np.delete(uniques, np.argmax(counts))
        # A_exploded = base_mask == uniques[:, None, None]
        # A_centroids = self.get_centroids(A_exploded)

        # Semi-paired GT
        i_paired = np.random.randint(len(self.paired_A_paths))
        A_paired = Image.fromarray(skio.imread(self.paired_A_paths[i_paired]).astype(np.uint8)).convert('RGB')
        B_paired = Image.fromarray(skio.imread(self.paired_B_paths[i_paired]).astype(np.uint8)).convert('RGB')
        A_paired, B_paired = transform(A_paired, B_paired)

        # Random affine 
        if self.opt.isTrain:
            A = self.synchronized_affine(A, interp_mode_A=TF.InterpolationMode.NEAREST)
            B = self.synchronized_affine(B, interp_mode_A=TF.InterpolationMode.BILINEAR)
            A_paired, B_paired = self.synchronized_affine(A_paired, B_paired, interp_mode_A=TF.InterpolationMode.NEAREST)

        # from matplotlib import pyplot as plt
        # plt.figure(figsize=(10, 10))
        # plt.subplot(221)
        # plt.imshow(A[0], cmap='gray')
        # plt.title("A")
        # plt.subplot(222)
        # plt.imshow(B[0], cmap='gray')
        # plt.title("B")
        # plt.subplot(223)
        # plt.imshow(A_paired[0], cmap='gray')
        # plt.title("A_paired")
        # plt.subplot(224)
        # plt.imshow(B_paired[0], cmap='gray')
        # plt.title("B_paired")
        # plt.savefig(f"examples/{i_paired}.png")

        return {'A': A, 'B': B, 
                'A_paths': A_path, 'B_paths': B_path, 
                # 'A_exploded': A_exploded, 'A_centroids': A_centroids,  # SAM GT
                'A_paired': A_paired, 'B_paired': B_paired
                }
    
    def get_centroids(self, masks_onehot):
        """
        Fast vectorized centroid calculation for NumPy arrays.
        Args:
            masks_onehot: Boolean or integer array of shape (N, H, W).
        Returns:
            centroids: Array of shape (N, 2) in [x, y] format.
        """
        N, H, W = masks_onehot.shape
        
        # 1. Create coordinate grids (y_grid is [0], x_grid is [1])
        grid = np.indices((H, W))
        y_grid = grid[0]
        x_grid = grid[1]

        # 2. Calculate areas
        areas = masks_onehot.sum(axis=(1, 2))
        areas = np.maximum(areas, 1e-8) # Prevent zero division

        # 3. Multiply and sum
        sum_y = (masks_onehot * y_grid).sum(axis=(1, 2))
        sum_x = (masks_onehot * x_grid).sum(axis=(1, 2))

        # 4. Average to get centroids
        cent_y = sum_y / areas
        cent_x = sum_x / areas

        # 5. Stack and convert to int - (X, Y) FORMAT IS REQUIRED BY SAM
        centroids = np.stack((cent_x, cent_y), axis=1).astype(int)

        return centroids
       
    def synchronized_affine(self, image, mask=None, interp_mode_A=TF.InterpolationMode.BILINEAR, reflect_size=200):
        # Augmentation parameters sampling
        angle = random.uniform(0.0, 360.0)
        image_side = image.shape[-1]
        max_dx = 0.2 * image_side
        max_dy = 0.2 * image_side
        translate = [int(random.uniform(-max_dx, max_dx)), 
                    int(random.uniform(-max_dy, max_dy))]
        scale = random.uniform(0.8, 1.2)
        shear = random.uniform(-10, 10)

        side_center = (image_side + 2 * reflect_size) / 2
        rotation_center = (side_center, side_center)

        # Pad and crop for reflection of nuclei before rotation / translation / zoom
        image_pad = TF.pad(image, padding=reflect_size, padding_mode="reflect")
        A_aug = TF.affine(image_pad, angle, translate, scale, shear, 
                        interpolation=interp_mode_A, fill=0, center=rotation_center) # CUT usually pads with 0 for reflect prep
        A_aug_crop = TF.crop(A_aug, reflect_size, reflect_size, image_side, image_side)

        if mask is not None:
            mask_pad = TF.pad(mask, padding=reflect_size, padding_mode="reflect")
            B_aug = TF.affine(mask_pad, angle, translate, scale, shear, 
                            interpolation=TF.InterpolationMode.BILINEAR, fill=0, center=rotation_center)
            B_aug_crop = TF.crop(B_aug, reflect_size, reflect_size, image_side, image_side)
            
            return A_aug_crop, B_aug_crop
        
        return A_aug_crop