import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
import torch.nn.functional as F
from torchvision.transforms import v2 as transforms

from segment_anything import sam_model_registry
from torchvision.ops import sigmoid_focal_loss
from scipy.optimize import linear_sum_assignment

import torch.nn as nn

class CUTModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices=['CUT', 'cut', 'FastCUT', 'fastcut'])

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss: GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        
        parser.add_argument('--discriminator', type=str, default="unconditional", choices=["unconditional", "conditional", "dual"], help='discriminator mode')
        parser.add_argument('--use_sam_after', type=int, default=10, help='Activates the SAM loss computation after the given epoch')

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE', "SAM", "tp_SAM", "fp_SAM", "G_cond", "D_cond", "D_cond_real", "D_cond_fake"]
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.discriminator_mode = opt.discriminator
            self.model_names = ['G', 'F']
            if self.discriminator_mode in ["dual", "unconditional"]:
                self.model_names.append('D')
            if self.discriminator_mode in ["dual", "conditional"]:
                self.model_names.append('D_cond')    
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        self.print_receptive_field(self.netG, "Generator")

        if self.isTrain:
            if self.discriminator_mode in ["dual", "unconditional"]:
                self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
                self.print_receptive_field(self.netD, "Unconditional discriminator")
            else:
                self.netD = None
                print("Unconditional discriminator not set")

            if self.discriminator_mode in ["dual", "conditional"]:
                cond_input_nc = opt.input_nc + opt.output_nc
                self.netD_cond = networks.define_D(cond_input_nc, opt.ndf, opt.netD, opt.n_layers_D + 3, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
                self.print_receptive_field(self.netD_cond, "Conditional discriminator")

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)

            if self.discriminator_mode in ["dual", "unconditional"]:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
                self.optimizers.append(self.optimizer_D)

            if self.discriminator_mode in ["dual", "conditional"]:
                self.optimizer_D_cond = torch.optim.Adam(self.netD_cond.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
                self.optimizers.append(self.optimizer_D_cond)
            
            # self.SAM = SAM("sam2_b.pt")
            self.SAM = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth" if not hasattr(opt, 'sam_path') else opt.sam_path)
            self.SAM.to(self.device)
            for param in self.SAM.parameters():
                param.requires_grad = False
            self.SAM.eval()
            self.SAM_epoch = opt.use_sam_after
            

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        if hasattr(self, "centroids_A"):
            self.centroids_A = self.centroids_A[:bs_per_gpu]
            self.exploded_A = self.exploded_A[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss(epoch=0).backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self, epoch):
        # forward
        self.forward()

        # update D
        if self.discriminator_mode in ["dual", "unconditional"]:
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
        if self.discriminator_mode in ["dual", "conditional"]:
            self.set_requires_grad(self.netD_cond, True)
            self.optimizer_D_cond.zero_grad()
            
        self.compute_D_loss().backward()

        if self.discriminator_mode in ["dual", "unconditional"]:
            self.optimizer_D.step()
        if self.discriminator_mode in ["dual", "conditional"]:
            self.optimizer_D_cond.step()

        # update G
        if self.discriminator_mode in ["dual", "unconditional"]:
            self.set_requires_grad(self.netD, False)
        if self.discriminator_mode in ["dual", "conditional"]:
            self.set_requires_grad(self.netD_cond, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss(epoch=epoch)
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        if 'A_paired' in input.keys():
            self.A_paired = input['A_paired'].to(self.device)
            self.B_paired = input['B_paired'].to(self.device)
        if 'A_centroids' in input.keys():
            self.centroids_A = input["A_centroids"]
            self.exploded_A = input["A_exploded"]
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()

        self.loss_D = 0
        if self.discriminator_mode in ["dual", "unconditional"]:
            # Fake; stop backprop to the generator by detaching fake_B
            pred_fake = self.netD(fake)
            self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
            # Real
            self.pred_real = self.netD(self.real_B)
            loss_D_real = self.criterionGAN(self.pred_real, True)
            self.loss_D_real = loss_D_real.mean()

            # combine loss and calculate gradients
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D_cond = 0.0
        self.loss_D_cond_fake = 0.0
        self.loss_D_cond_real = 0.0
            
        # Only compute this if paired data was provided in this batch
        if hasattr(self, 'A_paired') and hasattr(self, 'B_paired'):
            if self.discriminator_mode == "dual":
                # Generate fake image from the PAIRED mask
                fake_B_paired = self.netG(self.A_paired)                
                # Concatenate Mask + Fake Image
                fake_paired = torch.cat((self.A_paired, fake_B_paired), dim=1)
                pred_fake_cond = self.netD_cond(fake_paired.detach())
                self.loss_D_cond_fake = self.criterionGAN(pred_fake_cond, False).mean()

                # Concatenate Mask + Real Image
                real_AB = torch.cat((self.A_paired, self.B_paired), dim=1)
                pred_real_cond = self.netD_cond(real_AB)
                self.loss_D_cond_real = self.criterionGAN(pred_real_cond, True).mean()
                
                self.loss_D_cond = (self.loss_D_cond_fake + self.loss_D_cond_real) * 0.5
            elif self.discriminator_mode == "conditional":
                fake_AB = torch.cat((self.real_A, fake), dim=1)
                pred_fake = self.netD_cond(fake_AB)

                self.loss_D_cond_fake = self.criterionGAN(pred_fake, False).mean()
                # Real
                real_AB = torch.cat((self.A_paired, self.B_paired), dim=1)
                self.pred_real = self.netD_cond(real_AB)
                self.loss_D_cond_real = self.criterionGAN(self.pred_real, True)
                self.loss_D_cond_real = loss_D_real.mean()

                # combine loss and calculate gradients
                self.loss_D_cond = (self.loss_D_cond_fake + self.loss_D_cond_real) * 0.5
        
        return self.loss_D + self.loss_D_cond

    def compute_G_loss(self, epoch):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0 and self.discriminator_mode in ["dual", "unconditional"]:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        # 2. Fool Conditional D (Structure)
        self.loss_G_cond = 0.0
        if hasattr(self, 'A_paired') and hasattr(self, 'B_paired') \
              and self.discriminator_mode in ["dual", "conditional"]:
            fake_B_paired = self.netG(self.A_paired)
            fake_AB = torch.cat((self.A_paired, fake_B_paired), dim=1)
            
            # Generator wants D_cond to classify this as TRUE
            # We apply a HIGH weight (e.g., lambda_cond = 5.0) because 
            # paired data is rare and must be respected!
            # if epoch < 30:
            #     lambda_cond = 0
            # elif epoch < 60:
            #     lambda_cond = 10
            # elif epoch < 80:
            #     lambda_cond = 20
            # else:
            #     lambda_cond = 40
            lambda_cond = 5 if epoch < 50 else 20  #self.opt.lambda_GAN if not "dual" else 5
            self.loss_G_cond = self.criterionGAN(self.netD_cond(fake_AB), True).mean() * lambda_cond

        # TODO assuming batch = 1

        if epoch > self.SAM_epoch:
            # fp_centroids = self.get_fp_centroids(fake, targets=self.exploded_A)

            # if fp_centroids.shape[1] > 0:
            #     # Append false positive centroids to real centroids
            #     all_centroids = torch.cat([self.centroids_A.to(fake.device), fp_centroids], dim=1)
            #     # Append fake targets (all zeros) to match the FP centroids
            #     all_targets = torch.cat([self.exploded_A, torch.zeros((1, fp_centroids.shape[1], *self.exploded_A.shape[2:]))], dim=1)  # batch_size, 
            # else:
            #     all_centroids = self.centroids_A.to(fake.device)
            #     all_targets = self.exploded_A
                
            # self.loss_SAM = self.compute_sam_loss(fake, centroids=all_centroids, targets=all_targets, n_fp=fp_centroids.shape[1], lambda_sam=10)
            self.loss_SAM = self.compute_sam_loss(fake, centroids=self.centroids_A.to(fake.device), targets=self.exploded_A, n_fp=0, lambda_sam=1)
        else:
            self.loss_SAM = 0
            self.loss_tp_SAM = 0
            self.loss_fp_SAM = 0
        
        self.loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_G_cond + self.loss_SAM #+ self.loss_COLOR
        
        return self.loss_G
    
    def soft_dice_loss(self, logits, targets, smooth=1e-2):
        probs = torch.sigmoid(logits)
        
        probs_flat = probs.view(probs.shape[0], -1)
        targets_flat = targets.view(targets.shape[0], -1).float()
        
        # Intersection remains linear
        intersection = (probs_flat * targets_flat).sum(dim=1)
        
        # SQUARING the union terms kills the low-confidence background noise!
        union = (probs_flat ** 2).sum(dim=1) + (targets_flat ** 2).sum(dim=1)
        
        # We use a larger smooth value (1.0 instead of 1e-5) to prevent instability 
        # when squaring very small numbers
        dice_score = (2. * intersection + smooth) / (union + smooth)
        
        return (1.0 - dice_score).mean()

    def dice_focal_loss(self, inputs, targets, alpha, gamma, reduction):
        # 1. Compute your pixel-level loss (Provides crisp boundary gradients)
        # We use reduction='mean' here to keep it scaled down
        loss_focal = sigmoid_focal_loss(inputs, targets, alpha=alpha, gamma=gamma, reduction=reduction)

        # 2. Compute your shape-level loss (Provides background-agnostic shape gradients)
        loss_dice = self.soft_dice_loss(inputs, targets)

        # 3. Combine them! 
        # You might need to weight them (e.g., Focal is tiny, Dice is large, so maybe 20*Focal + Dice)
        return (10.0 * loss_focal) + loss_dice

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
    
    # def calculate_color_loss(self, src, tgt,
    #                         l_instance=1e1,
    #                         l_bg_mean=1e1,
    #                         l_bg_bright=1e-1,
    #                         l_fg_dark=1e0):
    #     src_unique = torch.unique(src, sorted=True)
    #     if len(src_unique) > 1:
    #         thresh = (src_unique[0] + src_unique[1]) / 2
    #     else:
    #         thresh = src_unique[0] + 1e-8
        
    #     # Ensure same device/dtype
    #     device = tgt.device
    #     dtype = tgt.dtype

    #     # If batched, we'll handle batch as single long vector but keep per-image separation
    #     if tgt.dim() == 2:
    #         B = 1
    #         src = src.unsqueeze(0)
    #         tgt = tgt.unsqueeze(0)
    #     else:
    #         B = tgt.size(0)

    #     # flatten per image
    #     src_flat = src.reshape(B, -1)
    #     tgt_flat = tgt.reshape(B, -1)

    #     total_instance_loss = torch.tensor(0.0, device=device, dtype=dtype)
    #     # We'll accumulate per-batch background terms and normalize by batch
    #     total_bg_mean_loss = torch.tensor(0.0, device=device, dtype=dtype)
    #     total_bg_bright_loss = torch.tensor(0.0, device=device, dtype=dtype)
    #     total_fg_dark_loss = torch.tensor(0.0, device=device, dtype=dtype)
        
    #     for i in range(B):
    #         s = src_flat[i]
    #         t = tgt_flat[i]

    #         # identify background label (min)
    #         bg_label = s.min()

    #         # ---------------- Instance loss (exclude background)
    #         unique_ids, inverse_idx = torch.unique(s, return_inverse=True)
    #         # mask of non-background
    #         mask_ids = unique_ids[unique_ids != bg_label]

    #         if mask_ids.numel() > 0:
    #             # restrict to pixels not background
    #             nonbg_mask = (s != bg_label)
    #             inv_nb = inverse_idx[nonbg_mask]         # indices into unique_ids
    #             t_nb = t[nonbg_mask]

    #             # remap inv_nb (which references positions in unique_ids) to 0..K-1
    #             _, remapped = torch.unique(inv_nb, return_inverse=True)
    #             K = mask_ids.numel()
    #             sums = torch.zeros(K, device=device, dtype=dtype)
    #             counts = torch.zeros(K, device=device, dtype=dtype)

    #             sums.scatter_add_(0, remapped, t_nb)
    #             counts.scatter_add_(0, remapped, torch.ones_like(t_nb, dtype=dtype))
    #             means = sums / counts.clamp_min(1.0)

    #             # mask_ids holds the target intensity for each instance (assumed already in same scale)
    #             target_vals = mask_ids.to(dtype)
    #             inst_loss = F.mse_loss(means, target_vals.to(means))

    #         else:
    #             inst_loss = torch.tensor(0.0, device=device, dtype=dtype)

    #         total_instance_loss += inst_loss

    #         # ---------------- Background mean + variance
    #         bg_mask = (s == bg_label)
    #         if bg_mask.any():
    #             t_bg = t[bg_mask]
    #             bg_mean = t_bg.mean()

    #             # mean loss (compare to bg_label)
    #             bg_mean_loss = F.mse_loss(bg_mean, bg_label.to(dtype))
    #             total_bg_mean_loss += bg_mean_loss
    #         else:
    #             # no background pixels (unlikely) -> zeros
    #             total_bg_mean_loss += torch.tensor(0.0, device=device, dtype=dtype)

    #         # Background over-threshold penalty
    #         bg_over = F.relu(t_bg - thresh)
    #         bg_over_loss = (bg_over ** 2).sum()

    #         # Foreground under-threshold penalty
    #         if mask_ids.numel() > 0:
    #             t_fg = t[nonbg_mask]
    #             fg_under = F.relu(thresh - t_fg)
    #             fg_under_loss = (fg_under ** 2).sum()
    #         else:
    #             fg_under_loss = torch.tensor(0.0, device=device, dtype=dtype)

    #         # Accumulate
    #         total_bg_bright_loss += bg_over_loss
    #         total_fg_dark_loss += fg_under_loss

    #     # average over batch
    #     self.loss_INSTMEAN = l_instance * total_instance_loss / float(B)
    #     self.loss_BGMEAN = l_bg_mean *  total_bg_mean_loss / float(B)
    #     self.loss_BGOVER = l_bg_bright * total_bg_bright_loss / float(B)
    #     self.loss_FGUNDER = l_fg_dark * total_fg_dark_loss / float(B)

    #     total_loss = (
    #         self.loss_INSTMEAN
    #         + self.loss_BGMEAN
    #         + self.loss_BGOVER
    #         + self.loss_FGUNDER
    #     )

    #     return total_loss
    
    def compute_dice_loss(self, preds, targets, smooth=1e-5):
        """
        Computes Dice Loss for binary (or continuous) masks.
        
        Args:
            preds: Tensor of shape (N, H, W) - strictly [0, 1]
            targets: Tensor of shape (N, H, W) - strictly [0, 1]
            smooth: A small constant to avoid division by zero.
        Returns:
            loss: Scalar mean Dice loss.
        """
        # Flatten the spatial dimensions
        preds_flat = preds.view(preds.shape[0], -1)
        targets_flat = targets.view(targets.shape[0], -1)
        
        # Compute intersection and sums
        intersection = (preds_flat * targets_flat).sum(dim=1)
        cardinality = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)
        
        # Compute Dice coefficient
        dice_score = (2.0 * intersection + smooth) / (cardinality + smooth)
        
        # Loss is 1 - Dice (we want to minimize the loss, maximize the score)
        dice_loss = 1.0 - dice_score
        
        return dice_loss.mean()
    
    def compute_sam_loss(self, fake, centroids, targets, n_fp, lambda_sam=1.):
        fake = (fake + 1.0) / 2.0
        
        rgb_transform = transforms.RGB()
        image = rgb_transform(fake)
        # img_tensor = image.detach()

        # 3. Manually resize to SAM's required 1024x1024 resolution
        img_1024 = F.interpolate(image, size=(1024, 1024), mode='bilinear', align_corners=False)

        # ==========================================
        # 5. Format Prompts (Force N independent masks)
        # ==========================================
        scaled_centroids = (centroids * 4.0).to(fake.device)

        # Reshape from (1, N, 2) -> (N, 1, 2)
        if scaled_centroids.dim() == 3 and scaled_centroids.shape[0] == 1:
            scaled_centroids = scaled_centroids.squeeze(0).unsqueeze(1)
        elif scaled_centroids.dim() == 2:
            scaled_centroids = scaled_centroids.unsqueeze(1)

        N = scaled_centroids.shape[0]

        # Create 1 label per point: (N, 1)
        point_labels = torch.ones((N, 1), dtype=torch.long, device=fake.device)

        # Package them as a tuple
        points = (scaled_centroids, point_labels)

        # ==========================================
        # THE DIFFERENTIABLE FORWARD PASS
        # ==========================================

        # A. Extract Image Features
        # Shape: (1, 256, 64, 64)
        image_embeddings = self.SAM.image_encoder(img_1024) 

        # B. Encode the Point Prompts
        # sparse batch size is N (21), dense batch size is N (21)
        sparse_embeddings, dense_embeddings = self.SAM.prompt_encoder(
            points=points,
            boxes=None,
            masks=None,
        )

        # C. Decode the Masks
        # SAM automatically broadcasts the 1 image to the N prompts!
        low_res_masks, iou_predictions = self.SAM.mask_decoder(
            image_embeddings=image_embeddings,  # <-- Pass the original tensor here!
            image_pe=self.SAM.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # low_res_masks is now exactly (N, 1, 256, 256)
        # Squeeze out the channel dim -> (N, 256, 256)
        logits = low_res_masks.squeeze(1)

        if targets.dim() == 4:
            targets = targets.squeeze(0)

        # print(sam_tensor_output.shape)
        targets = targets.to(logits.device).float()

        # logits_flat = torch.max(logits, dim=0).astype(float)
        # targets_flat = torch.any(targets, dim=0).astype(float)
        # loss = lambda_sam * sigmoid_focal_loss(
        #     inputs=logits_flat, 
        #     targets=targets_flat, 
        #     alpha=0.25, 
        #     gamma=2.0, 
        #     reduction='mean'
        # )

        tp_weight = 1
        fp_weight = 0
        if 0 < n_fp < targets.shape[0]:
            self.loss_tp_SAM = self.dice_focal_loss(
                inputs=logits[:-n_fp], 
                targets=targets[:-n_fp], 
                alpha=0.25, 
                gamma=2.0, 
                reduction='mean'
            )
            self.loss_fp_SAM = self.dice_focal_loss(
                inputs=logits[n_fp:], 
                targets=targets[n_fp:], 
                alpha=0.25, 
                gamma=2.0, 
                reduction='mean'
            )
        elif n_fp == 0:
            self.loss_tp_SAM = self.dice_focal_loss(
                inputs=logits, 
                targets=targets, 
                alpha=0.25, 
                gamma=2.0, 
                reduction='mean'
            )
            self.loss_fp_SAM = 0
        else:
            self.loss_tp_SAM = 0
            self.loss_fp_SAM = self.dice_focal_loss(
                inputs=logits, 
                targets=targets, 
                alpha=0.25, 
                gamma=2.0, 
                reduction='mean'
            )   
      
        loss = lambda_sam * (tp_weight * self.loss_tp_SAM + fp_weight * self.loss_fp_SAM) #/ (tp_weight + fp_weight)
        
        return loss

    def get_fp_centroids(self, fake, targets):
        fake = (fake + 1) / 2
        targets = targets.to(fake.device)
        flat_targets = torch.any(targets, dim=1)
        masked_fake = torch.where(flat_targets, 0, fake)

        anomalies, z_scores = self.get_contrast_anomalies(masked_fake, z_threshold=2.)
        clean_anomalies = self.binary_erosion(anomalies, kernel_size=11)
        fp_centroids = self.get_blob_centroids(clean_anomalies)

        # from torchvision.utils import save_image
        # save_image(masked_fake, "./_masked_fake.png")
        # save_image(fake, "./_fake.png")
        # save_image(z_scores.float(), "./_anomalies.png")
        # save_image(clean_anomalies.float(), "./_clean_anomalies.png")

        # print(fp_centroids)

        return fp_centroids      

    def get_contrast_anomalies(self, masked_image, z_threshold=2.0):
        local_mean = torch.mean(masked_image)
        local_mean_sq = local_mean**2
        local_var = torch.clamp(local_mean_sq - local_mean**2, min=1e-6)
        local_std = torch.sqrt(local_var)
        
        # 3. Compute Z-Score Map (Contrastive Map)
        z_scores = (masked_image - local_mean) / local_std
        
        # 4. Threshold to find anomalous bright blobs (False Positives)
        # A z_threshold of 2.0 means "is this pixel 2 standard deviations brighter than its neighborhood?"
        anomalies = z_scores > z_threshold

        # if torch.count_nonzero(anomalies) == torch.count_nonzero(masked_image):
        #     # Everything was seen as anomaly
        #     return torch.zeros_like(anomalies), torch.zeros_like(z_scores)
        
        # Optional: Clean up the edges around your pitch-black masked holes
        # (Because the black holes drag down the local mean, the edges might falsely spike)
        anomalies = anomalies & (masked_image > 0.05) 
        
        return anomalies, z_scores

    def binary_erosion(self, binary_mask, kernel_size=3):
        """
        Simulates a disk(1) erosion using the Max Pool Inversion trick.
        """
        mask_float = binary_mask.float()
        
        # 1. Invert mask
        # 2. Dilate (Max Pool)
        # 3. Invert back
        eroded_float = 1.0 - F.max_pool2d(
            1.0 - mask_float, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=kernel_size // 2
        )
        
        return eroded_float.bool()
    
    def get_blob_centroids(self, binary_mask):
        """
        Extracts the exact geometric centroid of every isolated blob in a binary mask.
        Pure PyTorch, fully on GPU, zero hyperparameters.
        """
        # binary_mask shape should be (1, 1, H, W)
        B, C, H, W = binary_mask.shape
        device = binary_mask.device

        # ==========================================
        # 1. Initialize Unique IDs
        # ==========================================
        # Create a tensor where every pixel has its flattened index + 1 as its ID.
        # (We add 1 so that index 0 is strictly reserved for the background).
        indices = torch.arange(1, H * W + 1, device=device, dtype=torch.float32).view(1, 1, H, W)

        # Apply the mask: foreground gets unique IDs, background stays 0
        labels = indices * binary_mask.float()

        # ==========================================
        # 2. Flood Fill (Iterative Max Pooling)
        # ==========================================
        # The highest ID within each blob will continuously spread until it fills the whole blob.
        while True:
            # Spread the IDs
            new_labels = F.max_pool2d(labels, kernel_size=3, stride=1, padding=1)
            
            # Multiply by the binary mask to stop IDs from bleeding into the background
            new_labels = new_labels * binary_mask.float()

            # If the image hasn't changed, every blob is fully flooded!
            if torch.equal(labels, new_labels):
                break
            labels = new_labels

        # ==========================================
        # 3. Extract the Unique Blobs
        # ==========================================
        unique_labels = torch.unique(labels)
        unique_labels = unique_labels[unique_labels > 0] # Remove the background 0

        if len(unique_labels) == 0:
            return torch.empty((1, 0, 2), device=device)

        # ==========================================
        # 4. Calculate Exact Geometric Centroids
        # ==========================================
        # Create grid coordinates for the whole image
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )

        centroids = []
        labels_2d = labels.squeeze() # Flatten to (H, W)

        # Loop through each isolated blob and calculate its center
        for label in unique_labels:
            # Get a boolean mask of JUST this specific blob
            blob_pixels = (labels_2d == label)

            # Average the x and y coordinates of those pixels
            y_c = y_grid[blob_pixels].float().mean()
            x_c = x_grid[blob_pixels].float().mean()

            centroids.append(torch.stack([x_c, y_c]))

        # Stack into a final (N, 2) tensor formatted for SAM!
        fp_centroids = torch.stack(centroids)[None, ...]

        return fp_centroids

    def print_receptive_field(self, network, name="Discriminator"):
        # """
        # Dynamically calculates and prints the receptive field of a sequential CNN.
        # """
        # receptive_field = 1
        # stride_product = 1

        # # Loop through all modules in the network
        # for module in network.modules():
        #     if isinstance(module, nn.Conv2d):
        #         # Assuming square kernels and strides (e.g., 4x4 kernel)
        #         k = module.kernel_size[0]
        #         s = module.stride[0]
                
        #         receptive_field += (k - 1) * stride_product
        #         stride_product *= s
                
        # print(f"{name} Receptive Field: {receptive_field} x {receptive_field} pixels")
        # # from torchscan import summary
        # # summary(network, (1, 256, 256), receptive_field=True, max_depth=0)

    # def get_true_receptive_field(network, name="Discriminator"):
        """
        Dynamically computes the receptive field by inspecting the actual 
        kernel sizes and strides of convolutions and pooling layers.
        Assumes no parameters and tracks Height and Width independently.
        """
        # [Height, Width]
        rf = [1, 1]   
        jump = [1, 1] 

        for module in network.modules():
            # Look for any layer that alters the spatial field of view
            if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
                
                # Extract kernel size (safely handle int vs tuple)
                k = module.kernel_size
                k = (k, k) if isinstance(k, int) else k
                
                # Extract stride (safely handle int vs tuple vs None)
                s = module.stride
                s = (s, s) if isinstance(s, int) else s
                if s is None or s == (): 
                    s = k if isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)) else (1, 1)
                
                # 1. Update Receptive Field: R = R + (Kernel - 1) * Jump
                rf[0] += (k[0] - 1) * jump[0]
                rf[1] += (k[1] - 1) * jump[1]
                
                # 2. Update Jump: Jump = Jump * Stride
                jump[0] *= s[0]
                jump[1] *= s[1]

        print(f"{name} Receptive Field: {rf[0]} x {rf[1]} pixels")
        return rf