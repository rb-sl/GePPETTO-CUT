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
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE', "SAM"]
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            
            # self.SAM = SAM("sam2_b.pt")
            self.SAM = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
            self.SAM.to(self.device)
            for param in self.SAM.parameters():
                param.requires_grad = False
            self.SAM.eval()
            

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
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
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
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self, epoch):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
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

        # TODO assuming batch = 1

        # with torch.no_grad():
            # rgb_transform = transforms.RGB()
            # image = rgb_transform(self.fake_B) #.contiguous()
        #     img_np = image[0].permute(1, 2, 0).cpu().detach().numpy().copy()
        #     print(image.shape)
        #     # self.loss_COLOR = self.calculate_color_loss(self.real_A, self.fake_B)
        #     sam_output = self.SAM(img_np, points=self.centroids_A)

        # with torch.no_grad():
        # 2. Check your normalization!
        # GANs often output [-1, 1]. If you pass a tensor directly, SAM expects [0, 1].
        # (If your transforms.RGB() already outputs [0, 1], you can remove this block)

        # if fake.min() < 0:
        # fake = (fake + 1.0) / 2.0
        
        # rgb_transform = transforms.RGB()
        # image = rgb_transform(fake)
        # # img_tensor = image.detach()

        # # 3. Manually resize to SAM's required 1024x1024 resolution
        # img_1024 = F.interpolate(image, size=(1024, 1024), mode='bilinear', align_corners=False)

        # # ==========================================
        # # 5. Format Prompts (Force N independent masks)
        # # ==========================================
        # scaled_centroids = (self.centroids_A * 4.0).to(fake.device)

        # # Reshape from (1, N, 2) -> (N, 1, 2)
        # if scaled_centroids.dim() == 3 and scaled_centroids.shape[0] == 1:
        #     scaled_centroids = scaled_centroids.squeeze(0).unsqueeze(1)
        # elif scaled_centroids.dim() == 2:
        #     scaled_centroids = scaled_centroids.unsqueeze(1)

        # N = scaled_centroids.shape[0]

        # # Create 1 label per point: (N, 1)
        # point_labels = torch.ones((N, 1), dtype=torch.long, device=fake.device)

        # # Package them as a tuple
        # points = (scaled_centroids, point_labels)

        # # ==========================================
        # # THE DIFFERENTIABLE FORWARD PASS
        # # ==========================================

        # # A. Extract Image Features
        # # Shape: (1, 256, 64, 64)
        # image_embeddings = self.SAM.image_encoder(img_1024) 

        # # B. Encode the Point Prompts
        # # sparse batch size is N (21), dense batch size is N (21)
        # sparse_embeddings, dense_embeddings = self.SAM.prompt_encoder(
        #     points=points,
        #     boxes=None,
        #     masks=None,
        # )

        # # C. Decode the Masks
        # # SAM automatically broadcasts the 1 image to the N prompts!
        # low_res_masks, iou_predictions = self.SAM.mask_decoder(
        #     image_embeddings=image_embeddings,  # <-- Pass the original tensor here!
        #     image_pe=self.SAM.prompt_encoder.get_dense_pe(),
        #     sparse_prompt_embeddings=sparse_embeddings,
        #     dense_prompt_embeddings=dense_embeddings,
        #     multimask_output=False,
        # )

        # # low_res_masks is now exactly (N, 1, 256, 256)
        # # Squeeze out the channel dim -> (N, 256, 256)
        # logits = low_res_masks.squeeze(1)

        # targets = self.exploded_A
        # if targets.dim() == 4:
        #     targets = targets.squeeze(0)

        # # # 4. Scale your prompts! 
        # # # Since we scaled the image from 256 to 1024 (exactly 4x), the centroids must move too.
        # # scaled_centroids = (self.centroids_A * 4.0).to(fake.device)

        # # # Guarantee it is exactly 3D: (Batch, Num_Points, 2)
        # # # Reshape from (1, N, 2) -> (N, 1, 2)
        # # if scaled_centroids.dim() == 3 and scaled_centroids.shape[0] == 1:
        # #     scaled_centroids = scaled_centroids.squeeze(0).unsqueeze(1)
        # # elif scaled_centroids.dim() == 2:
        # #     scaled_centroids = scaled_centroids.unsqueeze(1)

        # # # Extract the batch size (B) and number of points (N)
        # # N = scaled_centroids.shape[0]

        # # # Create a label (1 = Foreground) for EVERY single point in the batch
        # # # Shape must be exactly (B, N) -> e.g., (1, 21)
        # # point_labels = torch.ones((N, 1), dtype=torch.long, device=fake.device)

        # # # Package them as a tuple. NO extra lists or unsqueezes here!
        # # points = (scaled_centroids, point_labels)

        # # # A. Extract Image Features
        # # image_embeddings = self.SAM.image_encoder(img_1024)

        # # # B. Encode the Point Prompts
        # # sparse_embeddings, dense_embeddings = self.SAM.prompt_encoder(
        # #     points=points,
        # #     boxes=None,
        # #     masks=None,
        # # )

        # # image_embeddings_expanded = image_embeddings.repeat(N, 1, 1, 1)

        # # # C. Decode the Masks (Output is continuous logits!)
        # # low_res_masks, iou_predictions = self.SAM.mask_decoder(
        # #     image_embeddings=image_embeddings_expanded,
        # #     image_pe=self.SAM.prompt_encoder.get_dense_pe(),
        # #     sparse_prompt_embeddings=sparse_embeddings,
        # #     dense_prompt_embeddings=dense_embeddings,
        # #     multimask_output=False,
        # # )

        # # # low_res_masks shape is (1, 1, 256, 256). These are your RAW LOGITS!
        # # # Squeeze out the batch and channel dims -> (256, 256) or (N, 256, 256)
        # # logits = low_res_masks.squeeze(1)
 
        # # print(sam_output[0].masks.data.shape, torch.zeros_like(sam_output[0].masks.data).shape)

        # # sam_tensor_output = sam_output[0].masks.data #np.array([x.masks.data for x in sam_output[0]]).astype(bool)

        # # if sam_tensor_output.shape[0] > 0:
        # #     if sam_tensor_output.dim() == 3:
        # #         sam_tensor_output = sam_tensor_output.unsqueeze(0)

        # #     # Resize back down to original GAN resolution
        # #     preds_resized = F.interpolate(sam_tensor_output.float(), size=(256, 256), 
        # #                                   mode='bilinear', align_corners=False) > 0.5

        # #     # Squeeze the fake batch dimension back out -> (N, 256, 256)
        # #     sam_tensor_output = preds_resized.squeeze(0)

        # # print(sam_tensor_output.shape)
        # self.exploded_A = self.exploded_A.to(logits.device)[0]

        # lambda_sam = 10.
        # # self.loss_SAM = self.compute_sam_loss(logits, self.exploded_A, lambda_sam=0.1)
        # self.loss_SAM = lambda_sam * sigmoid_focal_loss(
        #     inputs=logits, 
        #     targets=self.exploded_A.float(), 
        #     alpha=0.25, 
        #     gamma=2.0, 
        #     reduction='mean'
        # )
        if epoch > 10:
            fp_centroids = self.get_fp_centroids(fake, targets=self.exploded_A)

            if fp_centroids.shape[1] > 0:
                # Append false positive centroids to real centroids
                all_centroids = torch.cat([self.centroids_A.to(fake.device), fp_centroids], dim=1)
                # Append fake targets (all zeros) to match the FP centroids
                all_targets = torch.cat([self.exploded_A, torch.zeros((1, fp_centroids.shape[1], *self.exploded_A.shape[2:]))], dim=1)  # batch_size, 
            else:
                all_centroids = self.centroids_A.to(fake.device)
                all_targets = self.exploded_A
                
            self.loss_SAM = self.compute_sam_loss(fake, centroids=all_centroids, targets=all_targets, n_fp=fp_centroids.shape[1], lambda_sam=10)
        else:
            self.loss_SAM = 0
        
        self.loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_SAM #+ self.loss_COLOR
        
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
    
    # def compute_iou_matrix(self, masks1, masks2):
    #     """
    #     Computes pairwise IoU between two sets of masks efficiently using matrix multiplication.
        
    #     Args:
    #         masks1: Tensor of shape (N, H, W)
    #         masks2: Tensor of shape (M, H, W)
    #     Returns:
    #         iou: Tensor of shape (N, M)
    #     """

    #     print(">>>", masks1.shape, masks2.shape)

    #     if masks1.shape[0] == 0 or masks2.shape[0] == 0:
    #         return torch.zeros((masks1.shape[0], masks2.shape[0]), device=masks1.device)

    #     # Flatten spatial dimensions: (N, H*W) and (M, H*W)
    #     m1 = masks1.view(masks1.shape[0], -1).float()
    #     m2 = masks2.view(masks2.shape[0], -1).float()



    #     # Intersection: Matrix multiplication of m1 and m2 transposed -> (N, M)
    #     intersection = torch.mm(m1, m2.t())
        
    #     # Areas: Sum along the spatial dimension
    #     area1 = m1.sum(dim=1).unsqueeze(1) # (N, 1)
    #     area2 = m2.sum(dim=1).unsqueeze(0) # (1, M)

    #     # Union = Area1 + Area2 - Intersection
    #     union = area1 + area2 - intersection
        
    #     # Add a small epsilon to prevent division by zero
    #     iou = intersection / torch.clamp(union, min=1e-8)
    #     return iou

    def compute_iou_matrix(self, pred_logits, targets, iou_threshold=0.1):
        # """
        # Computes pairwise IoU between two sets of masks efficiently.
        # Automatically handles rogue batch dimensions.
        # """
        # # 1. Force both tensors into strict 3D shapes: (Number_of_Masks, H, W)
        # # This turns (1, 10, 256, 256) into (10, 256, 256) automatically
        # masks1_3d = masks1.view(-1, masks1.shape[-2], masks1.shape[-1])
        # masks2_3d = masks2.view(-1, masks2.shape[-2], masks2.shape[-1])

        # N = masks1_3d.shape[0]
        # M = masks2_3d.shape[0]

        # if N == 0 or M == 0:
        #     return torch.zeros((N, M), device=masks1.device)

        # # 2. Flatten spatial dimensions: (N, H*W) and (M, H*W)
        # # m1 = masks1_3d.view(N, -1).float()
        # with torch.no_grad():
        #     m1 = (masks1_3d > 0.0).float().view(N, -1)
        # m2 = masks2_3d.view(M, -1).float()

        # # 3. Intersection: Matrix multiplication of m1 and m2 transposed -> (N, M)
        # intersection = torch.mm(m1, m2.t())
        
        # # 4. Areas: Sum along the spatial dimension
        # area1 = m1.sum(dim=1).unsqueeze(1) # (N, 1)
        # area2 = m2.sum(dim=1).unsqueeze(0) # (1, M)

        # # 5. Union = Area1 + Area2 - Intersection
        # union = area1 + area2 - intersection
        
        # # 6. Add a small epsilon to prevent division by zero
        # iou = intersection / torch.clamp(union, min=1e-8)
        
        # return iou
        """
        Globally optimal assignment using the Hungarian Algorithm.
        Symmetrically penalizes unmatched predictions and targets.
        """
        device = pred_logits.device
        N, H, W = pred_logits.shape
        M = targets.shape[0]

        # --- Edge Cases ---
        if N == 0 and M == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        if N == 0:
            return sigmoid_focal_loss(torch.zeros_like(targets), targets, reduction='mean')
        if M == 0:
            return sigmoid_focal_loss(pred_logits, torch.zeros_like(pred_logits), reduction='mean')

        # 1. Compute IoU Matrix
        iou_matrix = self.compute_iou_matrix(pred_logits, targets) 
        
        # 2. Hungarian Assignment (SciPy requires numpy, CPU)
        # We want to maximize IoU, so we pass negative IoU as the "cost"
        cost_matrix = -iou_matrix.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # 3. Filter out terrible matches using the threshold
        match_p = []
        match_t = []
        for p_idx, t_idx in zip(row_ind, col_ind):
            if iou_matrix[p_idx, t_idx] >= iou_threshold:
                match_p.append(p_idx)
                match_t.append(t_idx)

        # 4. Figure out total evaluations (N + M - matches)
        num_matches = len(match_p)
        total_evals = N + M - num_matches
        
        final_preds = torch.zeros((total_evals, H, W), dtype=pred_logits.dtype, device=device)
        final_targets = torch.zeros((total_evals, H, W), dtype=targets.dtype, device=device)

        # Tracking sets to find unmatched indices
        matched_p_set = set(match_p)
        matched_t_set = set(match_t)

        current_idx = 0

        # A. Insert Matches
        if num_matches > 0:
            final_preds[:num_matches] = pred_logits[match_p]
            final_targets[:num_matches] = targets[match_t]
            current_idx = num_matches

        # B. Insert Unmatched Predictions (False Positives -> penalized vs Zeros)
        unmatched_p = [i for i in range(N) if i not in matched_p_set]
        if len(unmatched_p) > 0:
            end_idx = current_idx + len(unmatched_p)
            final_preds[current_idx:end_idx] = pred_logits[unmatched_p]
            current_idx = end_idx

        # C. Insert Unmatched Targets (False Negatives -> penalized vs Zeros)
        # Notice final_preds remains 0.0 (Logit zero = 50% prob). 
        # To truly penalize it as a hard negative, we should push the logit very low (e.g., -10)
        unmatched_t = [i for i in range(M) if i not in matched_t_set]
        if len(unmatched_t) > 0:
            end_idx = current_idx + len(unmatched_t)
            # Deep negative logit = 0% probability prediction
            final_preds[current_idx:end_idx] = -10.0 
            final_targets[current_idx:end_idx] = targets[unmatched_t]

        # 5. Compute the final vectorized loss
        # Using Focal Loss because of the extreme class imbalance in segmentation
        loss = sigmoid_focal_loss(final_preds, final_targets, alpha=0.25, gamma=2.0, reduction='mean')

        return loss
    
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
        fp_weight = 1
        if 0 < n_fp < targets.shape[0]:
            sam_loss_tp = self.dice_focal_loss(
                inputs=logits[:-n_fp], 
                targets=targets[:-n_fp], 
                alpha=0.25, 
                gamma=2.0, 
                reduction='mean'
            )
            sam_loss_fp = self.dice_focal_loss(
                inputs=logits[n_fp:], 
                targets=targets[n_fp:], 
                alpha=0.25, 
                gamma=2.0, 
                reduction='mean'
            )
        elif n_fp == 0:
            sam_loss_tp = self.dice_focal_loss(
                inputs=logits, 
                targets=targets, 
                alpha=0.25, 
                gamma=2.0, 
                reduction='mean'
            )
            sam_loss_fp = 0
        else:
            sam_loss_tp = 0
            sam_loss_fp = self.dice_focal_loss(
                inputs=logits, 
                targets=targets, 
                alpha=0.25, 
                gamma=2.0, 
                reduction='mean'
            )   
      
        loss = lambda_sam * (tp_weight * sam_loss_tp + fp_weight * sam_loss_fp) #/ (tp_weight + fp_weight)
        
        return loss


    # def compute_sam_loss(self, preds, targets, iou_threshold=0., lambda_sam=1.):
        # """
        # Symmetric Greedy Assignment.
        # Penalizes both False Positives (unmatched preds) and False Negatives (unmatched targets)
        # by evaluating exactly (N + M - matches) elements.
        # """
        # device = preds.device
        # N, H, W = preds.shape
        # M = targets.shape[0]

        # # --- Fast-path Edge Cases ---
        # if N == 0 and M == 0:
        #     return torch.tensor(0.0, device=device, requires_grad=True)
        # if N == 0:
        #     # 0 preds, M targets -> All False Negatives
        #     return self.compute_dice_loss(torch.zeros_like(targets), targets)
        # if M == 0:
        #     # N preds, 0 targets -> All False Positives
        #     return self.compute_dice_loss(preds, torch.zeros_like(preds))

        # # 1. Compute IoU and Sort
        # iou_matrix = self.compute_iou_matrix(preds, targets) 
        # flat_iou = iou_matrix.view(-1)
        # sorted_iou, sorted_indices = torch.sort(flat_iou, descending=True)

        # pred_assigned = torch.zeros(N, dtype=torch.bool, device=device)
        # target_assigned = torch.zeros(M, dtype=torch.bool, device=device)

        # pred_indices = sorted_indices // M
        # target_indices = sorted_indices % M

        # match_p = []
        # match_t = []

        # # 2. Greedy Assignment
        # for p_idx, t_idx, iou in zip(pred_indices, target_indices, sorted_iou):
        #     if iou < iou_threshold:
        #         break 
                
        #     p = p_idx.item()
        #     t = t_idx.item()
            
        #     if not pred_assigned[p] and not target_assigned[t]:
        #         match_p.append(p)
        #         match_t.append(t)
        #         pred_assigned[p] = True
        #         target_assigned[t] = True

        # # 3. Pre-allocate aligned tensors based on total required evaluations
        # num_matches = len(match_p)
        # total_evals = N + M - num_matches
        
        # # Initialize with zeros. Unmatched items will default to comparing against these zeros!
        # final_preds = torch.zeros((total_evals, H, W), dtype=preds.dtype, device=device)
        # final_targets = torch.zeros((total_evals, H, W), dtype=targets.dtype, device=device)

        # current_idx = 0

        # # 4. Insert True Matches
        # if num_matches > 0:
        #     final_preds[:num_matches] = preds[match_p]
        #     final_targets[:num_matches] = targets[match_t]
        #     current_idx = num_matches

        # # 5. Insert Unmatched Predictions (False Positives)
        # # Their counterpart in final_targets remains zero.
        # unmatched_p = torch.where(~pred_assigned)[0]
        # if len(unmatched_p) > 0:
        #     end_idx = current_idx + len(unmatched_p)
        #     final_preds[current_idx:end_idx] = preds[unmatched_p]
        #     current_idx = end_idx

        # # 6. Insert Unmatched Targets (False Negatives)
        # # Their counterpart in final_preds remains zero.
        # unmatched_t = torch.where(~target_assigned)[0]
        # if len(unmatched_t) > 0:
        #     end_idx = current_idx + len(unmatched_t)
        #     final_targets[current_idx:end_idx] = targets[unmatched_t]

        # # 7. Compute a single, highly optimized Vectorized BCE Loss over all cases
        # loss = self.compute_dice_loss(final_preds, final_targets)

        # return lambda_sam * loss

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