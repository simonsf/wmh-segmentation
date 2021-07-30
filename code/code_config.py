from easydict import EasyDict as edict
from md_segmentation3d.utils.vseg_helpers import AdaptiveNormalizer, FixedNormalizer
import numpy as np

__C = edict()
cfg = __C


##################################
# general parameters
##################################

__C.general = {}

# image-segmentation pair list
# 1) single-modality image training, use txt annotation file
# 2) multi-modality image training, use csv annotation file
__C.general.imseg_list = "data_train2.csv"

# the output of training models and logs
__C.general.save_dir = 'wmh_2d_5'

# continue training from certain epoch, -1 to train from scratch
__C.general.resume_epoch = -1

# when finetune from certain model, can choose clear start epoch idx
__C.general.clear_start_epoch = False

# the number of GPUs used in training
__C.general.num_gpus = 1

# random seed used in training (debugging purpose)
__C.general.seed = 1


##################################
# data set parameters
##################################

__C.dataset = {}

# the number of classes
__C.dataset.num_classes = 2

# the resolution on which segmentation is performed
__C.dataset.spacing = [0.5, 0.5]

# the sampling crop size, e.g., determine the context information
__C.dataset.crop_size = [256, 256]


# the re-sample padding type (0 for zero-padding, 1 for edge-padding)
__C.dataset.pad_t = 0

# the default padding value list
__C.dataset.default_values = [-1]

# sampling method:
# 1) GLOBAL: sampling crops randomly in the entire image domain
# 2) MASK: sampling crops randomly within segmentation mask
# 3) CENTER: sampling crops in the center of the image
# 4) MIX: sampling crops with mixed global and mask method
# 5) RANDOM: sampling crops with randomly chosen methods
__C.dataset.sampling_method = 'RANDOM'

# translation augmentation (unit: mm)
__C.dataset.random_translation = [5, 5]

# interpolation method:
# 1) NN: nearest neighbor interpolation
# 2) LINEAR: linear interpolation
__C.dataset.interpolation = 'NN'

# crop intensity normalizers (to [-1,1])
# one normalizer corresponds to one input modality
# 1) FixedNormalizer: use fixed mean and standard deviation to normalize intensity
# 2) AdaptiveNormalizer: use minimum and maximum intensity of crop to normalize intensity
__C.dataset.crop_normalizers = [{'modality':'MR','min_p':0.01,'max_p':0.9999}]


####################################
# training loss
####################################

__C.loss = {}

# the name of loss function to use
# Focal: Focal loss, supports binary-class and multi-class segmentation
# Dice: Dice Similarity Loss, supports binary-class and multi-class segmentation
# Boundary: Dynamic boundary-weighted soft Dice
# WeightedDice: Soft Dice with higher weights on detailed (isolated) parts
# SSIM: Multi-scale SSIM
__C.loss.name = ['Dice', 'Focal']

# weights for each loss; should match the number of losses
# weights will NOT be normalized
__C.loss.loss_weight = [1, 1]

# the weight for each class including background class
# weights will be normalized
__C.loss.obj_weight = [1, 1]

# the gamma parameter in focal loss
__C.loss.focal_gamma = 2


#####################################
# net
#####################################

__C.net = {}

# the network name
# 1) 2D_net: 5-layer 2D VBNet
# 2) 25D_net: 4-layer 2.5D VBNet 
__C.net.name = 'whm_vbnet'


######################################
# training parameters
######################################

__C.train = {}

# the number of training epochs
__C.train.epochs = 4001

# the number of samples in a batch
__C.train.batchsize = 64

# the number of threads for IO
__C.train.num_threads = 16

# the learning rate
__C.train.lr = 1e-4

##### 方法 CosineAnnealing 参数 T_max,eta_min,last_epoch
##### 方法 Step            参数 step_size, gamma, last_epoch
##### 方法 MultiStep       参数 milestones, gamma, last_epoch
##### 方法 Exponential     参数 gamma, last_epoch
##### last_epoch如果没有设置或者设置为-1，last_epoch将被设置为__C.general.resume_epoch
##### 方法还有很多，自己pytorch查询
__C.train.lr_scheduler = {}
__C.train.lr_scheduler.name = "Step"
__C.train.lr_scheduler.params = {"step_size": 500, "gamma": 0.1, "last_epoch": -1}

##### 方法 Adam           参数 betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False
##### 方法 SGD            参数 momentum=0, dampening=0, weight_decay=0, nesterov=False
##### 方法还有很多，自己pytorch查询
__C.train.optimizer = {}
__C.train.optimizer.name = "Adam"
__C.train.optimizer.params = {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01, "amsgrad": False}

# the number of batches to update loss curve
__C.train.plot_snapshot = 10

# the number of batches to save model
__C.train.save_epochs = 100


########################################
# debug parameters
########################################

__C.debug = {}

# whether to save input crops
__C.debug.save_inputs = True


