from __future__ import print_function
import os
import numpy as np
import time
import argparse
import md
import os
import numpy as np
import importlib
import codecs
from easydict import EasyDict as edict
import torch
import torch.nn as nn
from code_whm_utils import read_test_txt, read_test_csv, read_test_folder, load_pytorch_model, last_checkpoint
from code_whm_dataset import fix_normalizers, adaptive_normalizers, resize_image_itk
from code_whm_network import SegmentationNet
import SimpleITK as sitk
import copy


class VSegModel2D(object):
    """ segmentation model at one resolution """

    def __init__(self):
        """ constructor """
        # network parameters
        self.net = None
        # image spacing
        self.spacing = np.array([1, 1, 1])
        # maximum stride of network
        self.max_stride = 1
        # image crop normalizers
        self.crop_normalizers = None
        # network output channels
        self.out_channels = 2
        # network input channels
        self.in_channels = 1
        # interpolation method
        self.interpolation = 'NN'
        # default paddding value list
        self.default_values = np.array([0], dtype=np.float)
        # name of network type
        self.network_type = None
        # sample method
        self.cropping_method = None
        # crop voxel size
        self.crop_voxel_size = np.array([0, 0, 0], dtype=np.int)
        # box percent padding
        self.box_percent_padding = 0.0
        # testing level
        self.level = None

    def load(self, model_dir):
        """ load python segmentation model from folder
        :param model_dir: model directory
        :return: None
        """
        if not os.path.isdir(model_dir):
            raise ValueError('model dir not found: {}'.format(model_dir))

        checkpoint_dir = last_checkpoint(os.path.join(model_dir, 'checkpoints'))
        param_file = os.path.join(checkpoint_dir, 'params.pth')

        if not os.path.isfile(param_file):
            raise ValueError('param file not found: {}'.format(param_file))

        # load network parameters
        state = load_pytorch_model(param_file)
        self.spacing = np.array(state['spacing'], dtype=np.double)
        assert self.spacing.ndim == 1, 'spacing must be 3-dim array'

        self.max_stride = state['max_stride']

        self.crop_normalizers = []
        if 'crop_normalizers' in state:
            for crop_normalizer in state['crop_normalizers']:
                self.crop_normalizers.append(self.__normalizer_from_dict(crop_normalizer))
        elif 'crop_normalizer' in state:
            self.crop_normalizers.append(self.__normalizer_from_dict(state['crop_normalizer']))
        else:
            raise ValueError('crop_normalizers not found in checkpoint')

        if 'default_values' in state:
            self.default_values = np.array(state['default_values'], dtype=np.double)
        else:
            self.default_values = np.array([0], dtype=np.double)

        self.out_channels = 2
        if 'out_channels' in state:
            self.out_channels = state['out_channels']

        self.in_channels = 1
        if 'in_channels' in state:
            self.in_channels = state['in_channels']

        self.interpolation = 'NN'
        if 'interpolation' in state:
            assert self.interpolation in ('NN', 'LINEAR', 'FILTER_NN'), '[Model] Invalid Interpolation'
            self.interpolation = state['interpolation']

        if 'cropping_method' in state:
            self.cropping_method = state['cropping_method']
            if self.cropping_method == 'fixed_box':
                self.crop_voxel_size = state['crop_voxel_size']
                self.box_percent_padding = state['box_percent_padding']
        else:
            self.cropping_method = 'fixed_spacing'
        assert self.cropping_method in ['fixed_spacing', 'fixed_box'], 'invalid cropping method'


        self.net = SegmentationNet(self.in_channels, self.out_channels)

        self.net = nn.parallel.DataParallel(self.net)
        self.net = self.net.cuda()
        self.net.load_state_dict(state['state_dict'])
        self.net.eval()

    @staticmethod
    def __normalizer_from_dict(crop_normalizer):
        """ convert dictionary to crop normalizer """

        if crop_normalizer['type'] == 0:
            params = {}
            params['mean'] = crop_normalizer['mean']
            params['stddev'] = crop_normalizer['stddev']
            params['clip'] = crop_normalizer['clip']
            ret = fix_normalizers
        elif crop_normalizer['type'] == 1:
            params = {}
            params['min'] = crop_normalizer['min_p']
            params['max'] = crop_normalizer['max_p']
            params['clip'] = crop_normalizer['clip']
            ret = adaptive_normalizers
        else:
            raise ValueError('unknown normalizer type: {}'.format(crop_normalizer['type']))
        return [ret, params]


def prepare_image_fixed_spacing(images, model):
    ori_spacing = images[0].GetSpacing()
    if len(model.spacing) == 2:
        spacing = np.append(model.spacing, ori_spacing[2])
    else:
        spacing = model.spacing
        spacing[2] = ori_spacing[2]
   
    prev_size = images[0].GetSize()

    box_size = (np.array(images[0].GetSize()) * ori_spacing / spacing + 0.5).astype(np.int32)

    for i in range(2):
        box_size[i] = int(np.ceil(box_size[i] * 1.0 / model.max_stride[i]) * model.max_stride[i])

    method = model.interpolation

    assert method in ('NN', 'LINEAR', 'FILTER_NN')

    resample_images = []
    iso_images = []
    for idx, image in enumerate(images):
        ret, params = model.crop_normalizers[idx]
        params['image'] = sitk.GetArrayFromImage(image)
        norm_data = ret(**params)

        image_origin =  image.GetOrigin()
        image_spacing = image.GetSpacing()
        image_direction = image.GetDirection()

        image = sitk.GetImageFromArray(norm_data)
        image.SetOrigin(image_origin)
        image.SetSpacing(image_spacing)
        image.SetDirection(image_direction)

        if method == 'NN':
            resample_image = resize_image_itk(image, box_size.tolist(), spacing.tolist())
        elif method == 'LINEAR':
            resample_image = resize_image_itk(image, box_size.tolist(), spacing.tolist(), resamplemethod=sitk.sitkLinear)
        resample_images.append(resample_image)

        resample_data = sitk.GetArrayFromImage(resample_image)
        iso_images.append(resample_data)      
  
    iso_image_tensor = torch.from_numpy(np.array(iso_images)).unsqueeze(0)

    iso_batch = []
    for i in range(iso_image_tensor.shape[2]):
        tmp = iso_image_tensor[:, :, i, :, :].to(torch.device('cuda'))
        iso_batch.append(tmp)

    return iso_batch, images[0], resample_images[0]


def load_model(folder, gpu_id=0):
    """ load segmentation model from folder
        :param folder:          the folder that contains segmentation model
        :param gpu_id:          which gpu to run segmentation model
        :return: a segmentation model
        """
    model = edict()
    model_name = os.path.basename(folder)
    model[model_name] = VSegModel2D()
    model[model_name].load(folder)
    model.num_labels = model[model_name].out_channels
    model.gpu_id = gpu_id

    return model


def network_output(iso_batch, model, pre_image, resample_image):
    probs=[]
    with torch.no_grad():
        for patch in iso_batch:
            prob = model.net(patch)
            probs.append(prob)

    prob = torch.cat(probs, 0)

    _, mask = prob.max(1)
    mask = mask.short()

    mask = np.array((mask.data.cpu()))
    prob_map = np.array((prob[:, 1].data.cpu()))

    ori_origin =  resample_image.GetOrigin()
    ori_spacing = resample_image.GetSpacing()
    ori_direction = resample_image.GetDirection()

    tar_size =  pre_image.GetSize()
    tar_spacing = pre_image.GetSpacing()

    mask = sitk.GetImageFromArray(mask)
    mask.SetOrigin(ori_origin)
    mask.SetSpacing(ori_spacing)
    mask.SetDirection(ori_direction)

    prob_map = sitk.GetImageFromArray(prob_map)
    prob_map.SetOrigin(ori_origin)
    prob_map.SetSpacing(ori_spacing)
    prob_map.SetDirection(ori_direction)

    pre_mask = resize_image_itk(mask, tar_size, tar_spacing)
    pre_prob_map = resize_image_itk(prob_map, tar_size, tar_spacing)

    return pre_mask, pre_prob_map


def test(input_path, model_path, output_folder, seg_name='seg.mha', gpu_id=0, save_image=True, save_single_prob=True):
    total_test_time = 0
    model = load_model(model_path, gpu_id)

    suffix = ['.mhd', '.nii', '.hdr', '.nii.gz', '.mha', '.image3d']
    if os.path.isfile(input_path):
        # test image files in the text (single-modality) or csv (multi-modality) file
        if input_path.endswith('txt'):
            file_list, case_list = read_test_txt(input_path)
        elif input_path.endswith('csv'):
            file_list, case_list = read_test_csv(input_path)
        else:
            raise ValueError('image test_list must either be a txt file or a csv file')
    elif os.path.isdir(input_path):
        # test all image file in input folder (single-modality)
        file_list, case_list = read_test_folder(input_path)
    else:
        raise ValueError('Input path do not exist!')

    success_cases = 0
    model_name = os.path.basename(model_path)
    model_in_channels = model.__dict__[model_name].in_channels
    for i, file in enumerate(file_list):
        print('{}: {}'.format(i, file))

        begin = time.time()
        images = []
        for image_path in file:
            image = sitk.ReadImage(image_path, outputPixelType=sitk.sitkFloat32)
            images.append(image)
        read_time = time.time() - begin

        begin = time.time()

        iso_batch, pre_image, resample_image = prepare_image_fixed_spacing(images[:model_in_channels], model[model_name])
        mask, prob_map = network_output(iso_batch, model[model_name], pre_image, resample_image)
        test_time = time.time() - begin

        casename = case_list[i]
        out_folder = os.path.join(output_folder, casename)
        if not os.path.isdir(out_folder):
            os.makedirs(out_folder)

        begin = time.time()
        if save_image:
            if len(images) == 1:
                ct_path = os.path.join(out_folder, 'org.nii.gz')
                sitk.WriteImage(images[0], ct_path)
            else:
                for num in range(len(images)):
                    ct_path = os.path.join(out_folder, 'org{}'.format(num+1) + '.nii.gz')
                    sitk.WriteImage(images[num], ct_path)

        seg_path = os.path.join(out_folder, seg_name)
        sitk.WriteImage(mask, seg_path)

        if save_single_prob and prob_map or True:
            prob_path = os.path.join(out_folder, 'prob2.nii.gz')
            sitk.WriteImage(prob_map, prob_path)
        output_time = time.time() - begin

        total_time = read_time + test_time + output_time
        total_test_time = test_time + total_test_time
        success_cases += 1
        print('read: {:.2f} s, test: {:.2f} s, write: {:.2f} s, total: {:.2f} s, avg test time: {:.2f}'.format(
            read_time, test_time, output_time, total_time, total_test_time / float(success_cases)))


def main():

    from argparse import RawTextHelpFormatter

    long_description = 'UII Brain Segmentation2d Batch Testing Engine\n\n' \
                       'It supports multiple kinds of input:\n' \
                       '1. Image list txt file\n' \
                       '2. Single image file\n' \
                       '3. A folder that contains all testing images\n'

    parser = argparse.ArgumentParser(description=long_description,
                                     formatter_class=RawTextHelpFormatter)

    parser.add_argument('-i', '--input', type=str, help='input folder/file for intensity images', default='/data/qingzhou/whm/vseg_test.csv')
    parser.add_argument('-m', '--model', type=str, help='model root folder', default='/data/qingzhou/whm/wmh_2d_4')
    parser.add_argument('-o', '--output', type=str, help='output folder for segmentation', default='/data/qingzhou/whm/wmh_2d/predicts')
    parser.add_argument('-n', '--seg_name', default='seg2.nii.gz', help='the name of the segmentation result to be saved')
    parser.add_argument('-g', '--gpu_id', default='5', help='the gpu id to run model')
    parser.add_argument('--save_image', help='whether to save original image', action="store_true")
    parser.add_argument('--save_single_prob', help='whether to save single prob map', action="store_true")
    args = parser.parse_args()
    test(args.input, args.model, args.output, args.seg_name, int(args.gpu_id), args.save_image,
                 args.save_single_prob)


if __name__ == '__main__':
    main()


