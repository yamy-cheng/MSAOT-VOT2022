from statistics import mode
import torch
import torch.nn.functional as F
import os
import sys
import cv2
import importlib
import math
import numpy as np
import time
from PIL import Image
from skimage.morphology.binary import binary_dilation
import copy

_palette = [
    255, 0, 0, 0, 0, 139, 255, 255, 84, 0, 255, 0, 139, 0, 139, 0, 128, 128,
    128, 128, 128, 139, 0, 0, 218, 165, 32, 144, 238, 144, 160, 82, 45, 148, 0,
    211, 255, 0, 255, 30, 144, 255, 255, 218, 185, 85, 107, 47, 255, 140, 0,
    50, 205, 50, 123, 104, 238, 240, 230, 140, 72, 61, 139, 128, 128, 0, 0, 0,
    205, 221, 160, 221, 143, 188, 143, 127, 255, 212, 176, 224, 230, 244, 164,
    96, 250, 128, 114, 70, 130, 180, 0, 128, 0, 173, 255, 47, 255, 105, 180,
    238, 130, 238, 154, 205, 50, 220, 20, 60, 176, 48, 96, 0, 206, 209, 0, 191,
    255, 40, 40, 40, 41, 41, 41, 42, 42, 42, 43, 43, 43, 44, 44, 44, 45, 45,
    45, 46, 46, 46, 47, 47, 47, 48, 48, 48, 49, 49, 49, 50, 50, 50, 51, 51, 51,
    52, 52, 52, 53, 53, 53, 54, 54, 54, 55, 55, 55, 56, 56, 56, 57, 57, 57, 58,
    58, 58, 59, 59, 59, 60, 60, 60, 61, 61, 61, 62, 62, 62, 63, 63, 63, 64, 64,
    64, 65, 65, 65, 66, 66, 66, 67, 67, 67, 68, 68, 68, 69, 69, 69, 70, 70, 70,
    71, 71, 71, 72, 72, 72, 73, 73, 73, 74, 74, 74, 75, 75, 75, 76, 76, 76, 77,
    77, 77, 78, 78, 78, 79, 79, 79, 80, 80, 80, 81, 81, 81, 82, 82, 82, 83, 83,
    83, 84, 84, 84, 85, 85, 85, 86, 86, 86, 87, 87, 87, 88, 88, 88, 89, 89, 89,
    90, 90, 90, 91, 91, 91, 92, 92, 92, 93, 93, 93, 94, 94, 94, 95, 95, 95, 96,
    96, 96, 97, 97, 97, 98, 98, 98, 99, 99, 99, 100, 100, 100, 101, 101, 101,
    102, 102, 102, 103, 103, 103, 104, 104, 104, 105, 105, 105, 106, 106, 106,
    107, 107, 107, 108, 108, 108, 109, 109, 109, 110, 110, 110, 111, 111, 111,
    112, 112, 112, 113, 113, 113, 114, 114, 114, 115, 115, 115, 116, 116, 116,
    117, 117, 117, 118, 118, 118, 119, 119, 119, 120, 120, 120, 121, 121, 121,
    122, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126,
    127, 127, 127, 128, 128, 128, 129, 129, 129, 130, 130, 130, 131, 131, 131,
    132, 132, 132, 133, 133, 133, 134, 134, 134, 135, 135, 135, 136, 136, 136,
    137, 137, 137, 138, 138, 138, 139, 139, 139, 140, 140, 140, 141, 141, 141,
    142, 142, 142, 143, 143, 143, 144, 144, 144, 145, 145, 145, 146, 146, 146,
    147, 147, 147, 148, 148, 148, 149, 149, 149, 150, 150, 150, 151, 151, 151,
    152, 152, 152, 153, 153, 153, 154, 154, 154, 155, 155, 155, 156, 156, 156,
    157, 157, 157, 158, 158, 158, 159, 159, 159, 160, 160, 160, 161, 161, 161,
    162, 162, 162, 163, 163, 163, 164, 164, 164, 165, 165, 165, 166, 166, 166,
    167, 167, 167, 168, 168, 168, 169, 169, 169, 170, 170, 170, 171, 171, 171,
    172, 172, 172, 173, 173, 173, 174, 174, 174, 175, 175, 175, 176, 176, 176,
    177, 177, 177, 178, 178, 178, 179, 179, 179, 180, 180, 180, 181, 181, 181,
    182, 182, 182, 183, 183, 183, 184, 184, 184, 185, 185, 185, 186, 186, 186,
    187, 187, 187, 188, 188, 188, 189, 189, 189, 190, 190, 190, 191, 191, 191,
    192, 192, 192, 193, 193, 193, 194, 194, 194, 195, 195, 195, 196, 196, 196,
    197, 197, 197, 198, 198, 198, 199, 199, 199, 200, 200, 200, 201, 201, 201,
    202, 202, 202, 203, 203, 203, 204, 204, 204, 205, 205, 205, 206, 206, 206,
    207, 207, 207, 208, 208, 208, 209, 209, 209, 210, 210, 210, 211, 211, 211,
    212, 212, 212, 213, 213, 213, 214, 214, 214, 215, 215, 215, 216, 216, 216,
    217, 217, 217, 218, 218, 218, 219, 219, 219, 220, 220, 220, 221, 221, 221,
    222, 222, 222, 223, 223, 223, 224, 224, 224, 225, 225, 225, 226, 226, 226,
    227, 227, 227, 228, 228, 228, 229, 229, 229, 230, 230, 230, 231, 231, 231,
    232, 232, 232, 233, 233, 233, 234, 234, 234, 235, 235, 235, 236, 236, 236,
    237, 237, 237, 238, 238, 238, 239, 239, 239, 240, 240, 240, 241, 241, 241,
    242, 242, 242, 243, 243, 243, 244, 244, 244, 245, 245, 245, 246, 246, 246,
    247, 247, 247, 248, 248, 248, 249, 249, 249, 250, 250, 250, 251, 251, 251,
    252, 252, 252, 253, 253, 253, 254, 254, 254, 255, 255, 255, 0, 0, 0
]

AOT_PATH = os.path.join(os.path.dirname(__file__), 'MS_AOT')
sys.path.append(AOT_PATH)

import dataloaders.video_transforms as tr
from torchvision import transforms
from networks.engines import build_engine
from utils.checkpoint import load_network
from networks.models import build_vos_model

def debug(info):
    assert 1==0, f"{info}"

color_palette = np.array(_palette).reshape(-1, 3)

def overlay(image, mask, colors=[255, 0, 0], cscale=1, alpha=0.4):
    colors = np.atleast_2d(colors) * cscale

    im_overlay = image.copy()
    object_ids = np.unique(mask)

    for object_id in object_ids[1:]:
        # Overlay color on  binary mask

        foreground = image * alpha + np.ones(
            image.shape) * (1 - alpha) * np.array(colors[object_id])
        binary_mask = mask == object_id

        # Compose image
        im_overlay[binary_mask] = foreground[binary_mask]

        countours = binary_dilation(binary_mask) ^ binary_mask
        im_overlay[countours, :] = 0

    return im_overlay.astype(image.dtype)


class AOTTracker(object):
    def __init__(self, cfg, sr, input_sz, visualize_folder, gpu_id):
        self.gpu_id = gpu_id
        self.model = build_vos_model(cfg.MODEL_VOS, cfg).cuda(gpu_id)
        self.model, _ = load_network(self.model, cfg.TEST_CKPT_PATH, gpu_id)
        self.engine = build_engine(cfg.MODEL_ENGINE,
                                   phase='eval',
                                   aot_model=self.model,
                                   gpu_id=gpu_id,
                                   short_term_mem_skip=4,
                                   long_term_mem_gap=cfg.TEST_LONG_TERM_MEM_GAP)
       
        self.transform = transforms.Compose([
            tr.MultiRestrictSize(cfg.TEST_MAX_SHORT_EDGE,
                                 cfg.TEST_MAX_LONG_EDGE, cfg.TEST_FLIP, cfg.TEST_INPLACE_FLIP,
                                 cfg.TEST_MULTISCALE, cfg.MODEL_ALIGN_CORNERS),
            tr.MultiToTensor()
        ])

        self.model.eval()

        self.sr = sr
        self.input_sz = input_sz
        self.search_factor = self.sr
        self.visualize_folder = visualize_folder

        self.id = 2

    def reset_id(self):
        self.id = 2
    
    def get_new_sf(self, gt_box, edge):
        w, h = gt_box[-2:]
        new_sf = edge / max(int(w), int(h))
        return new_sf

    def vis_mask(self, output_mask, output_mask_path):
        output_mask = Image.fromarray(
                        output_mask.astype('uint8')
                    ).convert('P')

        output_mask.putpalette(_palette)
        output_mask.save(output_mask_path)


    def add_reference_frame(self, frame, bbox, mask, fixed_edge):
        input_frame = frame.copy()
        mask = cv2.resize(mask, frame.shape[:2][::-1], interpolation = cv2.INTER_NEAREST)
        self.search_factor = self.get_new_sf(bbox, fixed_edge)
        Cpatch, Cmask, output_h, output_w = sample_target_SE_with_mask_2(frame, bbox, mask, self.search_factor, self.input_sz, mode=cv2.BORDER_CONSTANT)

        sample = {
            'current_img': Cpatch,
            'current_label': Cmask.squeeze(2) if Cmask.ndim == 3 else Cmask,
        }
    
        sample = self.transform(sample)
        frame = sample[0]['current_img'].unsqueeze(0).float().cuda(self.gpu_id, non_blocking=True)
        mask = sample[0]['current_label'].unsqueeze(0).float().cuda(self.gpu_id, non_blocking=True)

        self.engine.add_reference_frame(frame, mask, frame_step=0, obj_nums=1)



    
    def track(self, image, bbox, fixed_edge):
        output_height, output_width = image.shape[0], image.shape[1]
        input_img = image.copy()
        with torch.no_grad():
            # extract the region for aot to predict
            self.search_factor = self.get_new_sf(bbox, fixed_edge)
            Cpatch, output_h, output_w = sample_target_SE_2(image, bbox, self.search_factor, self.input_sz, mode=cv2.BORDER_CONSTANT)

            sample = {'current_img': Cpatch}
            sample = self.transform(sample)
            image = sample[0]['current_img'].unsqueeze(0).float().cuda(self.gpu_id, non_blocking=True)
            self.engine.match_propogate_one_frame(image)
            pred_logit = self.engine.decode_current_logits(
                            (output_h, output_w))
            pred_prob = torch.softmax(pred_logit, dim=1)
            pred_label = torch.argmax(pred_prob, dim=1,
                                        keepdim=True).float()
            _pred_label = F.interpolate(pred_label,
                                size=(self.input_sz, self.input_sz),
                                mode="nearest")
            conf = torch.sum(pred_prob[:, 1, :, :] * pred_label) / torch.sum(pred_label)
            conf = torch.nan_to_num(conf, nan=0)
            # update memory
            # if have object and conf > 0.7, update memory
            is_valid = torch.sum(_pred_label > 0) and conf > 0.7
            self.engine.update_memory(_pred_label, is_valid)

            Cmask = pred_label.detach().cpu().numpy()[0][0].astype(np.uint8)

            # self.id += 1
            mask = map_mask_back_2(input_img, bbox, self.search_factor, Cmask, mode=cv2.BORDER_REPLICATE)
            rect_bbox = self._rect_from_mask(mask)

            return {
                'mask': mask,
                'bbox': rect_bbox if rect_bbox is not None else bbox,
                'conf': conf,
                'valid': True if rect_bbox is not None else False
            }

    def track_without_update(self, image, bbox, fixed_edge):
        output_height, output_width = image.shape[0], image.shape[1]
        input_img = image.copy()
        with torch.no_grad():
            # extract the region for aot to predict
            self.search_factor = self.get_new_sf(bbox, fixed_edge)
            Cpatch, output_h, output_w = sample_target_SE_2(image, bbox, self.search_factor, self.input_sz, mode=cv2.BORDER_CONSTANT)
            sample = {'current_img': Cpatch}

            sample = self.transform(sample)
            image = sample[0]['current_img'].unsqueeze(0).float().cuda(self.gpu_id)
            self.engine.match_propogate_one_frame(image)
            pred_logit = self.engine.decode_current_logits(
                            (output_h, output_w))
            pred_prob = torch.softmax(pred_logit, dim=1)
            pred_label = torch.argmax(pred_prob, dim=1,
                                        keepdim=True).float()
            _pred_label = F.interpolate(pred_label,
                                size=(self.input_sz, self.input_sz),
                                mode="nearest")
            conf = torch.sum(pred_prob[:, 1, :, :] * pred_label) / torch.sum(pred_label)
            conf = torch.nan_to_num(conf, nan=0)
            # update memory
            # if have object and conf > 0.7, update memory
            is_valid = torch.sum(_pred_label > 0) and conf > 0.7
            #self.engine.update_memory(_pred_label, is_valid)

            Cmask = pred_label.detach().cpu().numpy()[0][0].astype(np.uint8)
            mask = map_mask_back_2(input_img, bbox, self.search_factor, Cmask, mode=cv2.BORDER_REPLICATE)
            
            rect_bbox = self._rect_from_mask(mask)

            return {
                'mask': mask,
                'bbox': rect_bbox if rect_bbox is not None else bbox,
                'conf': conf,
                'valid': True if rect_bbox is not None else False,
                '_pred_label':_pred_label,
                'memory_is_valid':is_valid
            }
        
    def update_memory(self, _pred_label, is_valid):
        self.engine.update_memory(_pred_label, is_valid)

    def _rect_from_mask(self, mask):
        if len(np.where(mask==1)[0]) == 0:
            return None
        x_ = np.sum(mask, axis=0)
        y_ = np.sum(mask, axis=1)
        x0 = np.min(np.nonzero(x_))
        x1 = np.max(np.nonzero(x_))
        y0 = np.min(np.nonzero(y_))
        y1 = np.max(np.nonzero(y_))
        return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]

    
    def crop(self, image, bbox):
        bbox = [int(f) for f in bbox]
        x, y, w, h = bbox
        draw = cv2.rectangle(image.copy(), (x,y), (x+w,y+h), (0,255,0), 2)
        draw = cv2.resize(draw, self.size)
        self.video.write(draw)

    def crop_mask(self, mask, input_image):

        mask_ = copy.deepcopy(mask)
        mask_ = Image.fromarray(
                        mask_.astype('uint8')
                    ).convert('P')
        mask_.putpalette(_palette)
        
        overlayed_image = overlay(
                            np.array(input_image, dtype=np.uint8),
                            np.array(mask_, dtype=np.uint8), color_palette)
        self.video_mask.write(overlayed_image)


    def get_Cpatch_mask(self, mask, bbox, search_factor = None, output_sz = None, mode=cv2.BORDER_CONSTANT):
        if search_factor is None:
            search_factor = self.search_factor
        if output_sz is None:
            output_sz = self.input_sz

        if not isinstance(bbox, list):
            x, y, w, h = bbox.tolist()
        else:
            x, y, w, h = bbox

        max_length = int(max(w, h) * search_factor)

        if max_length < 1:
            raise Exception('Too small bounding box.')
        
        # Crop image
        x1 = round(x + 0.5*w - max_length*0.5)
        x2 = x1 + max_length

        y1 = round(y + 0.5*h - max_length*0.5)
        y2 = y1 + max_length


        x1_pad = max(0, -x1)
        x2_pad = max(x2-mask.shape[1]+1, 0)

        y1_pad = max(0, -y1)
        y2_pad = max(y2-mask.shape[0]+1, 0)

        # Crop target
        mask_crop = mask[y1+y1_pad:y2-y2_pad, x1+x1_pad:x2-x2_pad]

        # Pad
        mask_crop_padded = cv2.copyMakeBorder(mask_crop, y1_pad, y2_pad, x1_pad, x2_pad, mode)

        if output_sz is not None:
            w_rsz_f = output_sz / max_length
            h_rsz_f = output_sz / max_length
            im_crop_padded_rsz = cv2.resize(mask_crop_padded, (output_sz, output_sz))
            if len(im_crop_padded_rsz.shape)==2:
                im_crop_padded_rsz = im_crop_padded_rsz[...,np.newaxis]
            return im_crop_padded_rsz
        else:
            return mask_crop_padded


config = {
    'phase': 'test',
    'model': 'R50_AOTv3',
    'pretrain_model_path': 'pretrain_models/ms_aot_model.pth',
    'gpu_id': 0,
}


def get_aot(sr, input_sz, visualize_folder):
    # build vos engine
    engine_config = importlib.import_module('configs.' + 'ms_aot')
    cfg = engine_config.EngineConfig(config['phase'], config['model'])
    cfg.TEST_CKPT_PATH = os.path.join(AOT_PATH, config['pretrain_model_path'])

    # init AOTTracker
    tracker = AOTTracker(cfg, sr, input_sz, visualize_folder, config['gpu_id'])
    return tracker


def sample_target_SE_with_mask(im, target_bb, mask, search_area_factor, output_sz=None, mode=cv2.BORDER_REPLICATE):
    """ Extracts a crop centered at target_bb box, of size search_area_factor times target_bb(Both height and width)

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """

    if not isinstance(target_bb, list):
        x, y, w, h = target_bb.tolist()
    else:
        x, y, w, h = target_bb

    # Crop image
    ws = math.ceil(search_area_factor * w)
    hs = math.ceil(search_area_factor * h)

    if ws < 1 or hs < 1:
        raise Exception('Too small bounding box.')

    x1 = round(x + 0.5*w - ws*0.5)
    x2 = x1 + ws

    y1 = round(y + 0.5 * h - hs * 0.5)
    y2 = y1 + hs

    x1_pad = max(0, -x1)
    x2_pad = max(x2-im.shape[1]+1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2-im.shape[0]+1, 0)

    # Crop target
    im_crop = im[y1+y1_pad:y2-y2_pad, x1+x1_pad:x2-x2_pad, :]
    mask_crop = mask[y1+y1_pad:y2-y2_pad, x1+x1_pad:x2-x2_pad, None]

    # Pad
    im_crop_padded = cv2.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, mode)
    mask_crop_padded = cv2.copyMakeBorder(mask_crop, y1_pad, y2_pad, x1_pad, x2_pad,
                                         borderType=cv2.BORDER_CONSTANT, value=0)

    if output_sz is not None:
        w_rsz_f = output_sz / ws
        h_rsz_f = output_sz / hs
        im_crop_padded_rsz = cv2.resize(im_crop_padded, (output_sz, output_sz))
        mask_crop_padded_rsz = cv2.resize(mask_crop_padded, (output_sz, output_sz))
        if len(im_crop_padded_rsz.shape)==2:
            im_crop_padded_rsz = im_crop_padded_rsz[...,np.newaxis]
        if len(mask_crop_padded_rsz.shape)==2:
            mask_crop_padded_rsz = mask_crop_padded_rsz[...,np.newaxis]
        return im_crop_padded_rsz, mask_crop_padded_rsz, h_rsz_f, w_rsz_f
    else:
        return im_crop_padded, mask_crop_padded, 1.0, 1.0


def sample_target_SE(im, target_bb, search_area_factor, output_sz=None, mode=cv2.BORDER_REPLICATE):
    """ Extracts a crop centered at target_bb box, of size search_area_factor times target_bb(Both height and width)

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """

    if not isinstance(target_bb, list):
        x, y, w, h = target_bb.tolist()
    else:
        x, y, w, h = target_bb

    # Crop image
    ws = math.ceil(search_area_factor * w)
    hs = math.ceil(search_area_factor * h)

    if ws < 1 or hs < 1:
        raise Exception('Too small bounding box.')

    x1 = round(x + 0.5*w - ws*0.5)
    x2 = x1 + ws

    y1 = round(y + 0.5 * h - hs * 0.5)
    y2 = y1 + hs

    x1_pad = max(0, -x1)
    x2_pad = max(x2-im.shape[1]+1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2-im.shape[0]+1, 0)

    # Crop target
    im_crop = im[y1+y1_pad:y2-y2_pad, x1+x1_pad:x2-x2_pad, :]

    # Pad
    im_crop_padded = cv2.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, mode)

    if output_sz is not None:
        w_rsz_f = output_sz / ws
        h_rsz_f = output_sz / hs
        im_crop_padded_rsz = cv2.resize(im_crop_padded, (output_sz, output_sz))
        if len(im_crop_padded_rsz.shape)==2:
            im_crop_padded_rsz = im_crop_padded_rsz[...,np.newaxis]
        return im_crop_padded_rsz, h_rsz_f, w_rsz_f
    else:
        return im_crop_padded, 1.0, 1.0


def map_mask_back(im, target_bb, search_area_factor, mask, mode=cv2.BORDER_REPLICATE):
    """ Extracts a crop centered at target_bb box, of size search_area_factor times target_bb(Both height and width)

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """
    H,W = (im.shape[2],im.shape[3])
    base = np.zeros((H,W))

    if not isinstance(target_bb, list):
        x, y, w, h = target_bb.tolist()
    else:
        x, y, w, h = target_bb

    # Crop image
    ws = math.ceil(search_area_factor * w)
    hs = math.ceil(search_area_factor * h)

    if ws < 1 or hs < 1:
        raise Exception('Too small bounding box.')

    x1 = round(x + 0.5*w - ws*0.5)
    x2 = x1 + ws

    y1 = round(y + 0.5 * h - hs * 0.5)
    y2 = y1 + hs

    x1_pad = max(0, -x1)
    x2_pad = max(x2-im.shape[1]+1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2-im.shape[0]+1, 0)

    '''pad base'''
    base_padded = cv2.copyMakeBorder(base, y1_pad, y2_pad, x1_pad, x2_pad, mode)
    '''Resize mask'''
    mask_rsz = cv2.resize(mask,(ws,hs))
    '''fill region with mask'''
    base_padded[y1+y1_pad:y2+y1_pad, x1+x1_pad:x2+x1_pad] = mask_rsz.copy()
    '''crop base_padded to get final mask'''
    final_mask = base_padded[y1_pad:y1_pad+H,x1_pad:x1_pad+W]
    assert (final_mask.shape == (H,W))
    return final_mask




def sample_target_SE_with_mask_2(im, target_bb, mask, search_area_factor, output_sz=None, mode=cv2.BORDER_REPLICATE):
    """ Extracts a crop centered at target_bb box, of size search_area_factor times target_bb(Both height and width)

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """

    if not isinstance(target_bb, list):
        x, y, w, h = target_bb.tolist()
    else:
        x, y, w, h = target_bb

    
    max_length = int(max(w, h) * search_area_factor)

    if max_length < 1:
        raise Exception('Too small bounding box.')
    
    # Crop image
    x1 = round(x + 0.5*w - max_length*0.5)
    x2 = x1 + max_length

    y1 = round(y + 0.5*h - max_length*0.5)
    y2 = y1 + max_length

    x1_pad = max(0, -x1)
    x2_pad = max(x2-im.shape[1]+1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2-im.shape[0]+1, 0)

    # Crop target
    im_crop = im[y1+y1_pad:y2-y2_pad, x1+x1_pad:x2-x2_pad, :]
    mask_crop = mask[y1+y1_pad:y2-y2_pad, x1+x1_pad:x2-x2_pad, None]

    # Pad
    im_crop_padded = cv2.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, mode)
    mask_crop_padded = cv2.copyMakeBorder(mask_crop, y1_pad, y2_pad, x1_pad, x2_pad,
                                         borderType=cv2.BORDER_CONSTANT, value=0)

    if output_sz is not None:
        w_rsz_f = output_sz / max_length
        h_rsz_f = output_sz / max_length
        im_crop_padded_rsz = cv2.resize(im_crop_padded, (output_sz, output_sz))
        mask_crop_padded_rsz = cv2.resize(mask_crop_padded, (output_sz, output_sz))
        if len(im_crop_padded_rsz.shape)==2:
            im_crop_padded_rsz = im_crop_padded_rsz[...,np.newaxis]
        if len(mask_crop_padded_rsz.shape)==2:
            mask_crop_padded_rsz = mask_crop_padded_rsz[...,np.newaxis]
        return im_crop_padded_rsz, mask_crop_padded_rsz, h_rsz_f, w_rsz_f
    else:
        return im_crop_padded, mask_crop_padded, 1.0, 1.0


def sample_target_SE_2(im, target_bb, search_area_factor, output_sz=None, mode=cv2.BORDER_REPLICATE):
    """ Extracts a crop centered at target_bb box, of size search_area_factor times target_bb(Both height and width)

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """

    if not isinstance(target_bb, list):
        x, y, w, h = target_bb.tolist()
    else:
        x, y, w, h = target_bb

    max_length = int(max(w, h) * search_area_factor)

    if max_length < 1:
        raise Exception('Too small bounding box.')
    
    # Crop image
    x1 = round(x + 0.5*w - max_length*0.5)
    x2 = x1 + max_length

    y1 = round(y + 0.5*h - max_length*0.5)
    y2 = y1 + max_length


    x1_pad = max(0, -x1)
    x2_pad = max(x2-im.shape[1]+1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2-im.shape[0]+1, 0)

    # Crop target
    im_crop = im[y1+y1_pad:y2-y2_pad, x1+x1_pad:x2-x2_pad, :]

    # Pad
    im_crop_padded = cv2.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, mode)

    if output_sz is not None:
        w_rsz_f = output_sz / max_length
        h_rsz_f = output_sz / max_length
        im_crop_padded_rsz = cv2.resize(im_crop_padded, (output_sz, output_sz))
        if len(im_crop_padded_rsz.shape)==2:
            im_crop_padded_rsz = im_crop_padded_rsz[...,np.newaxis]
        return im_crop_padded_rsz, im_crop_padded.shape[0], im_crop_padded.shape[1]
    else:
        return im_crop_padded, 1.0, 1.0


def map_mask_back_2(im, target_bb, search_area_factor, mask, mode=cv2.BORDER_REPLICATE):
    """ Extracts a crop centered at target_bb box, of size search_area_factor times target_bb(Both height and width)

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """
    H,W = (im.shape[0],im.shape[1])
    base = np.zeros((H,W))
    if not isinstance(target_bb, list):
        x, y, w, h = target_bb.tolist()
    else:
        x, y, w, h = target_bb

    max_length = int(max(w, h) * search_area_factor)

    if max_length < 1:
        raise Exception('Too small bounding box.')
    
    # Crop image
    x1 = round(x + 0.5*w - max_length*0.5)
    x2 = x1 + max_length

    y1 = round(y + 0.5*h - max_length*0.5)
    y2 = y1 + max_length

    x1_pad = max(0, -x1)
    x2_pad = max(x2-im.shape[1]+1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2-im.shape[0]+1, 0)

    '''pad base'''
    base_padded = cv2.copyMakeBorder(base, y1_pad, y2_pad, x1_pad, x2_pad, mode)
    '''Resize mask'''
    #mask_rsz = cv2.resize(mask,(max_length, max_length))
    mask_rsz = mask
    '''fill region with mask'''
    base_padded[y1+y1_pad:y2+y1_pad, x1+x1_pad:x2+x1_pad] = mask_rsz.copy()
    '''crop base_padded to get final mask'''
    final_mask = base_padded[y1_pad:y1_pad+H,x1_pad:x1_pad+W]
    assert (final_mask.shape == (H,W))
    return final_mask