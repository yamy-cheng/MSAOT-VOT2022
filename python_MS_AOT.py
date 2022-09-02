
from random import sample
from PIL import Image
from rsa import sign
import torch
import torch.nn.functional as F
import vot_utils
import os
import sys
import cv2
import importlib
import numpy as np
import math
import random

AOT_PATH = os.path.join(os.path.dirname(__file__), 'MS_AOT')
MIXFORMER_PATH = os.path.join(os.path.dirname(__file__), 'MS_AOT/MixFormer')
MIXFORMER_PYTRACKING_PATH = os.path.join(os.path.dirname(__file__), 'MS_AOT/MixFormer/external/AR/pytracking')
sys.path.append(AOT_PATH)
sys.path.append(MIXFORMER_PATH)
sys.path.append(MIXFORMER_PYTRACKING_PATH)

import MS_AOT.dataloaders.video_transforms as tr
from torchvision import transforms
from MS_AOT.networks.engines import build_engine
from MS_AOT.utils.checkpoint import load_network
from MS_AOT.networks.models import build_vos_model

from aot_tools import get_aot
from MS_AOT.MixFormer.lib.test.tracker.mixformer_online import MixFormerOnline
import MS_AOT.MixFormer.lib.test.parameter.mixformer_online as vot_params

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

seed_torch(1000000007)
torch.set_num_threads(4)
torch.autograd.set_grad_enabled(False)

class AOTMixFormerTracker(object):
    def __init__(self):
        self.THRES_OUTER = 0.75
        self.THRES_INNER = 0.55
        self.THRES_CROSS = 0.45
        self.tracker = self.load_mixformer()
        self.sr = 1
        self.input_sz = 465
        self.visualize_folder = 'tracker_MIXAOT'
        self.aot = get_aot(self.sr, self.input_sz, self.visualize_folder)
        self.id = 2
        self.fixed_edge = None
        self.thres = 0.1
        self.large_ratio = 0.7
        self.mid_ratio = 120

    def initialize(self, image, mask):
        self.aot.reset_id()
        self.aot.model.eval()
        self.aot.engine.restart_engine()
        
        # generate bbox as vot2019 
        rotated_bbox = self._mask_post_processing(mask)
        rotated_bbox=np.array([rotated_bbox[0][0],rotated_bbox[0][1],rotated_bbox[1][0],rotated_bbox[1][1],rotated_bbox[2][0],rotated_bbox[2][1],rotated_bbox[3][0],rotated_bbox[3][1]])
        cx,cy,w,h = self.get_axis_aligned_bbox(rotated_bbox)
        gt_bbox = [cx-w/2, cy-h/2, w, h]             

        # TODO: only use the center coordinate, the size of croped region is fix
        gt_bbox_sz = w *h
        img_sz = int(image.shape[1] * image.shape[0])
        
        # get the edge of the croped region
        self.fixed_edge = self.get_fix_edge(img_sz, gt_bbox_sz)
        init_info = {'init_bbox': gt_bbox}
        self.tracker.initialize(image, init_info)
        self.aot.add_reference_frame(image, gt_bbox, mask, self.fixed_edge)
        
    def track(self, img_RGB):
        '''TRACK'''
        '''bbox: [x, y, w, h]'''
        '''base tracker'''
        outputs = self.tracker.track_without_update(img_RGB)
        pred_bbox = outputs['target_bbox']
        pred_score = outputs['pred_score']

        # AOT tracker
        pred = self.aot.track_without_update(img_RGB, pred_bbox, self.fixed_edge)
        bbox, pred_mask = pred['bbox'], pred['mask']

        if pred['valid']:
            # compare bbox with pred_bbox(the pred of aot and mixformer)
            iou = self.IoU(pred_bbox, bbox)
            if iou < self.thres :
                pred_bbox_sz = pred_bbox[2] * pred_bbox[3]
                bbox_sz = bbox[2] * bbox[3]
                
                if  max(pred_bbox_sz, bbox_sz) / min(pred_bbox_sz, bbox_sz) < 5: 
                    self.tracker.update_state(bbox)
                    self.tracker.update_online_template(bbox, pred_score, img_RGB)
                    self.aot.update_memory(pred['_pred_label'], pred['memory_is_valid'])
                else:
                    pred_mask_sub = pred_mask.copy()
                    pred_mask_sub[:,:] = 0
                    x, y, w, h = pred_bbox
                    pred_mask_sub[int(y):int(y)+int(h), int(x):int(x)+int(w)] = 1
                    pred_mask = pred_mask * pred_mask_sub
                    self.tracker.update_online_template(pred_bbox, pred_score, img_RGB)
                        
            else:
                self.aot.update_memory(pred['_pred_label'], pred['memory_is_valid'])
                self.tracker.update_online_template(pred_bbox, pred_score, img_RGB)

        else:
            x, y, w, h = pred_bbox
            pred_mask[int(y):int(y)+int(h), int(x):int(x)+int(w)] = 1

        return pred_mask.astype(np.uint8), 1


    def IoU(self, box1, box2):
        """
        :param box1: list in format [lt_x, lt_y, w, h]
        :param box2:  list in format [lt_x, lt_y, w, h]
        :return:    returns IoU ratio (intersection over union) of two boxes
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        xmin1, ymin1 = x1, y1
        xmin2, ymin2 = x2, y2

        xmax1, ymax1 = xmin1+w1, ymin1+h1
        xmax2, ymax2 = xmin2+w2, ymin2+h2
        x_overlap = max(0, min(xmax1, xmax2) - max(xmin1, xmin2))
        y_overlap = max(0, min(ymax1, ymax2) - max(ymin1, ymin2))
        intersection = x_overlap * y_overlap
        union = (w1) * (h1) + (w2) * (h2) - intersection
        return float(intersection) / union

    def get_fix_edge(self, img_sz, gt_bbox_sz):
        ratio = img_sz / gt_bbox_sz
        if ratio > 900:
            fixed_edge = int(math.sqrt((img_sz / 12)))
        else:
            fixed_edge = int(math.sqrt((gt_bbox_sz * self.mid_ratio)))
        
        # upper bound
        if gt_bbox_sz * self.mid_ratio > self.large_ratio * img_sz:
            fixed_edge = int(math.sqrt((self.large_ratio * img_sz)))
            if gt_bbox_sz * 2 > self.large_ratio * img_sz:
                fixed_edge = int(math.sqrt((gt_bbox_sz * 2)))  

        return fixed_edge

    def _rect_from_mask(self, mask):
        x_ = np.sum(mask, axis=0)
        y_ = np.sum(mask, axis=1)
        x0 = np.min(np.nonzero(x_))
        x1 = np.max(np.nonzero(x_))
        y0 = np.min(np.nonzero(y_))
        y1 = np.max(np.nonzero(y_))
        return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


    def _mask_post_processing(self, mask):
        target_mask = (mask > 0.5)
        target_mask = target_mask.astype(np.uint8)
        if cv2.__version__[-5] == '4':
            contours, _ = cv2.findContours(target_mask, 
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:
            _, contours, _ = cv2.findContours(target_mask, 
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        if len(contours) != 0 and np.max(cnt_area) > 50:#cnt_area=100
            contour = contours[np.argmax(cnt_area)] 
            polygon = contour.reshape(-1, 2)

            ## the following code estimate the shape angle with ellipse
            ## then fit a axis-aligned bounding box on the rotated image
            
            ellipseBox = cv2.fitEllipse(polygon)
            # get the center of the ellipse and the angle
            angle = ellipseBox[-1]
            #print(angle)
            center = np.array(ellipseBox[0])
            axes = np.array(ellipseBox[1])
            
            # get the ellipse box
            ellipseBox = cv2.boxPoints(ellipseBox)
            
            #compute the rotation matrix
            rot_mat = cv2.getRotationMatrix2D((center[0],center[1]), angle, 1.0)
            
            # rotate the ellipse box
            one = np.ones([ellipseBox.shape[0],3,1])
            one[:,:2,:] = ellipseBox.reshape(-1,2,1)
            ellipseBox = np.matmul(rot_mat, one).reshape(-1,2)
            
            # to xmin ymin xmax ymax
            xs = ellipseBox[:,0]
            xmin, xmax = np.min(xs), np.max(xs)
            ys = ellipseBox[:,1]
            ymin, ymax = np.min(ys), np.max(ys)
            ellipseBox = [xmin, ymin, xmax, ymax]
            
            # rotate the contour
            one = np.ones([polygon.shape[0],3,1])
            one[:,:2,:] = polygon.reshape(-1,2,1)
            polygon = np.matmul(rot_mat, one).astype(int).reshape(-1,2)
            
            # remove points outside of the ellipseBox
            logi = polygon[:,0]<=xmax
            logi = np.logical_and(polygon[:,0]>=xmin, logi)
            logi = np.logical_and(polygon[:,1]>=ymin, logi)
            logi = np.logical_and(polygon[:,1]<=ymax, logi)
            polygon = polygon[logi,:]
            
            x,y,w,h = cv2.boundingRect(polygon)
            bRect = [x, y, x+w, y+h]
            
            # get the intersection of ellipse box and the rotated box
            x1, y1, x2, y2 = ellipseBox[0], ellipseBox[1], ellipseBox[2], ellipseBox[3]
            tx1, ty1, tx2, ty2 = bRect[0], bRect[1], bRect[2], bRect[3]
            xx1 = min(max(tx1, x1, 0), target_mask.shape[1]-1)
            yy1 = min(max(ty1, y1, 0), target_mask.shape[0]-1)
            xx2 = max(min(tx2, x2, target_mask.shape[1]-1), 0)
            yy2 = max(min(ty2, y2, target_mask.shape[0]-1), 0)
            
            rotated_mask = cv2.warpAffine(target_mask, rot_mat,(target_mask.shape[1],target_mask.shape[0]))
            
            #refinement
            alpha_factor = 0.2583#cfg.TRACK.FACTOR
            while True:
                if np.sum(rotated_mask[int(yy1):int(yy2),int(xx1)]) < (yy2-yy1)*alpha_factor:
                    temp = xx1+(xx2-xx1)*0.02
                    if not (temp >= target_mask.shape[1]-1 or xx2-xx1 < 1):
                        xx1 = temp
                    else:
                        break
                else:
                    break
            while True:
                if np.sum(rotated_mask[int(yy1):int(yy2),int(xx2)]) < (yy2-yy1)*alpha_factor:
                    temp = xx2-(xx2-xx1)*0.02
                    if not (temp <= 0 or xx2-xx1 < 1):
                        xx2 = temp
                    else:
                        break
                else:
                    break
            while True:
                if np.sum(rotated_mask[int(yy1),int(xx1):int(xx2)]) < (xx2-xx1)*alpha_factor:
                    temp = yy1+(yy2-yy1)*0.02
                    if not (temp >= target_mask.shape[0]-1 or yy2-yy1 < 1):
                        yy1 = temp
                    else:
                        break
                else:
                    break
            while True:
                if np.sum(rotated_mask[int(yy2),int(xx1):int(xx2)]) < (xx2-xx1)*alpha_factor:
                    temp = yy2-(yy2-yy1)*0.02
                    if not (temp <= 0 or yy2-yy1 < 1):
                        yy2 = temp
                    else:
                        break
                else:
                    break
            
            prbox = np.array([[xx1,yy1],[xx2,yy1],[xx2,yy2],[xx1,yy2]])
            
            # inverse of the rotation matrix
            M_inv = cv2.invertAffineTransform(rot_mat)
            # project the points back to image coordinate
            one = np.ones([prbox.shape[0],3,1])
            one[:,:2,:] = prbox.reshape(-1,2,1)
            prbox = np.matmul(M_inv, one).reshape(-1,2)
            
            rbox_in_img = prbox
        else:  # empty mask
            # location = cxy_wh_2_rect(self.center_pos, self.size)
            location = [0,0,1,1]
            rbox_in_img = np.array([[location[0], location[1]],
                        [location[0] + location[2], location[1]],
                        [location[0] + location[2], location[1] + location[3]],
                        [location[0], location[1] + location[3]]])
        return rbox_in_img


    def get_axis_aligned_bbox(self,region):
        """ convert region to (cx, cy, w, h) that represent by axis aligned box
        """
        nv = region.size
        if nv == 8:
            cx = np.mean(region[0::2])
            cy = np.mean(region[1::2])
            x1 = min(region[0::2])
            x2 = max(region[0::2])
            y1 = min(region[1::2])
            y2 = max(region[1::2])
            A1 = np.linalg.norm(region[0:2] - region[2:4]) * \
                np.linalg.norm(region[2:4] - region[4:6])
            A2 = (x2 - x1) * (y2 - y1)
            s = np.sqrt(A1 / A2)
            w = s * (x2 - x1) + 1
            h = s * (y2 - y1) + 1
        else:
            x = region[0]
            y = region[1]
            w = region[2]
            h = region[3]
            cx = x+w/2
            cy = y+h/2
        return cx, cy, w, h       


    def _rect_from_mask(self, mask):
        '''
        create an axis-aligned rectangle from a given binary mask
        mask in created as a minimal rectangle containing all non-zero pixels
        '''
        # print(mask)
        x_ = np.sum(mask, axis=0)
        y_ = np.sum(mask, axis=1)
        x0 = np.min(np.nonzero(x_))
        x1 = np.max(np.nonzero(x_))
        y0 = np.min(np.nonzero(y_))
        y1 = np.max(np.nonzero(y_))
        return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


    def load_mixformer(self):
        refine_model_name = 'ARcm_coco_seg_only_mask_384'
        params = vot_params.parameters("baseline_large", model="mixformerL_online_22k.pth.tar")
        params.debug = False
        mixformer = MixFormerOnline(params, "VOT20")
        # mixformer.eval()
        return mixformer


class AOTTracker(object):
    def __init__(self, cfg, gpu_id):
        self.with_crop = False
        self.EXPAND_SCALE = None
        self.small_ratio = 12
        self.mid_ratio = 100
        self.large_ratio = 0.5
        self.AOT_INPUT_SIZE = (465, 465)
        self.cnt = 2
        self.gpu_id = gpu_id
        self.model = build_vos_model(cfg.MODEL_VOS, cfg).cuda(gpu_id)
        print('cfg.TEST_CKPT_PATH = ', cfg.TEST_CKPT_PATH)
        self.model, _ = load_network(self.model, cfg.TEST_CKPT_PATH, gpu_id)
        self.engine = build_engine(cfg.MODEL_ENGINE,
                                   phase='eval',
                                   aot_model=self.model,
                                   gpu_id=gpu_id,
                                   short_term_mem_skip=4,
                                   long_term_mem_gap=cfg.TEST_LONG_TERM_MEM_GAP)

        self.transform = transforms.Compose([
        tr.MultiRestrictSize_(cfg.TEST_MAX_SHORT_EDGE,
                                cfg.TEST_MAX_LONG_EDGE, cfg.TEST_FLIP, cfg.TEST_INPLACE_FLIP,
                                cfg.TEST_MULTISCALE, cfg.MODEL_ALIGN_CORNERS),
        tr.MultiToTensor()
        ])  
        self.model.eval()
        # add the first frame and label
    def add_first_frame(self, frame, mask): 

        sample = {
            'current_img': frame,
            'current_label': mask,
            'height':frame.shape[0],
            'weight':frame.shape[1]
        }
        sample = self.transform(sample)
        
        frame = sample[0]['current_img'].unsqueeze(0).float().cuda(self.gpu_id, non_blocking=True)
        mask = sample[0]['current_label'].unsqueeze(0).float().cuda(self.gpu_id, non_blocking=True)

        mask = F.interpolate(mask, size=frame.size()[2:], mode="nearest")

       
        # add reference frame
        self.engine.add_reference_frame(frame, mask, frame_step=0, obj_nums=1)

    
    def track(self, image):
        
        height = image.shape[0]
        width = image.shape[1]
        
           
        sample = {'current_img': image}
        sample['meta'] = {
            'height': height,
            'width': width,
        }
        sample = self.transform(sample)
        output_height = sample[0]['meta']['height']
        output_width = sample[0]['meta']['width']
        image = sample[0]['current_img'].unsqueeze(0).float().cuda(self.gpu_id, non_blocking=True)

        self.engine.match_propogate_one_frame(image)
        pred_logit = self.engine.decode_current_logits((output_height, output_width))
        pred_prob = torch.softmax(pred_logit, dim=1)
        pred_label = torch.argmax(pred_prob, dim=1,
                                    keepdim=True).float()

        _pred_label = F.interpolate(pred_label,
                                    size=self.engine.input_size_2d,
                                    mode="nearest")
        conf = torch.sum(pred_prob[:, 1, :, :] * pred_label) / torch.sum(pred_label)
        conf = torch.nan_to_num(conf, nan=0)
        # update memory
        # if have object and conf > 0.7, update memory
        is_valid = torch.sum(_pred_label > 0) and conf > 0.7
        self.engine.update_memory(_pred_label, is_valid)

        mask = pred_label.detach().cpu().numpy()[0][0].astype(np.uint8)
        return mask, conf


def make_full_size(x, output_sz):
    '''
    zero-pad input x (right and down) to match output_sz
    x: numpy array e.g., binary mask
    output_sz: size of the output [width, height]
    '''
    if x.shape[0] == output_sz[1] and x.shape[1] == output_sz[0]:
        return x
    pad_x = output_sz[0] - x.shape[1]
    if pad_x < 0:
        x = x[:, :x.shape[1] + pad_x]
        # padding has to be set to zero, otherwise pad function fails
        pad_x = 0
    pad_y = output_sz[1] - x.shape[0]
    if pad_y < 0:
        x = x[:x.shape[0] + pad_y, :]
        # padding has to be set to zero, otherwise pad function fails
        pad_y = 0
    return np.pad(x, ((0, pad_y), (0, pad_x)), 'constant', constant_values=0)


def read_img(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def _rect_from_mask(mask):
    if len(np.where(mask==1)[0]) == 0:
        return None
    x_ = np.sum(mask, axis=0)
    y_ = np.sum(mask, axis=1)
    x0 = np.min(np.nonzero(x_))
    x1 = np.max(np.nonzero(x_))
    y0 = np.min(np.nonzero(y_))
    y1 = np.max(np.nonzero(y_))
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]

def select_tracker(img, mask):
    img_sz = img.shape[0] * img.shape[1]
    _, _, w, h = _rect_from_mask(mask)
    max_edge = max(w, h)
    rect_sz = max_edge * max_edge
    ratio = img_sz / rect_sz
    print("ratio = {ratio}")
    if ratio > 900:
        return "aot_mix"
    else:
        return "aot"

class MSAOTTracker(object):
    def __init__(self, cfg, config):
        self.aot_tracker = AOTTracker(cfg, config['gpu_id'])
        self.aot_mix_tracker = AOTMixFormerTracker()
        self.tracker = self.aot_tracker
        self.mask_size = None

    def initialize(self, image, mask):
        tracker_name = select_tracker(image, mask)
        if tracker_name == "aot":
            self.tracker = self.aot_tracker
            self.tracker.add_first_frame(image, mask)
            del self.aot_mix_tracker
            self.aot_mix_tracker = None
        else:
            self.tracker = self.aot_mix_tracker
            self.tracker.initialize(image, mask)
            del self.aot_tracker
            self.aot_tracker = None
        self.mask_size = mask.shape

    def track(self, image):
        m, confidence = self.tracker.track(image)
        m = F.interpolate(torch.tensor(m)[None, None, :, :],
                          size=self.mask_size, mode="nearest").numpy().astype(np.uint8)[0][0]
        return m, confidence

#####################
# config
#####################

config = {
    'exp_name': 'default',
    'model': 'R50_AOTv3',
    'pretrain_model_path': 'pretrain_models/ms_aot_model.pth',
    'gpu_id': 0,
}

# set cfg
engine_config = importlib.import_module('configs.' + 'ms_aot')
cfg = engine_config.EngineConfig(config['exp_name'], config['model'])
cfg.TEST_CKPT_PATH = os.path.join(AOT_PATH, config['pretrain_model_path'])


### init trackers
tracker = MSAOTTracker(cfg, config)

# get first frame and mask
handle = vot_utils.VOT("mask")
selection = handle.region()
imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

# get first frame and mask
image = read_img(imagefile)
mask = make_full_size(selection, (image.shape[1], image.shape[0]))
mask = (mask > 0).astype(np.uint8)

# initialize tracker
tracker.initialize(image, mask)

while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    image = read_img(imagefile)
    m, confidence = tracker.track(image)
    handle.report(m, confidence)
