'''Encode object boxes and labels.

Reference :
  https://github.com/kuangliu/pytorch-retinanet/blob/master/encoder.py
'''

import math
import torch

import config
from utils import meshgrid, box_iou, box_nms, change_box_order

class DataEncoder:
    def __init__(self):
        self.anchor_areas = config.anchor_areas # same with num. of feature maps used to predict
        self.aspect_ratios = config.aspect_ratios
        self.scale_ratios = config.scale_ratios
        self.anchor_wh = self._get_anchor_wh()
        self.num_anchors = len(config.aspect_ratios) * len(config.scale_ratios)
    def _get_anchor_wh(self):
        '''Compute anchor width and height for each feature map.

        Returns:
          anchor_wh: (tensor) anchor wh, sized [#fm, #anchors_per_cell, 2].
        '''
        anchor_wh = []
        for s in self.anchor_areas:
            for ar in self.aspect_ratios:  # w/h = ar
                h = math.sqrt(s/ar)
                w = ar * h
                for sr in self.scale_ratios:  # scale
                    anchor_h = h*sr
                    anchor_w = w*sr
                    anchor_wh.append([anchor_w, anchor_h])
        num_fms = len(self.anchor_areas)
        return torch.Tensor(anchor_wh).view(num_fms, -1, 2)

    def _get_anchor_boxes(self, input_size):
        '''Compute anchor boxes for each feature map.

        Args:
          input_size: (tensor) model input size of (w,h).

        Returns:
          boxes: (list) anchor boxes for each feature map. Each of size [#anchors,4],
                        where #anchors = fmw * fmh * #anchors_per_cell
        '''
        num_fms = len(self.anchor_areas)
        downsample_cnt = 3
        # fm_sizes = [(input_size / pow(2., i + downsample_cnt)).ceil() for i in range(num_fms)]  # p3 -> p7 feature map sizes

        fm_sizes = []
        for i in range(num_fms):
            if i >= 4:
                fm_sizes.append((input_size / pow(2., 3 + downsample_cnt)) - (2. * (i-3)))
            else:
                fm_sizes.append(input_size / pow(2., i + downsample_cnt))

        boxes = []
        for i in range(num_fms):
            fm_size = fm_sizes[i]
            grid_size = input_size / fm_size
            fm_w, fm_h = int(fm_size[0]), int(fm_size[1])
            xy = meshgrid(fm_w,fm_h) + 0.5  # [fm_h*fm_w, 2]
            xy = xy.float()
            xy = (xy*grid_size).view(fm_h,fm_w,1,2).expand(fm_h,fm_w,self.num_anchors,2)
            wh = self.anchor_wh[i].view(1,1,self.num_anchors,2).expand(fm_h,fm_w,self.num_anchors,2)
            box = torch.cat([xy,wh], 3)  # [x,y,w,h]
            boxes.append(box.view(-1,4))
        return torch.cat(boxes, 0)

    def encode(self, boxes, labels, input_size):
        '''Encode target bounding boxes and class labels.

        We obey the Faster RCNN box coder:
          tx = (x - anchor_x) / anchor_w
          ty = (y - anchor_y) / anchor_h
          tw = log(w / anchor_w)
          th = log(h / anchor_h)

        Args:
          boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [#obj, 4].
          labels: (tensor) object class labels, sized [#obj,].
          input_size: (int/tuple) model input size of (w,h).

        Returns:
          loc_targets: (tensor) encoded bounding boxes, sized [#anchors,4].
          cls_targets: (tensor) encoded class labels, sized [#anchors,].
        '''
        input_size = torch.Tensor([input_size,input_size]) if isinstance(input_size, int) \
                     else torch.Tensor(input_size)
        anchor_boxes = self._get_anchor_boxes(input_size)
        boxes = change_box_order(boxes, 'xyxy2xywh')

        ious = box_iou(anchor_boxes, boxes, order='xywh')
        max_ious, max_ids = ious.max(1)
        boxes = boxes[max_ids]

        loc_xy = (boxes[:,:2]-anchor_boxes[:,:2]) / anchor_boxes[:,2:]
        loc_wh = torch.log(boxes[:,2:]/anchor_boxes[:,2:])
        loc_targets = torch.cat([loc_xy,loc_wh], 1)
        cls_targets = labels[max_ids]

        cls_targets[max_ious<0.5] = 0
        ignore = (max_ious>0.4) & (max_ious<0.5)  # ignore ious between [0.4,0.5]
        #cls_targets[ignore] = 1  # for now just mark ignored to -1
        cls_targets[ignore] = 0
        return loc_targets, cls_targets

    def decode(self, loc_preds, cls_preds, input_size):
        '''Decode outputs back to bouding box locations and class labels.

        Args:
          loc_preds: (tensor) predicted locations, sized [#anchors, 4].
          cls_preds: (tensor) predicted class labels, sized [#anchors, #classes].
          input_size: (int/tuple) model input size of (w,h).

        Returns:
          boxes: (tensor) decode box locations, sized [#obj,4].
          labels: (tensor) class labels for each box, sized [#obj,].
        '''
        CLS_THRESH = config.cls_threshold
        NMS_THRESH = config.nms_threshold

        input_size = torch.Tensor([input_size,input_size]) if isinstance(input_size, int) \
                     else torch.Tensor(input_size)
        anchor_boxes = self._get_anchor_boxes(input_size)

        loc_xy = loc_preds[:,:2]
        loc_wh = loc_preds[:,2:]

        xy = loc_xy * anchor_boxes[:,2:] + anchor_boxes[:,:2]
        wh = loc_wh.exp() * anchor_boxes[:,2:]
        boxes = torch.cat([xy-wh/2, xy+wh/2], 1)  # [#anchors,4]
        """
        cl = cls_preds.sigmoid()
        idd = cl[:, 1] > 0.7
        sum = idd.sum()
        ids = idd == 1
        idss = ids.nonzero().squeeze()
        score, labels = cls_preds.sigmoid().max(1)
        ids = score > CLS_THRESH
        ids = ids.nonzero().squeeze() 
        sum = labels.sum()
        ids = labels == 1
        ids = ids.nonzero().squeeze()             # [#obj,]
        """
        cl = cls_preds.sigmoid()
        #score, labels = cls_preds.sigmoid().max(1)
        ids = cl[:, 1] > CLS_THRESH
        ids = ids == 1
        ids = ids.nonzero().squeeze()
        pre_score = cl[ids,1]
        pre_boxes = boxes[ids]

        if ids.dim() == 0:
            return None
        keep = box_nms(pre_boxes, pre_score, threshold=NMS_THRESH)
        return boxes[ids][keep] #,labels[ids][keep]
