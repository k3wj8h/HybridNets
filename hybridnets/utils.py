import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.ops.boxes import batched_nms

import cv2
import os
import re
import yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob

from hybridnets.backbone import HNBackBone

class BBoxTransform(nn.Module):

	def forward(self, anchors, regression):
		y_centers_a = (anchors[..., 0] + anchors[..., 2]) / 2
		x_centers_a = (anchors[..., 1] + anchors[..., 3]) / 2
		ha = anchors[..., 2] - anchors[..., 0]
		wa = anchors[..., 3] - anchors[..., 1]

		w = regression[..., 3].exp() * wa
		h = regression[..., 2].exp() * ha

		y_centers = regression[..., 0] * ha + y_centers_a
		x_centers = regression[..., 1] * wa + x_centers_a

		ymin = y_centers - h / 2.
		xmin = x_centers - w / 2.
		ymax = y_centers + h / 2.
		xmax = x_centers + w / 2.

		return torch.stack([xmin, ymin, xmax, ymax], dim=2)


class ClipBoxes(nn.Module):

	def __init__(self):
		super(ClipBoxes, self).__init__()

	def forward(self, boxes, img):
		batch_size, num_channels, height, width = img.shape

		boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
		boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

		boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width - 1)
		boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height - 1)

		return boxes


def postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold):
	transformed_anchors = regressBoxes(anchors, regression)
	transformed_anchors = clipBoxes(transformed_anchors, x)
	scores = torch.max(classification, dim=2, keepdim=True)[0]
	scores_over_thresh = (scores > threshold)[:, :, 0]
	
	out = []
	for i in range(x.shape[0]):
		if scores_over_thresh[i].sum() == 0:
			out.append({'rois': np.array(()), 'class_ids': np.array(()), 'scores': np.array(())})
			continue

		classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
		transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
		scores_per = scores[i, scores_over_thresh[i, :], ...]
		scores_, classes_ = classification_per.max(dim=0)
		anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=iou_threshold)

		if anchors_nms_idx.shape[0] != 0:
			classes_ = classes_[anchors_nms_idx]
			scores_ = scores_[anchors_nms_idx]
			boxes_ = transformed_anchors_per[anchors_nms_idx, :]

			out.append({'rois': boxes_.cpu().numpy(), 'class_ids': classes_.cpu().numpy(), 'scores': scores_.cpu().numpy()})
		else:
			out.append({'rois': np.array(()), 'class_ids': np.array(()), 'scores': np.array(())})

	return out


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
	if len(coords) == 0:
		return []
	# Rescale coords (xyxy) from img1_shape to img0_shape
	if ratio_pad is None:  # calculate from img0_shape
		gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
		pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
	else:
		gain = ratio_pad[0][0]
		pad = ratio_pad[1]

	coords[:, [0, 2]] -= pad[0]  # x padding
	coords[:, [1, 3]] -= pad[1]  # y padding
	coords[:, :4] /= gain
	clip_coords(coords, img0_shape)
	return coords


def clip_coords(boxes, shape):
	# Clip bounding xyxy bounding boxes to image shape (height, width)
	if isinstance(boxes, torch.Tensor):  # faster individually
		boxes[:, 0].clamp_(0, shape[1])  # x1
		boxes[:, 1].clamp_(0, shape[0])  # y1
		boxes[:, 2].clamp_(0, shape[1])  # x2
		boxes[:, 3].clamp_(0, shape[0])  # y2
	else:  # np.array (faster grouped)
		boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
		boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def plot_one_box(img, coord, label=None, score=None, color=None, line_thickness=None):
	tl = line_thickness or int(round(0.001 * max(img.shape[0:2])))  # line thickness
	c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
	cv2.rectangle(img, c1, c2, color, thickness=tl)
	if label:
		tf = max(tl - 2, 1)  # font thickness
		s_size = cv2.getTextSize(str('{:.0%}'.format(score)), 0, fontScale=float(tl) / 3, thickness=tf)[0]
		t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
		c2 = c1[0] + t_size[0] + s_size[0] + 15, c1[1] - t_size[1] - 3
		cv2.rectangle(img, c1, c2, color, -1)  # filled
		cv2.putText(img, '{}: {:.0%}'.format(label, score), (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0], thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)


def save_checkpoint(model_with_loss, saved_path, name, optimizer=None, scheduler=None, step=None, results=None):
	ckpt_obj = {}
	ckpt_obj['model'] = model_with_loss.model.state_dict()
	ckpt_obj['optimizer'] = optimizer.state_dict() if optimizer else None
	ckpt_obj['scheduler'] = scheduler.state_dict() if scheduler else None
	ckpt_obj['step'] = step if step else 0
	ckpt_obj['results'] = results.to_json(orient='columns') if isinstance(results, pd.DataFrame) else None
	filename = os.path.join(saved_path, name)
	torch.save(ckpt_obj, filename)


def save_image(image, filename, image_path='./sample_images/', figsize=(15,9)):
	os.makedirs(image_path, exist_ok=True)

	fig, ax = plt.subplots(1,1,figsize=figsize)
	ax.imshow(image, interpolation='nearest', aspect='auto')
	filename = os.path.join(image_path, filename)
	fig.savefig(filename)
	plt.close()


def apply_model(checkpoint_name, model=None, num_of_images=1, param_file='./hybridnets/hybridnets.yml', dataset='val'):
	params = yaml.safe_load(open(param_file).read())

	if model is None:
		model = HNBackBone(num_classes=len(params['categories']), ratios=params['anchor_ratios'], scales=params['anchor_scales'], seg_classes=len(params['seg_list']))

	####################
	# load model state #
	####################
	checkpoint = torch.load(checkpoint_name, map_location='cuda')
	model.load_state_dict(checkpoint['model'])
	model.requires_grad_(False)
	model.eval()
	model = model.cuda()

	####################
	# transform images #
	####################
	imgs = glob(f'{params["img_root"]}/{dataset}/*.jpg')
	input_imgs = []
	shapes = []
	det_only_imgs = []

	# list of images
	ori_imgs = [cv2.imread(i, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION) for i in imgs[:num_of_images]]
	ori_imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in ori_imgs]

	# resize images
	resized_shape = max(params['input_size'])
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=params['rgb_mean'] ,std=params['rgb_std'])])

	for ori_img in ori_imgs:
		h0, w0 = ori_img.shape[:2]
		r = resized_shape / max(h0, w0)
		input_img = cv2.resize(ori_img, (int(w0*r), int(h0*r)), interpolation=cv2.INTER_AREA)
		h, w = input_img.shape[:2]

		# letterbox
		r = min(resized_shape/h, resized_shape/w)
		dw, dh = np.mod(resized_shape-int(round(w*r)), 32) / 2, np.mod(resized_shape-int(round(h*r)), 32) / 2

		# add border
		top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
		left, right = int(round(dw-0.1)), int(round(dw+0.1))
		input_img = cv2.copyMakeBorder(input_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))

		input_imgs.append(input_img)
		shapes.append(((h0, w0), ((h/h0, w/w0), (dw, dh))))

	x = torch.stack([transform(fi).cuda() for fi in input_imgs], 0).to(torch.float32)

	with torch.no_grad():
		features, regression, classification, anchors, seg = model(x)

		seg = seg[:, :, 12:372, :]
		da_seg_mask = torch.nn.functional.interpolate(seg, size=[720, 1280], mode='nearest')
		_, da_seg_mask = torch.max(da_seg_mask, 1)
		
		for i in range(da_seg_mask.size(0)):
			da_seg_mask_ = da_seg_mask[i].squeeze().cpu().numpy().round()
			
			color_area = np.zeros((da_seg_mask_.shape[0], da_seg_mask_.shape[1], 3), dtype=np.uint8)
			color_area[da_seg_mask_ == 1] = [0, 255, 0]
			color_area[da_seg_mask_ == 2] = [0, 0, 255]
			color_seg = color_area[..., ::-1]
			

			color_mask = np.mean(color_seg, 2)
			det_only_imgs.append(ori_imgs[i].copy())
			seg_img = ori_imgs[i]
			seg_img[color_mask != 0] = seg_img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
			seg_img = seg_img.astype(np.uint8)

		regressBoxes = BBoxTransform()
		clipBoxes = ClipBoxes()
		out = postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, 0.25, 0.3)

		for i in range(len(ori_imgs[:num_of_images])):
			out[i]['rois'] = scale_coords(ori_imgs[i][:2], out[i]['rois'], shapes[i][0], shapes[i][1])
			for j in range(len(out[i]['rois'])):
				x1, y1, x2, y2 = out[i]['rois'][j].astype(int)
				obj = params['categories'][out[i]['class_ids'][j]]
				score = float(out[i]['scores'][j])
				plot_one_box(ori_imgs[i], [x1, y1, x2, y2], color=((255,255,0)), label=obj, score=score, line_thickness=2)

			epoch = re.findall(r'\d+',checkpoint_name)[0]
			save_image(image=ori_imgs[i], filename=f"val{i+1}_e{epoch}.png")


