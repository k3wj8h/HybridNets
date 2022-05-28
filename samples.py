import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import yaml
import glob
import cv2
import re
import os

import torch.nn as nn
from torchvision.ops.boxes import batched_nms
import torchvision.transforms as transforms

from hybridnets.backbone import HNBackBone
from hybridnets.utils import BBoxTransform, ClipBoxes, postprocess, scale_coords, clip_coords, plot_one_box


def get_args():
	parser = argparse.ArgumentParser('Result printer')
	parser.add_argument('--checkpoint', type=str, default=None, help='Load a saved checkpoint')
	parser.add_argument('--image_path', type=str, default='./sample_images', help='Path to save images')
	parser.add_argument('--num_of_images', type=int, default=1, help='Number of sample images')
	parser.add_argument('--param_file', type=str, default='./hybridnets/hybridnets.yml', help='Parameter yaml file')
	parser.add_argument('--dataset', type=str, default='val', help='Dataset train or val')
	args = parser.parse_args()
	return args


def apply_model(checkpoint_name, num_of_images, param_file, dataset):
	
	params = yaml.safe_load(open(param_file).read())
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
	imgs = glob.glob(f'{params["img_root"]}/{dataset}/*.jpg')
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
		da_seg_mask = torch.nn.functional.interpolate(seg, size=params['orig_image_size'], mode='nearest')
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
			fig, ax = plt.subplots(1,1,figsize=(15,9))
			ax.imshow(ori_imgs[i], interpolation='nearest', aspect='auto')
			filename = os.path.join(params['img_path'], f"{dataset}{i+1}_e{epoch}.png")
			fig.savefig(filename)
			plt.close()
			print(f'{filename} saved')


if __name__ == '__main__':
	opt = get_args()
	
	apply_model(checkpoint_name=opt.checkpoint, num_of_images=opt.num_of_images, param_file=opt.param_file, dataset=opt.dataset)
