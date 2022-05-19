import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.ops.boxes import batched_nms

import os
import yaml
import math
import datetime
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.autonotebook import tqdm

from hybridnets.dataset import BDD100K, BatchGenerator
from hybridnets.backbone import HNBackBone
from hybridnets.loss import ModelWithLoss


def get_args():
	parser = argparse.ArgumentParser('HybridNets: End-to-End Perception Network - DatVu')
	parser.add_argument('--num_epochs', type=int, default=5)
	parser.add_argument('--num_gpus', type=int, default=1)
	parser.add_argument('--load_checkpoint', type=str, default=None, help='Load previous checkpoint, set None to initialize')
	parser.add_argument('--num_sample_images', type=int, default=0, help='Number of generated sample images')
	parser.add_argument('--download', type=boolean_string, default=False, help='Download checkpoints and sample images')
	args = parser.parse_args()
	return args

def train(num_gpus=1, num_epochs=5, load_checkpoint=None, num_sample_images=0, download=False, param_file='./hybridnets/hybridnets.yml'):
	
	params = yaml.safe_load(open(param_file).read())
	print(f'params: {params}')
	
	if torch.cuda.is_available():
		torch.cuda.manual_seed(42)
	else:
		torch.manual_seed(42)

	os.makedirs(params['save_path'], exist_ok=True)
	os.makedirs(params['img_path'], exist_ok=True)
	
	############
	# datasets #
	############
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=params['rgb_mean'], std=params['rgb_mean'])])

	# train data
	train_dataset = BDD100K(is_train=True, inputsize=params['input_size'], transform=transform, param_file=param_file)
	train_generator = BatchGenerator(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=2, pin_memory=False, collate_fn=BDD100K.collate_fn)

	# val data
	val_dataset = BDD100K(is_train=False, inputsize=params['input_size'], transform=transform, param_file=param_file)
	val_generator = BatchGenerator(val_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=2, pin_memory=False, collate_fn=BDD100K.collate_fn)
	
	for i in range(opt.num_sample_images):
		save_image(image=plt.imread(val_dataset.__getitem__(i)[1]), image_path=params['img_path'], filename=f"val{i+1}_e0.png", download=False)

	
	#########
	# model #
	#########
	model = HNBackBone(num_classes=len(params['categories']), compound_coef=params['compound_coef'], ratios=params['anchor_ratios'], scales=params['anchor_scales'], seg_classes=len(params['seg_list']))

	ckpt = {}
	if load_checkpoint:
		try:
			ckpt = torch.load(load_checkpoint)
			model.load_state_dict(ckpt['model'])
		except RuntimeError as e:
			print(f'[Warning] Ignoring {e}')
			print('[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')
	else:
		print('[Info] initializing weights...')
		init_weights(model)

	print('[Info] Successfully!!!')

	model = ModelWithLoss(model).cuda()
	optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

	if load_checkpoint is not None and ckpt.get('optimizer', None):
		optimizer.load_state_dict(ckpt['optimizer'])

	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

	# load loss data
	results = pd.DataFrame(columns=['Dataset','Step','Epoch','Loss','Regression_loss','Classification_loss','Segmentation_loss','Learning_rate','Timestamp'])
	if load_checkpoint and ckpt.get('results', None) is not None:
		try:
			print('[Info] Load results...')
			results = pd.read_json(ckpt['results'], orient='columns')
		except ValueError as e:
			print('[Warning] Load results failed: there is no result saved.')  
	#display(results.tail(5))


	epoch = 0
	best_loss = 1e5
	best_epoch = 0
	last_step = ckpt['step'] if load_checkpoint is not None and ckpt.get('step', None) else 0
	step = max(0, last_step)
	model.train()

	num_iter_per_epoch = len(train_generator)

	print(f'step: {step}, last_step: {last_step}, num_iter_per_epoch: {num_iter_per_epoch}')

	try:
		for epoch in range(1, num_epochs+1):
			last_epoch = step // num_iter_per_epoch
			if epoch < last_epoch:
				continue

			epoch_loss = []
			progress_bar = tqdm(train_generator)
			for iter, data in enumerate(progress_bar):
				if iter < step - last_epoch * num_iter_per_epoch:
					progress_bar.update()
					continue
				try:
					imgs = data['img'].cuda()
					annot = data['annot'].cuda()
					seg_annot = data['segmentation'].cuda().long()

					optimizer.zero_grad()
					cls_loss, reg_loss, seg_loss, regression, classification, anchors, segmentation = model(imgs, annot, seg_annot, obj_list=params['categories'])
					cls_loss = cls_loss.mean()
					reg_loss = reg_loss.mean()
					seg_loss = seg_loss.mean()

					loss = cls_loss + reg_loss + seg_loss
					if loss == 0 or not torch.isfinite(loss):
						continue

					loss.backward()
					optimizer.step()
					epoch_loss.append(float(loss))

					progress_bar.set_description(f'Step: {step}. Epoch: {epoch}/{num_epochs}. Iteration: {iter + 1}/{num_iter_per_epoch}. Cls loss: {cls_loss.item():.5f}. Reg loss: {reg_loss.item():.5f}. Seg loss: {seg_loss.item():.5f}. Total loss: {loss.item():.5f}')

					# log learning_rate
					current_lr = optimizer.param_groups[0]['lr']
					results = results.append({'Dataset':'Train',
												'Step':step,
												'Epoch':epoch,
												'Loss':float(loss.cpu().detach().numpy()),
												'Regression_loss':float(reg_loss.cpu().detach().numpy()),
												'Classification_loss':float(cls_loss.cpu().detach().numpy()),
												'Segmentation_loss':float(seg_loss.cpu().detach().numpy()),
												'Learning_rate':current_lr,
												'Timestamp':datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}, ignore_index=True)

					step += 1

				except Exception as e:
					print('[Error]', traceback.format_exc())
					print(e)
					continue

			scheduler.step(np.mean(epoch_loss))

			results = val(model, optimizer, val_generator, results, epoch, num_epochs, step, num_sample_images, download=download)
	except KeyboardInterrupt:
		save_checkpoint(model, params['save_path'], f'hybridnet_e{epoch}_s{step}.ckpt', optimizer=optimizer, step=step, results=results, download=download)


def init_weights(model):
	for name, module in model.named_modules():
		if isinstance(module, nn.Conv2d):
			if "conv_list" or "header" in name:
				fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight.data)
				std = math.sqrt(1/float(fan_in))
				nn.init._no_grad_normal_(module.weight.data, 0., std)
			else:
				nn.init.kaiming_uniform_(module.weight.data)

			if module.bias is not None:
				if "classifier.header" in name:
					bias_value = -np.log((1 - 0.01) / 0.01)
					torch.nn.init.constant_(module.bias, bias_value)
				else:
					module.bias.data.zero_()
					
					
def save_checkpoint(model_with_loss, saved_path, name, optimizer=None, step=None, results=None, download=False):
	ckpt_obj = {}
	ckpt_obj['model'] = model_with_loss.model.state_dict()
	ckpt_obj['optimizer'] = optimizer.state_dict() if optimizer else None
	ckpt_obj['step'] = step if step else 0
	ckpt_obj['results'] = results.to_json(orient='columns') if isinstance(results, pd.DataFrame) else None
	filename = os.path.join(saved_path, name)
	torch.save(ckpt_obj, filename)

	if download:
		files.download(filename)
	

def save_image(image, filename, image_path, download=False, figsize=(15,9)):
	os.makedirs(image_path, exist_ok=True)

	fig, ax = plt.subplots(1,1,figsize=figsize)
	ax.imshow(image, interpolation='nearest', aspect='auto')
	filename = os.path.join(image_path, filename)
	fig.savefig(filename)
	plt.close()

	if download:
		files.download(filename)


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
		
		
@torch.no_grad()
def val(model, optimizer, val_generator, results, epoch, num_epochs, step, num_sample_images, download=False, param_file='./hybridnets/hybridnets.yml'):
	model.eval()
	loss_regression_ls = []
	loss_classification_ls = []
	loss_segmentation_ls = []

	regressBoxes = BBoxTransform()
	clipBoxes = ClipBoxes()
	
	params = yaml.safe_load(open(param_file).read())
	
	val_loader = tqdm(val_generator)
	for iter, data in enumerate(val_loader):
		imgs = data['img'].cuda()
		annot = data['annot'].cuda()
		seg_annot = data['segmentation'].cuda()
		filenames = data['filenames']
		shapes = data['shapes']

		cls_loss, reg_loss, seg_loss, regression, classification, anchors, segmentation = model(imgs, annot, seg_annot, obj_list=params['categories'])
		cls_loss = cls_loss.mean()
		reg_loss = reg_loss.mean()
		seg_loss = seg_loss.mean()

		loss = cls_loss + reg_loss + seg_loss
		if loss == 0 or not torch.isfinite(loss):
			continue

		loss_classification_ls.append(cls_loss.item())
		loss_regression_ls.append(reg_loss.item())
		loss_segmentation_ls.append(seg_loss.item())

	cls_loss = np.mean(loss_classification_ls)
	reg_loss = np.mean(loss_regression_ls)
	seg_loss = np.mean(loss_segmentation_ls)
	loss = cls_loss + reg_loss + seg_loss


	print(f'Val. Epoch: {epoch}/{num_epochs}. Classification loss: {cls_loss:1.5f}. Regression_loss: {reg_loss:1.5f}. Segmentation loss: {seg_loss:1.5f}. Total loss: {loss:1.5f}')
	results = results.append({'Dataset':'Val',
							  'Step':step,
							  'Epoch':epoch,
							  'Loss':float(loss),
							  'Regression_loss':float(reg_loss),
							  'Classification_loss':float(cls_loss),
							  'Segmentation_loss':float(seg_loss),
							  'Learning_rate':None,
							  'Timestamp':datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}, ignore_index=True)
	
	save_checkpoint(model_with_loss=model, saved_path=params['save_path'], name=f'hybridnet_e{epoch}.ckpt', optimizer=optimizer, step=step, results=results, download=download)
	try:
		from google.colab import files
		ckpt_name = os.path.join(params['save_path'], f'hybridnet_e{epoch}.ckpt')
		apply_model(img_path=f'{params["img_root"]}/val/*.jpg', checkpoint_name=ckpt_name, num_of_images=num_sample_images, params=params)
	except:
		print('Failed to download files')
	model.train()
	return results
	
	
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
		
		
def apply_model(img_path, checkpoint_name, model=None, num_of_images=0, params=None, download=False):
	if model is None:
		model = HNBackBone(num_classes=len(params['categories']), compound_coef=params['compound_coef'], ratios=params['anchor_ratios'], scales=params['anchor_scales'], seg_classes=len(params['seg_list']))

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
	imgs = glob(img_path)
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
			save_image(image=ori_imgs[i], image_path=params['img_path'], filename=f"val{i+1}_e{epoch}.png", download=download)
			

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'
	

if __name__ == '__main__':
	opt = get_args()
	
	train(num_gpus=opt.num_gpus, num_epochs=opt.num_epochs, load_checkpoint=opt.load_checkpoint, num_sample_images=opt.num_sample_images, download=opt.download)