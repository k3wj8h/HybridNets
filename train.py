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
	args = parser.parse_args()
	return args

def train(num_gpus=1, num_epochs=5, load_checkpoint=None, param_file='./hybridnets/hybridnets.yml'):
	
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
			# find the last complete epoch
			if step - last_epoch * num_iter_per_epoch > 0
				last_epoch =- 1
			if epoch <= last_epoch:
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

			results = val(model, optimizer, val_generator, results, epoch, num_epochs, step)
	except KeyboardInterrupt:
		save_checkpoint(model, params['save_path'], f'hybridnet_e{epoch}_s{step}.ckpt', optimizer=optimizer, step=step, results=results)


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
					
					
def save_checkpoint(model_with_loss, saved_path, name, optimizer=None, step=None, results=None):
	ckpt_obj = {}
	ckpt_obj['model'] = model_with_loss.model.state_dict()
	ckpt_obj['optimizer'] = optimizer.state_dict() if optimizer else None
	ckpt_obj['step'] = step if step else 0
	ckpt_obj['results'] = results.to_json(orient='columns') if isinstance(results, pd.DataFrame) else None
	filename = os.path.join(saved_path, name)
	torch.save(ckpt_obj, filename)


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
def val(model, optimizer, val_generator, results, epoch, num_epochs, step, param_file='./hybridnets/hybridnets.yml'):
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
	
	save_checkpoint(model_with_loss=model, saved_path=params['save_path'], name=f'hybridnet_e{epoch}.ckpt', optimizer=optimizer, step=step, results=results)
	model.train()
	
	return results
	

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'
	

if __name__ == '__main__':
	opt = get_args()
	
	train(num_gpus=opt.num_gpus, num_epochs=opt.num_epochs, load_checkpoint=opt.load_checkpoint)
