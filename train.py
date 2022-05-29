
## Import modules
import torch.nn as nn
import torch
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_normal_
import torchvision.transforms as transforms

from tqdm.autonotebook import tqdm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import json
import cv2
import os
import yaml
import datetime
import traceback
import re
import argparse

from hybridnets.loss import *
from hybridnets.network_blocks import *
from hybridnets.dataset import *
from hybridnets.detection_head import *
from hybridnets.segmentation_head import *
from hybridnets.neck import *
from hybridnets.backbone import *
from hybridnets.utils import *


def get_args():
	parser = argparse.ArgumentParser('HybridNets')
	parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
	parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs')
	parser.add_argument('--num_sample_images', type=int, default=3, help='Number of sample images')
	parser.add_argument('--load_checkpoint', type=str, default=None, help='Load previous checkpoint, set None to initialize')
	parser.add_argument('--param_file', type=str, default='./hybridnets/hybridnets.yml', help='Path of parameter file')
	args = parser.parse_args()
	return args


def train(num_gpus=1, num_epochs=10, load_checkpoint=None, num_sample_images=0, param_file='./hybridnets/hybridnets.yml'):

	params = yaml.safe_load(open(param_file).read())
	
	if torch.cuda.is_available():
		torch.cuda.manual_seed(42)
	else:
		torch.manual_seed(42)

	os.makedirs(params['save_path'], exist_ok=True)
	os.makedirs(params['img_path'], exist_ok=True)


	############
	# datasets #
	############
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=params['rgb_mean'], std=params['rgb_std'])])

	# train data
	train_dataset = BDD100K(is_train=True, inputsize=params['input_size'], transform=transform, param_file=param_file)
	train_generator = BatchGenerator(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=2, pin_memory=False, collate_fn=BDD100K.collate_fn)

	# val data
	val_dataset = BDD100K(is_train=False, inputsize=params['input_size'], transform=transform, param_file=param_file)
	val_generator = BatchGenerator(val_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=2, pin_memory=False, collate_fn=BDD100K.collate_fn)


	#########
	# model #
	#########
	model = HNBackBone(num_classes=len(params['categories']), ratios=params['anchor_ratios'], scales=params['anchor_scales'], seg_classes=len(params['seg_list']))

	ckpt = {}
	if load_checkpoint:
		try:
			ckpt = torch.load(load_checkpoint)
			model.load_state_dict(ckpt['model'])
			print('[Info] Checkpoint loaded')
		except RuntimeError as e:
			print(f'[Warning] Ignoring {e}')
			print('[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')
	else:
		init_weights(model)
		print('[Info] Weights initialized')

	model = ModelWithLoss(model).cuda()
	
	# Optimizer
	optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
	if load_checkpoint is not None and ckpt.get('optimizer', None):
		optimizer.load_state_dict(ckpt['optimizer'])
		print('[Info] Optimizer loaded from history')
	
	# Scheduler
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
	if load_checkpoint is not None and ckpt.get('scheduler', None):
		scheduler.load_state_dict(ckpt['scheduler'])
		print('[Info] Scheduler loaded from history')

	# load loss data
	results = pd.DataFrame(columns=['Dataset','Step','Epoch','Loss','Regression_loss','Classification_loss','Segmentation_loss','Learning_rate','Timestamp'])
	if load_checkpoint and ckpt.get('results', None) is not None:
		try:
			results = pd.read_json(ckpt['results'], orient='columns')
			print('[Info] Results loaded from history')
		except ValueError as e:
			print('[Warning] Load results failed: there is no result saved.')

	epoch = 0
	last_step = ckpt['step'] if load_checkpoint is not None and ckpt.get('step', None) else 0
	step = max(0, last_step)
	model.train()

	num_iter_per_epoch = len(train_generator)
	
	print(f'step: {step}, last_step: {last_step}, num_iter_per_epoch: {num_iter_per_epoch}')

	try:
		for epoch in range(1, num_epochs+1):
			last_epoch = step // num_iter_per_epoch
			# find the last complete epoch
			if step - last_epoch * num_iter_per_epoch > 0:
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

			results = val(model, optimizer, scheduler, val_generator, results, epoch, num_epochs, step, num_sample_images, param_file)
	except KeyboardInterrupt:
		save_checkpoint(model, params['save_path'], f'hybridnet_e{epoch}_s{step}.ckpt', optimizer=optimizer, scheduler=scheduler, step=step, results=results)



def init_weights(model):
	for name, module in model.named_modules():
		if isinstance(module, nn.Conv2d):
			if "conv_list" or "header" in name:
				fan_in, fan_out = _calculate_fan_in_and_fan_out(module.weight.data)
				std = math.sqrt(1/float(fan_in))
				_no_grad_normal_(module.weight.data, 0., std)
			else:
				nn.init.kaiming_uniform_(module.weight.data)

			if module.bias is not None:
				if "classifier.header" in name:
					bias_value = -np.log((1 - 0.01) / 0.01)
					torch.nn.init.constant_(module.bias, bias_value)
				else:
					module.bias.data.zero_()




@torch.no_grad()
def val(model, optimizer, scheduler, val_generator, results, epoch, num_epochs, step, num_sample_images, param_file='./hybridnets/hybridnets.yml'):
	params = yaml.safe_load(open(param_file).read())
	
	model.eval()
	loss_regression_ls = []
	loss_classification_ls = []
	loss_segmentation_ls = []

	regressBoxes = BBoxTransform()
	clipBoxes = ClipBoxes()

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
	
	save_checkpoint(model_with_loss=model, saved_path=params['save_path'], name=f'hybridnet_e{epoch}.ckpt', optimizer=optimizer, scheduler=scheduler, step=step, results=results)
	try:
		from google.colab import files
		print('colab.files imported')
		ckpt_name = os.path.join(params['save_path'], f'hybridnet_e{epoch}.ckpt')
		print(f'ckpt_name: {ckpt_name}')
		apply_model(checkpoint_name=ckpt_name, num_of_images=num_sample_images, param_file=param_file, dataset='val')
	except Exception as e:
		print(f'Exception: {e}')
		print('Failed to download files')
		
	model.train()
	return results




#if __name__ == 'main':
opt = get_args()
print('Training started')

train(num_gpus=opt.num_gpus, num_epochs=opt.num_epochs, load_checkpoint=opt.load_checkpoint, num_sample_images=opt.num_sample_images, param_file=opt.param_file)

print('Training finished')