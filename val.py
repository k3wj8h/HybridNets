import torch
import numpy as np
import argparse
from tqdm.autonotebook import tqdm
import os

from hybridnets import smp_metrics
from hybridnets.backbone import *
from hybridnets.dataset import *
from hybridnets.utils import *
from torchvision import transforms


@torch.no_grad()
def val_from_cmd(model, val_generator, params, opt):
	model.eval()
	jdict, stats, ap, ap_class = [], [], [], []
	iou_thresholds = torch.linspace(0.5, 0.95, 10).cuda()  # iou vector for mAP@0.5:0.95
	num_thresholds = iou_thresholds.numel()
	names = {i: v for i, v in enumerate(params['categories'])}
	nc = len(names)
	seen = 0
	s = ('%15s' + '%11s' * 14) % (
	'Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95', 'mIoU', 'mF1', 'fIoU', 'sIoU', 'rIoU', 'rF1', 'lIoU', 'lF1')
	dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	iou_ls = [[] for _ in range(3)]
	f1_ls = [[] for _ in range(3)]
	regressBoxes = BBoxTransform()
	clipBoxes = ClipBoxes()

	val_loader = tqdm(val_generator)	
	for iter, data in enumerate(val_loader):
		imgs = data['img']
		annot = data['annot']
		seg_annot = data['segmentation']
		filenames = data['filenames']
		shapes = data['shapes']

		if opt.num_gpus == 1:
			imgs = imgs.cuda()
			annot = annot.cuda()
			seg_annot = seg_annot.cuda()

		features, regressions, classifications, anchors, segmentation = model(imgs)

		out = postprocess(imgs.detach(),
						  torch.stack([anchors[0]] * imgs.shape[0], 0).detach(), regressions.detach(),
						  classifications.detach(),
						  regressBoxes, clipBoxes,
						  0.001, 0.6)  # 0.5, 0.3

		# imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
		# imgs = ((imgs * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255).astype(np.uint8)
		# imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs]
		# display(out, imgs, ['car'], imshow=False, imwrite=True)

		# for index, filename in enumerate(filenames):
		#   ori_img = cv2.imread('datasets/bdd100k/val/'+filename)
		#   if len(out[index]['rois']):
		#	 for roi in out[index]['rois']:
		#	   x1,y1,x2,y2 = [int(x) for x in roi]
		#	   cv2.rectangle(ori_img, (x1,y1), (x2,y2), (255,0,0), 1)
		#   cv2.imwrite(filename, ori_img)

		for i in range(annot.size(0)):
			seen += 1
			labels = annot[i]
			labels = labels[labels[:, 4] != -1]

			ou = out[i]
			nl = len(labels)

			pred = np.column_stack([ou['rois'], ou['scores']])
			pred = np.column_stack([pred, ou['class_ids']])
			pred = torch.from_numpy(pred).cuda()

			target_class = labels[:, 4].tolist() if nl else []  # target class

			if len(pred) == 0:
				if nl:
					stats.append((torch.zeros(0, num_thresholds, dtype=torch.bool),
								  torch.Tensor(), torch.Tensor(), target_class))
				# print("here")
				continue

			if nl:
				pred[:, :4] = scale_coords(imgs[i][1:], pred[:, :4], shapes[i][0], shapes[i][1])

				labels = scale_coords(imgs[i][1:], labels, shapes[i][0], shapes[i][1])

				# ori_img = cv2.imread('datasets/bdd100k_effdet/val/' + filenames[i],
				#					  cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_UNCHANGED)
				# for label in labels:
				#	 x1, y1, x2, y2 = [int(x) for x in label[:4]]
				#	 ori_img = cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
				# for pre in pred:
				#	 x1, y1, x2, y2 = [int(x) for x in pre[:4]]
				#	 # ori_img = cv2.putText(ori_img, str(pre[4].cpu().numpy()), (x1 - 10, y1 - 10),
				#	 #					   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
				#	 ori_img = cv2.rectangle(ori_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

				# cv2.imwrite('pre+label-{}.jpg'.format(filenames[i]), ori_img)
				correct = process_batch(pred, labels, iou_thresholds)

			else:
				correct = torch.zeros(pred.shape[0], num_thresholds, dtype=torch.bool)
			stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), target_class))

			# print(stats)

			# Visualization
			# seg_0 = segmentation[i]
			# # print('bbb', seg_0.shape)
			# seg_0 = torch.argmax(seg_0, dim = 0)
			# # print('before', seg_0.shape)
			# seg_0 = seg_0.cpu().numpy()
			#	 #.transpose(1, 2, 0)
			# # print(seg_0.shape)
			# anh = np.zeros((384,640,3))
			# anh[seg_0 == 0] = (255,0,0)
			# anh[seg_0 == 1] = (0,255,0)
			# anh[seg_0 == 2] = (0,0,255)
			# anh = np.uint8(anh)
			# cv2.imwrite('segmentation-{}.jpg'.format(filenames[i]),anh)
			
		# Convert segmentation tensor --> 3 binary 0 1
		# batch_size, num_classes, height, width
		_, segmentation = torch.max(segmentation, 1)
#		 _, seg_annot = torch.max(seg_annot, 1)
		seg = torch.zeros((seg_annot.size(0), 3, 384, 640), dtype=torch.int32)
		seg[:, 0, ...][segmentation == 0] = 1
		seg[:, 1, ...][segmentation == 1] = 1
		seg[:, 2, ...][segmentation == 2] = 1

		tp_seg, fp_seg, fn_seg, tn_seg = smp_metrics.get_stats(seg.cuda(), seg_annot.long().cuda(), mode='multilabel',
															   threshold=None)

		iou = smp_metrics.iou_score(tp_seg, fp_seg, fn_seg, tn_seg, reduction='none')
		#		 print(iou)
		f1 = smp_metrics.balanced_accuracy(tp_seg, fp_seg, fn_seg, tn_seg, reduction='none')

		for i in range(len(params['seg_list']) + 1):
			print(f'{iter} - iou.T[i].detach().cpu().numpy(): {iou.T[i].detach().cpu().numpy()}')
			iou_ls[i].append(iou.T[i].detach().cpu().numpy())
			f1_ls[i].append(f1.T[i].detach().cpu().numpy())

		# Visualize
		# for i in range(segmentation.size(0)):
		#	 if iou_ls[1][iter][i] < 0.4:
		#		 import cv2
		#
		#		 ori = cv2.imread('datasets/bdd100k/val/{}'.format(filenames[i]))
		#		 cv2.imwrite('ori-segmentation-{}-{}.jpg'.format(iter,filenames[i]),ori)
		#
		#		 gt = seg_annot[i].detach()
		#		 gt = torch.argmax(gt, dim = 0).cpu().numpy()
		#
		#		 anh = np.zeros((384,640,3))
		#		 anh[gt == 0] = (255,0,0)
		#		 anh[gt == 1] = (0,255,0)
		#		 anh[gt == 2] = (0,0,255)
		#		 cv2.imwrite('gt-segmentation-{}-{}.jpg'.format(iter,filenames[i]),anh)
		#
		#		 seg_0 = seg[i]
		#		 seg_0 = torch.argmax(seg_0, dim = 0)
		#		 seg_0 = seg_0.cpu().numpy()
		#		 anh = np.zeros((384,640,3))
		#		 anh[seg_0 == 0] = (255,0,0)
		#		 anh[seg_0 == 1] = (0,255,0)
		#		 anh[seg_0 == 2] = (0,0,255)
		#		 anh = np.uint8(anh)
		#		 cv2.imwrite('segmentation-{}-{}.jpg'.format(iter,filenames[i]),anh)

	# print(len(iou_ls[0]))
	# print(iou_ls)
	iou_score = np.mean(iou_ls)
	# print(iou_score)
	f1_score = np.mean(f1_ls)

	iou_first_decoder = iou_ls[0] + iou_ls[1]
	iou_first_decoder = np.mean(iou_first_decoder)

	iou_second_decoder = iou_ls[0] + iou_ls[2]
	iou_second_decoder = np.mean(iou_second_decoder)
	
	print('ckpt1')

	for i in range(len(params['seg_list']) + 1):
		iou_ls[i] = np.mean(iou_ls[i])
		f1_ls[i] = np.mean(f1_ls[i])

	print('ckpt2')
	
	# Compute statistics
	stats = [np.concatenate(x, 0) for x in zip(*stats)]

	# Count detected boxes per class
	# boxes_per_class = np.bincount(stats[2].astype(np.int64), minlength=1)
	
	print('ckpt3')
	
	ap50 = None
	save_dir = 'plots'
	os.makedirs(save_dir, exist_ok=True)

	# Compute metrics
	if len(stats) and stats[0].any():
		p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir=save_dir, names=names)
		ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
		mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
		nt = np.bincount(stats[3].astype(np.int64), minlength=1)  # number of targets per class
	else:
		nt = torch.zeros(1)

	print('ckpt4')

	# Print results
	print(f'mAP50: {ap50}, IoU_first: {iou_first_decoder}, iou_second_decoder: {iou_second_decoder}')
	
	print(s)
	pf = '%15s' + '%11i' * 2 + '%11.3g' * 12  # print format
	print(pf % ('all', seen, nt.sum(), mp, mr, map50, map, iou_score, f1_score, iou_first_decoder, iou_second_decoder,
				iou_ls[1], f1_ls[1], iou_ls[2], f1_ls[2]))
				
	

	# Print results per class
	training = False
	if (True or (nc < 50 and not training)) and nc > 1 and len(stats):
		pf = '%15s' + '%11i' * 2 + '%11.3g' * 4
		for i, c in enumerate(ap_class):
			print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument('--batch_size', type=int, default=10, help='The number of images per batch among all devices')
	ap.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to be used (0 to use CPU)')
	ap.add_argument('--checkpoint', type=str, help='Path of checkpoint file')
	ap.add_argument('--param_file', type=str, default='./hybridnets/hybridnets.yml', help='Path of parameter file')
	args = ap.parse_args()

	
	params = yaml.safe_load(open(args.param_file).read())
	
	# Validation dataset
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=params['rgb_mean'], std=params['rgb_std'])])

	val_dataset = BDD100K(is_train=False, inputsize=params['input_size'], transform=transform, param_file=args.param_file)
	val_generator = BatchGenerator(val_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=2, pin_memory=False, collate_fn=BDD100K.collate_fn)
	
	# Model
	model = HNBackBone(num_classes=len(params['categories']), ratios=params['anchor_ratios'], scales=params['anchor_scales'], seg_classes=len(params['seg_list']))

	ckpt = torch.load(args.checkpoint)
	model.load_state_dict(ckpt['model'])
	print('[Info] Checkpoint loaded')
	
	optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
	optimizer.load_state_dict(ckpt['optimizer'])
	print('[Info] Optimizer loaded from history')
	model.requires_grad_(False)

	if args.num_gpus > 0:
		model.cuda()

	val_from_cmd(model, val_generator, params, args)
