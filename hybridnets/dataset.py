
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import albumentations as A
from pathlib import Path
import numpy as np
import json
import cv2
import yaml

from tqdm.autonotebook import tqdm

# dataset handler object
class BDD100K(Dataset):
	def __init__(self, is_train, inputsize=640, transform=None, param_file='./hybridnets/hybridnets.yml'):
		self.is_train = is_train
		self.inputsize = inputsize
		self.Tensor = transforms.ToTensor()
		self.transform = transform
		
		params = yaml.safe_load(open(param_file).read())

		train_or_val = 'train' if is_train else 'val'
		self.img_root = Path(f'{params["img_root"]}/{train_or_val}')
		self.label_root = Path(f'{params["label_root"]}/{train_or_val}')
		self.darea_root = Path(f'{params["darea_root"]}/{train_or_val}')
		self.lane_root = Path(f'{params["lane_root"]}/{train_or_val}')
		
		self.image_list = self.img_root.iterdir()
		self.categories = params['categories']

		self.flip = True
		self.albumentations_transform = A.Compose([
				A.Blur(p=0.01),
				A.MedianBlur(p=0.01),
				A.ToGray(p=0.01),
				A.CLAHE(p=0.01),
				A.RandomBrightnessContrast(p=0.01),
				A.RandomGamma(p=0.01),
				A.ImageCompression(quality_lower=75, p=0.01)],
				bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']),
				additional_targets={'mask0': 'mask'})

		self.shapes = np.array(params['orig_image_size'])
		self.db = self._get_db()


	def _get_db(self):
		print('buildign database...')
		gt_db = []
		height, width = self.shapes
		for mask in tqdm(list(self.image_list)):
			image_path = str(mask)
			label_path = image_path.replace(str(self.img_root), str(self.label_root)).replace('.jpg', '.json')
			darea_path = image_path.replace(str(self.img_root), str(self.darea_root)).replace('.jpg', '.png')
			lane_path = image_path.replace(str(self.img_root), str(self.lane_root)).replace('.jpg', '.png')
			with open(label_path, 'r') as f:
			  label = json.load(f)

			# filter: only use labels are in self.categories
			data = [obj for obj in label['frames'][0]['objects'] if ('box2d' in obj.keys()) and (obj['category'] in self.categories)]
			gt = np.zeros((len(data), 5))
			for idx, obj in enumerate(data):
				x1 = float(obj['box2d']['x1'])
				y1 = float(obj['box2d']['y1'])
				x2 = float(obj['box2d']['x2'])
				y2 = float(obj['box2d']['y2'])
				#gt[idx] = [self.categories.index(obj['category']), np.mean([x1,x2])*(1./width), np.mean([y1,y2])*(1./width), (x2-x1)*(1./height), (y2-y1)*(1./height)]
				gt[idx] = [self.categories.index(obj['category']), np.mean([x1,x2])*(1./width), np.mean([y1,y2])*(1./height), (x2-x1)*(1./width), (y2-y1)*(1./height)]

			gt_db += [{'image': image_path, 'label': gt, 'mask': darea_path, 'lane': lane_path}]
		print('database is ready')
		return gt_db


	def __getitem__(self, idx):
		data = self.db[idx]
		img = cv2.imread(data['image'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		seg_label = cv2.imread(data['mask'], cv2.IMREAD_GRAYSCALE)
		lane_label = cv2.imread(data['lane'], cv2.IMREAD_GRAYSCALE)

		# resize the input image
		resized_shape = max(self.inputsize)
		h0, w0 = img.shape[:2]
		ratio = resized_shape / max(h0, w0)
		if ratio != 1:
			interp = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
			img = cv2.resize(img, (int(w0 * ratio), int(h0 * ratio)), interpolation=interp)
			seg_label = cv2.resize(seg_label, (int(w0 * ratio), int(h0 * ratio)), interpolation=interp)
			lane_label = cv2.resize(lane_label, (int(w0 * ratio), int(h0 * ratio)), interpolation=interp)
		h, w = img.shape[:2]

		# letterbox: create boarder - height and with should be divisible by 32
		r = resized_shape / max(h, w)
		if self.is_train: r = min(r, 1.0) # scaling up only by train set

		dw, dh = np.mod(resized_shape-int(round(w*r)), 32) / 2, np.mod(resized_shape-int(round(h*r)), 32) / 2

		top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
		left, right = int(round(dw-0.1)), int(round(dw+0.1))

		img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=params['padding_rgb'])
		seg_label = cv2.copyMakeBorder(seg_label, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
		lane_label = cv2.copyMakeBorder(lane_label, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

		shapes = (h0, w0), ((h/h0, w/w0), (dw, dh))

		# labels
		det_label = data['label']

		labels = []
		labels_app = np.array([])

		if det_label.size > 0:
			labels = det_label.copy()
			labels[:,1] = r * w * (det_label[:, 1] - det_label[:, 3]/2) + dw # pad width
			labels[:,2] = r * h * (det_label[:, 2] - det_label[:, 4]/2) + dh # pad height
			labels[:,3] = r * w * (det_label[:, 1] + det_label[:, 3]/2) + dw
			labels[:,4] = r * h * (det_label[:, 2] + det_label[:, 4]/2) + dh

		if self.is_train:
			# albumentations
			try:
				new = self.albumentations_transform(image=img,
													mask=seg_label,
													mask0=lane_label,
													bboxes=labels[:, 1:] if len(labels) else labels,
													class_labels=labels[:,0] if len(labels) else labels)
				img = new['image']
				labels = np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])]) if len(labels) else labels
				seg_label = new['mask']
				lane_label = new['mask0']
			except ValueError:  # bbox have width or height == 0
				pass

			# random left-right flip
			if random.random() < 0.5:
				img = np.fliplr(img)
				#img = img[:, ::-1, :]
				seg_label = np.fliplr(seg_label)
				lane_label = np.fliplr(lane_label)
				if len(labels):
					rows, cols, channels = img.shape
					x1 = labels[:, 1].copy()
					x2 = labels[:, 3].copy()
					x_tmp = x1.copy()
					labels[:, 1] = cols - x2
					labels[:, 3] = cols - x_tmp

			""""""
			# random up-down flip
			if random.random() < 0.5:
				img = np.flipud(img)
				seg_label = np.flipud(seg_label)
				lane_label = np.flipud(lane_label)
				if len(labels):
					rows, cols, channels = img.shape
					y1 = labels[:, 2].copy()
					y2 = labels[:, 4].copy()
					y_tmp = y1.copy()
					labels[:, 2] = rows - y2
					labels[:, 4] = rows - y_tmp
			""""""

		if len(labels):
			  labels_app = np.zeros((len(labels),5))
			  labels_app[:, 0:4] = labels[:, 1:5]
			  labels_app[:, 4] = labels[:, 0]

		img = np.ascontiguousarray(img)
		_, segl = cv2.threshold(seg_label, 1, 255, cv2.THRESH_BINARY)
		_, lanel = cv2.threshold(lane_label, 1, 255, cv2.THRESH_BINARY)
		
		# lane to the forground
		segl = segl - (segl & lanel)
		union = segl | lanel
		background = 255-union

		segl = self.Tensor(segl)
		lanel = self.Tensor(lanel)
		background = self.Tensor(background)

		segmentation = torch.cat([background, segl, lanel], dim=0)

		img = self.transform(img)

		return img, data['image'], shapes, torch.from_numpy(labels_app), segmentation

	def __len__(self):
		return len(self.db)

	@staticmethod
	def collate_fn(batch):
		img, paths, shapes, labels_app, segmentation = zip(*batch)
		filenames = [file.split('/')[-1] for file in paths]
		max_num_annots = max(label.size(0) for label in labels_app)

		if max_num_annots > 0:
			annot_padded = -torch.ones((len(labels_app), max_num_annots, 5))
			for idx, label in enumerate(labels_app):
				if label.size(0) > 0:
					annot_padded[idx, :label.size(0), :] = label
		else:
			annot_padded = -torch.ones((len(labels_app), 1, 5))

		return {'img': torch.stack(img,0),
				'annot': annot_padded,
				'segmentation': torch.stack(segmentation,0),
				'filenames': filenames,
				'shapes': shapes}


class BatchGenerator(DataLoader):
	def __iter__(self):
		return BackgroundGenerator(super().__iter__())
