import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os

def get_args():
	parser = argparse.ArgumentParser('Result printer')
	parser.add_argument('--checkpoint', type=str, default=None, help='Load a saved checkpoint')
	parser.add_argument('--image_path', type=str, default='./sample_images', help='Path to save images')
	args = parser.parse_args()
	return args


def load_results_by_checkpoints(checkpoint_name):
	try:
		ckpt = torch.load(checkpoint_name)
		return pd.read_json(ckpt['results'], orient='columns')
	except Exception as e:
		print(f'Loading failed: {e}')
		return -1
		
		
def plot_losses(data, image_path):
	num_of_epochs = data['Epoch'].max()
	
	# plot detailed train losses sharey=True
	d = {0:'Loss', 1:'Classification_loss', 2:'Segmentation_loss', 3:'Regression_loss'}
	fig, axs = plt.subplots(4,num_of_epochs,figsize=(20,8),sharey='row')
	for idx1, row in enumerate(axs):
		for idx2, ax in enumerate(row):
			sns.lineplot(data=data[(data['Epoch']==idx2+1) & (data['Dataset']=='Train')], x='Step', y=d[idx1], color=(0,0,1), ax=ax)
	fig.suptitle('Detailed train losses by epochs')
	plt.savefig(os.path.join(image_path, f'res_details_e{num_of_epochs}_sharey.png'))
	print(f'{image_path}/res_details_e{num_of_epochs}_sharey.png saved')
	
	# plot detailed train losses sharey=False
	d = {0:'Loss', 1:'Classification_loss', 2:'Segmentation_loss', 3:'Regression_loss'}
	fig, axs = plt.subplots(4,num_of_epochs,figsize=(20,8),sharey=False)
	for idx1, row in enumerate(axs):
		for idx2, ax in enumerate(row):
			sns.lineplot(data=data[(data['Epoch']==idx2+1) & (data['Dataset']=='Train')], x='Step', y=d[idx1], color=(0,0,1), ax=ax)
	fig.suptitle('Detailed train losses by epochs')
	plt.savefig(os.path.join(image_path, f'res_details_e{num_of_epochs}.png'))
	print(f'{image_path}/res_details_e{num_of_epochs}.png saved')

	# plot losses by epochs
	data_agg = pd.DataFrame(columns=['Dataset','Epoch','Loss','Classification_loss','Segmentation_loss','Regression_loss'])
	for e in range(1, num_of_epochs+1):
		for ds in ['Train','Val']:
			l = data[(data['Dataset']==ds) & (data['Epoch']==e)][['Loss','Classification_loss','Segmentation_loss','Regression_loss']].mean(axis=0)
			record = {'Dataset':ds, 'Epoch':e, 'Loss':l[0], 'Classification_loss':l[1], 'Segmentation_loss':l[2], 'Regression_loss':l[3]}
			data_agg = data_agg.append(record, ignore_index=True)

	fig, ax = plt.subplots(4,1,figsize=(20,8))
	sns.barplot(data=data_agg, x='Epoch', y='Loss', hue='Dataset', ax=ax[0])
	sns.barplot(data=data_agg, x='Epoch', y='Classification_loss', hue='Dataset', ax=ax[1])
	sns.barplot(data=data_agg, x='Epoch', y='Segmentation_loss', hue='Dataset', ax=ax[2])
	sns.barplot(data=data_agg, x='Epoch', y='Regression_loss', hue='Dataset', ax=ax[3])
	for a in ax:
		for p in a.patches:
			_x = p.get_x() + p.get_width() / 2
			_y = p.get_y() + p.get_height() + (p.get_height()*0.01)
			value = '{:.3f}'.format(p.get_height())
			a.text(_x, _y, value, ha="center")
	plt.savefig(os.path.join(image_path, f'res_losses_e{num_of_epochs}.png'))
	print(f'{image_path}/res_losses_e{num_of_epochs}.png saved')

	
if __name__ == '__main__':
	opt = get_args()
	results_df = load_results_by_checkpoints(opt.checkpoint)
	
	plot_losses(results_df, opt.image_path)
