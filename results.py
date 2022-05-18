import pandas as pd



def load_results_by_checkpoints(checkpoint_name):
    try:
        ckpt = torch.load(checkpoint_name)
        print(ckpt.keys())
        return pd.read_json(ckpt['results'], orient='columns')
    except Exception as e:
        print(f'Loading failed: {e}')
        return -1
		
		
def plot_losses(data, details=False, sharey=False):
    num_of_epochs = data['Epoch'].max()
    
    # plot detailed train losses
    if details:
        d = {0:'Loss', 1:'Classification_loss', 2:'Segmentation_loss', 3:'Regression_loss'}
        fig, axs = plt.subplots(4,num_of_epochs+1,figsize=(20,8),sharey=sharey)
        for idx1, row in enumerate(axs):
            for idx2, ax in enumerate(row):
                sns.lineplot(data=data[(data['Epoch']==idx2) & (data['Dataset']=='Train')], x='Step', y=d[idx1], color=(0,0,1), ax=ax)
        fig.suptitle('Detailed train losses by epochs')

    # plot losses by epochs
    data_agg = pd.DataFrame(columns=['Dataset','Epoch','Loss','Classification_loss','Segmentation_loss','Regression_loss'])
    for e in range(num_of_epochs+1):
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
			
	
