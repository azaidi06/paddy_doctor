from sklearn.model_selection import StratifiedKFold
from fastai.vision.all import *
import timm

def get_dfs(n_splits=5):
    path = Path('../paddy_doctor')
    train_path = path/'data/train_images/'
    test_path = path/'data/test_images/'
    Path.BASE_PATH = path
    train_df = pd.read_csv(path/'data/train.csv')
    sub_df = pd.read_csv(path/'data/sample_submission.csv')
    
    #this fxn is now overloaded..... 
    skf = StratifiedKFold(n_splits=n_splits)
    skf.get_n_splits(train_df.variety, train_df.label)
    for t_idx, v_idx in skf.split(train_df.variety, train_df.label):
        break
    is_valid = np.zeros(len(train_df)).astype(int)
    np.put(is_valid, v_idx, [1])
    train_df['is_valid'] = is_valid
    
    
    train_df['path'] = str(train_path) + '/' + train_df['label'] \
                                + '/' + train_df['image_id']

    sub_df['path'] = str(test_path) + '/' + sub_df.image_id
    return train_df, sub_df



def get_dls(df, bs=8,size=224, val_idxs=False):
    if(val_idxs):
        val_col=-2
    else: val_col=None
    dls = ImageDataLoaders.from_df(df=df, fn_col=-1, 
                               label_col=1, bs=bs,
                               valid_col=val_col,
                               item_tfms=Resize(460),
                               batch_tfms=aug_transforms(size=size),
                               )
    return dls


def create_learner(dls, model_name='resnet18'):
    learner = Learner(dls, timm.create_model(model_name, pretrained=True,
                                             num_classes=dls.c), metrics=accuracy,)
    return learner