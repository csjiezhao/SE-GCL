import os
import torchvision.models as inbuilt_models
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import numpy as np
import pandas as pd


class ImageFeatureExtractor(nn.Module):
    def __init__(self):
        super(ImageFeatureExtractor, self).__init__()
        self.extractor = inbuilt_models.swin_t(weights=inbuilt_models.Swin_T_Weights.DEFAULT)
        self.linear = nn.Linear(1000, 128, bias=False)

    def forward(self, x):
        x = self.extractor(x)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    city_name = 'Porto'
    img_data_path = f'../data/{city_name}/image'

    vis_feature_extractor = ImageFeatureExtractor()
    vis_feature_extractor.eval()
    vis_features = []

    edge_df = pd.read_csv(os.path.join(img_data_path, 'pano.csv'), index_col=None)
    for _, row in edge_df.iterrows():
        edge_idx, pano_id = row
        print(f'Processing Image: {edge_idx}')
        pil_image = Image.open(os.path.join(img_data_path, f'{pano_id}.jpg'))  # (H, W, C)
        img = T.ToTensor()(pil_image)  # (H, W, C)
        with torch.no_grad():
            img_feature = vis_feature_extractor(img.unsqueeze(0))
        vis_features.append(img_feature.squeeze(0).detach().numpy())

    vis_features = np.asarray(vis_features)
    print(vis_features.shape)
    np.savez_compressed(os.path.join(img_data_path, 'segment_vis_feat.npz'), data=np.asarray(vis_features))
