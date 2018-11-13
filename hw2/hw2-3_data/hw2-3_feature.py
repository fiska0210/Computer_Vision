import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from hw2_3_model import LeNet
from torch.utils.data import DataLoader, Dataset
from sklearn.manifold import TSNE
import cv2

class ImageSet(Dataset):
   def __init__(self, dataPath, Range):
      images = []
      for i in range(10):
         for j in range(*Range):
            img_path = 'class_{}/{}.png'.format(i, str(j).zfill(4))
            img = cv2.imread(os.path.join(dataPath, img_path))[:,:,0]
            img = img.astype(np.float32)
            # 28X28
            img = img.reshape(1, 28, 28)
            # normalize
            img = img / 255 
            label = i
            images.append({'img': img, 'label': label})
      self.data = images
   def __getitem__(self, index):
      # return img, label
      img = self.data[index]['img']
      label = self.data[index]['label']
      return img, label 
   def __len__(self):
      return len(self.data)

o_prefix = './out/'
# detect gpu
use_cuda = torch.cuda.is_available()

model = LeNet()
if use_cuda:
   model.cuda()
model.load_state_dict(torch.load('lenet5.pt')) # remap everything onto CPU
model.eval()


def visualize_feature(image, layer_idx):
   """
   convnet
   (c1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
   (relu1): ReLU()
   (s2): MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1, ceil_mode=False)
   (c3): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
   (relu2): ReLU()
   (s4): MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1, ceil_mode=False)
   """
   if use_cuda:
      image = image.cuda()
   output = image
   for index, layer in enumerate(model.convnet):
      output = layer(output)
      if index == layer_idx:
         return output

data = ImageSet('hw2-3_data/valid/', [5000, 5100]) # 每組100張
data_loader = DataLoader(data, batch_size = 1)

def plot(X, title):
   em = TSNE(n_components = 2).fit_transform(X)
   fig, ax = plt.subplots()
   for i in range(0, em.shape[0], 100):
      x_em, y_em = em[i : i + 100, 0], em[i : i + 100, 1]
      ax.scatter(x_em, y_em, alpha = 0.8, edgecolors='none')
   plt.title('{}'.format(title))
   plt.savefig(os.path.join(o_prefix, '{}.png'.format(title)))
   print('Saving {}.png ...'.format(title))

def run():
   l_fs, h_fs = [], []
   for i, (image, label) in enumerate(data_loader):
      l_f = visualize_feature(image, 0)
      h_f = visualize_feature(image, 5)
      l_fs.append(l_f.detach().numpy())
      h_fs.append(h_f.detach().numpy()) # convert to np

   l_fs = np.array(l_fs, dtype = np.float_).reshape(len(l_fs), -1)
   h_fs = np.array(h_fs, dtype = np.float_).reshape(len(h_fs), -1)
   plot(l_fs, 'Low-Level-Features')
   plot(h_fs, 'High-Level-Features')

if __name__ == '__main__':
   run()

