# code adapted from https://gist.github.com/z-a-f/b862013c0dc2b540cf96a123a6766e54
import torch
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset
import numpy as np
from tqdm.autonotebook import tqdm
import imageio
from collections import defaultdict
from PIL import Image
import time
from random import choices
import random
from torch.autograd import Variable
from collections import Counter
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode

# global variable
num_classes = 200

# helper function
def _add_channels(img, total_channels=3):
    while len(img.shape) < 3:  # third axis is the channels
        img = (np.expand_dims(img, axis=0))
    while(img.shape[0]) < 3:
        img = (np.concatenate([img, img[-1:,:,:]], axis=0))
        
    if torch.is_tensor(img):
        return img
    else:
        return (transforms.ToTensor()(img)).transpose(0, 1)
    
    
"""Creates a paths datastructure for the tiny imagenet.
Args:
  root_dir: Where the data is located
  download: Download if the data is not there
Members:
  label_id:
  ids:
  nit_to_words:
  data_dict:
"""
class TinyImageNetPaths:
    def __init__(self, root_dir):
        train_path = os.path.join(root_dir, 'train')
        val_path = os.path.join(root_dir, 'val')
        test_path = os.path.join(root_dir, 'test')

        wnids_path = os.path.join(root_dir, 'wnids.txt')
        words_path = os.path.join(root_dir, 'words.txt')

        self._make_paths(train_path, val_path, test_path,
                     wnids_path, words_path)
    def _make_paths(self, train_path, val_path, test_path,
                  wnids_path, words_path):
        self.ids = []
        with open(wnids_path, 'r') as idf:
            for nid in idf:
                nid = nid.strip()
                self.ids.append(nid)
        self.nid_to_words = defaultdict(list)
        with open(words_path, 'r') as wf:
            for line in wf:
                nid, labels = line.split('\t')
                labels = list(map(lambda x: x.strip(), labels.split(',')))
                self.nid_to_words[nid].extend(labels)

        self.paths = {
          'train': [],  # [img_path, id, nid, box]
          'val': [],  # [img_path, id, nid, box]
          'test': []  # img_path
        }
        # Get the test paths
        self.paths['test'] = list(map(lambda x: os.path.join(test_path, x),
                                          os.listdir(test_path)))
        # Get the validation paths and labels
        with open(os.path.join(val_path, 'val_annotations.txt')) as valf:
            for line in valf:
                fname, nid, x0, y0, x1, y1 = line.split()
                fname = os.path.join(val_path, 'images', fname)
                bbox = int(x0), int(y0), int(x1), int(y1)
                label_id = self.ids.index(nid)
                self.paths['val'].append((fname, label_id, nid, bbox))

        # Get the training paths
        train_nids = os.listdir(train_path)
        for nid in train_nids:
            anno_path = os.path.join(train_path, nid, nid+'_boxes.txt')
            imgs_path = os.path.join(train_path, nid, 'images')
            label_id = self.ids.index(nid)
            with open(anno_path, 'r') as annof:
                for line in annof:
                    fname, x0, y0, x1, y1 = line.split()
                    fname = os.path.join(imgs_path, fname)
                    bbox = int(x0), int(y0), int(x1), int(y1)
                    self.paths['train'].append((fname, label_id, nid, bbox))
                    
                    
"""Datastructure for the tiny image dataset.
Args:
  root_dir: Root directory for the data
  mode: One of "train", "test", or "val"
  preload: Preload into memory
  load_transform: Transformation to use at the preload time
  transform: Transformation to use at the retrieval time
  download: Download the dataset
Members:
  tinp: Instance of the TinyImageNetPaths
  img_data: Image data
  label_data: Label data
"""
class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, transform, normalize, mode='train', preload=True, load_transform=None,
               max_samples=None, image_shape=(3, 64, 64)):
        tinp = TinyImageNetPaths(root_dir)
        self.mode = mode
        self.label_idx = 1  # from [image, id, nid, box]
        self.preload = preload

        self.IMAGE_SHAPE = image_shape

        self.img_data = []
        self.label_data = []
        self.img_id = []

        self.max_samples = max_samples
        self.samples = tinp.paths[mode]
        self.samples_num = len(self.samples)

        if self.max_samples is not None:
            self.samples_num = min(self.max_samples, self.samples_num)
            self.samples = np.random.permutation(self.samples)[:self.samples_num]
        if self.preload:
            load_desc = "Preloading {} data...".format(mode)
            self.img_data = np.zeros((self.samples_num,) + self.IMAGE_SHAPE,
                                   dtype=np.float32)
            self.label_data = np.zeros((self.samples_num,), dtype=np.int)
            self.img_id = np.array(['aaaaaaaaaaa' for _ in range(self.samples_num)])
            
            for idx in tqdm(range(self.samples_num), desc=load_desc):
                s = self.samples[idx]
                img = Image.open(s[0])
                img = transform(img)
                img = _add_channels(img)
                img = normalize(img)
                self.img_data[idx] = img
                if mode != 'test':
                    self.label_data[idx] = s[self.label_idx]
                    self.img_id[idx] = s[0][-15:-5]
                
            if load_transform:
                for lt in load_transform:
                    result = lt(self.img_data, self.label_data)
                    self.img_data, self.label_data = result[:2]

    def __len__(self):
        return self.samples_num
    
    def __getitem__(self, idx):
        if self.preload:
            img = self.img_data[idx]
            img_id = self.img_id[idx]
            
            lbl = None if self.mode == 'test' else self.label_data[idx]
        else:
            s = self.samples[idx]
            img = imageio.imread(s[0])
            img_id = s[0][-15:-5]
            lbl = None if self.mode == 'test' else s[self.label_idx]
        sample = {'id': img_id, 'image': img, 'label': lbl}
        
        return sample['id'], sample['image'], sample['label']
    
    
def epsilon_per_label(label, max_hardness=0.01, min_hardness=0):
    step = (max_hardness-min_hardness)/4
    if label%40 < 8:
        return min_hardness + 0.25*step
    elif label%40 < 16:
        return min_hardness + 1.75*step 
    elif label%40 < 24:
        return min_hardness + 2.*step
    elif label%40 < 32:
        return min_hardness + 3.*step
    else:
        return max_hardness
def LNL_per_class(label, max_noise_rate):
    step = max_noise_rate/4
    if label < 40:
        return max_noise_rate
    elif label < 80:
        return max_noise_rate-step
    elif label < 120:
        return max_noise_rate-2*step
    elif label < 160:
        return max_noise_rate-3*step
    else:
        return 0.
def new_label_per_class(label):
    if label < 40:
        label_choices = list(np.arange(0,40))
    elif label < 80:
        label_choices = list(np.arange(40,80))
    elif label < 120:
        label_choices = list(np.arange(80,120))
    elif label < 160:
        label_choices = list(np.arange(120,160))
    else:
        label_choices = list(np.arange(160,200))
    new_label = choices(label_choices, k=1)
    return new_label[0]


# the gt_model is the proxy for the ground truth model to make samples be closer to the "true decision boundary"
class TinyImageNetAdvNoisyDataset(Dataset):
    def __init__(self, root_dir, transform, normalize, gt_model, samples_per_class, mode='train', preload=True, load_transform=None, max_samples=None, image_shape=(3, 64, 64), max_noise_rate=0.4, max_hardness=0.01, min_hardness=0):
        
        gt_model.eval()
        tinp = TinyImageNetPaths(root_dir)
        self.mode = mode
        self.label_idx = 1  # from [image, id, nid, box]
        self.preload = preload

        self.IMAGE_SHAPE = image_shape

        self.max_samples = max_samples
        self.original_samples = tinp.paths[mode]
        
        subset_indices = []
        for cls in range(num_classes):
            indices = [i for i, x in enumerate(self.original_samples) if x[1]==cls]
            indices_sampled = random.sample(indices, samples_per_class)
            subset_indices += indices_sampled 
        
        
        self.samples = [self.original_samples[i] for i in subset_indices]
        self.samples_num = len(self.samples)

        if self.max_samples is not None:
            self.samples_num = min(self.max_samples, self.samples_num)
            self.samples = np.random.permutation(self.samples)[:self.samples_num]
            
        if self.preload:
            load_desc = "Preloading {} data...".format(mode)
            self.img_data = np.zeros((self.samples_num,) + self.IMAGE_SHAPE,
                                   dtype=np.float32)
            self.original_img_data = np.zeros((self.samples_num,) + self.IMAGE_SHAPE,
                                   dtype=np.float32)
            self.label_data = np.zeros((self.samples_num,), dtype=np.int)
            self.original_label_data = np.zeros((self.samples_num,), dtype=np.int)
            self.img_id = np.array(['aaaaaaaaaaa' for _ in range(self.samples_num)])
            num_removed = 0
            
            for idx in tqdm(range(self.samples_num), desc=load_desc):
                s = self.samples[idx]
                img = Image.open(s[0])
                img = transform(img)
                img = _add_channels(img)
                img = normalize(img)
                self.original_img_data[idx] = img
                if mode == 'val':
                    self.label_data[idx] = s[self.label_idx]
                    self.img_id[idx] = s[0][-15:-5]
                elif mode == 'train':
                    self.label_data[idx] = s[self.label_idx]
                    self.img_id[idx] = s[0][-15:-5]
                    
                    if max_hardness != 0:
                        # create the adversarial version of the sample             
                        img_size = 224
                        image_tensor = img.unsqueeze(0)
                        resized_tensor = transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC)(image_tensor)
                        img_variable = Variable(resized_tensor, requires_grad=True)
                        
                        gt_model = gt_model.to('cpu')
                        output = gt_model.forward(img_variable)
                        y_pred = torch.max(output.data, 1)[1][0].item()   #get an index(class number) of a largest element
                        y_probs = F.softmax(output, dim=1)
                        pred_prob =  round((torch.max(y_probs.data, 1)[0][0].item()) * 100, 4)

                        if self.label_data[idx]!=y_pred or pred_prob < 70:
                            # skip this image
                            img = None
                            num_removed += 1 
                        else:
                            
                            target = Variable(torch.LongTensor([y_pred]), requires_grad=False)
                            loss = torch.nn.CrossEntropyLoss()
                            loss_cal = loss(output, target)
                            loss_cal.backward(retain_graph=True)    
                            # this will calculate gradient of each variable (with requires_grad=True) and can be accessed by "var.grad.data"
                            
                            # find epsilon based on label
                            eps = epsilon_per_label(self.label_data[idx], max_hardness, min_hardness)

                            x_grad = torch.sign(img_variable.grad.data)                # calculate the sign of gradient of the loss func (with respect to input X) (adv)
                            x_adversarial = img_variable.data + eps * x_grad          # find adv example using formula shown above
                            output_adv = gt_model.forward(Variable(x_adversarial))   # perform a forward pass on adv example
                            x_adv_pred = torch.max(output_adv.data, 1)[1][0].item()  # classify the adv example
                            
                            if x_adv_pred != y_pred or np.isnan(x_adversarial.numpy()).any():
                                # skip this image
                                img = None
                                num_removed += 1
                            else: 
                                img = x_adversarial.squeeze(0)
                                img_size = 64
                                resized_tensor = transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC)(img)
                                img = resized_tensor
                self.img_data[idx] = img
                
            self.original_label_data = self.label_data
            # The label noise level per class k: function of k
            corrupt_prob_per_sample = [LNL_per_class(l, max_noise_rate) for l in self.label_data]
            print('corrupt probabilities: ')
            print(corrupt_prob_per_sample[0])
            print('labels: ')
            print(self.label_data[0])
            rs = np.random.RandomState(seed=np.random.seed(int(time.time())))
            should_corrupt = rs.uniform(0, 1, size=self.samples_num) < corrupt_prob_per_sample
            self.label_data = [new_label_per_class(l) if p>0 else l for l, p in zip(self.label_data, should_corrupt)]
            
            self.samples_num_remaining = self.samples_num - num_removed
            self.remaining_img_data = np.zeros((self.samples_num_remaining,) + self.IMAGE_SHAPE,
                                   dtype=np.float32)
            self.remaining_label_data = np.zeros((self.samples_num_remaining,), dtype=np.int)
            self.remaining_original_label_data = np.zeros((self.samples_num_remaining,), dtype=np.int)
            self.remaining_img_id = np.array(['aaaaaaaaaaa' for _ in range(self.samples_num_remaining)])
            
            index = 0
            print('how many?', num_removed)
            cnt = 0
            h0 = 0
            h1 = 0
            h2 = 0
            h3 = 0
            h4 = 0
            for idx in tqdm(range(self.samples_num), desc=load_desc):
                if np.isnan(self.img_data[idx]).any():
                    cnt += 1
                    if int((self.original_label_data[idx]%40)/8) == 0:
                        h0 +=1
                    if int((self.original_label_data[idx]%40)/8) == 1:
                        h1 +=1
                    if int((self.original_label_data[idx]%40)/8) == 2:
                        h2 +=1
                    if int((self.original_label_data[idx]%40)/8) == 3:
                        h3 +=1
                    if int((self.original_label_data[idx]%40)/8) == 4:
                        h4 +=1
                    continue
                else:
                    self.remaining_img_data[index] = self.img_data[idx]
                    self.remaining_label_data[index] = self.label_data[idx]
                    self.remaining_original_label_data[index] = self.original_label_data[idx]
                    self.remaining_img_id[index] = self.img_id[idx]
                    index += 1
            print('how many?', cnt)  
            print('how many with hardness 0?', h0)
            print('how many with hardness 1?', h1)
            print('how many with hardness 2?', h2)
            print('how many with hardness 3?', h3)
            print('how many with hardness 4?', h4)
            self.cnt = cnt
            self.h0 = h0
            self.h1 = h1
            self.h2 = h2
            self.h3 = h3
            self.h4 = h4
            
            # now we should keep the same number of samples
            new_num_per_class = min(Counter(self.remaining_original_label_data).values())
            
            subset_indices = []
            for cls in range(num_classes):
                indices = [i for i, x in enumerate(self.remaining_original_label_data) if x==cls]
                indices_sampled = random.sample(indices, new_num_per_class)
                subset_indices += indices_sampled 

            print('New length of dataset', len(subset_indices))

            self.remaining_label_data = [self.remaining_label_data[i] for i in subset_indices]
            self.remaining_original_label_data = [self.remaining_original_label_data[i] for i in subset_indices]
            self.remaining_img_data = [self.remaining_img_data[i] for i in subset_indices]
            self.remaining_img_id = [self.remaining_img_id[i] for i in subset_indices]
            
            self.samples_num_remaining = len(self.remaining_label_data)
            print('dataset length', self.samples_num_remaining)
            
            if load_transform:
                for lt in load_transform:
                    result = lt(self.img_data, self.label_data, self.original_label_data)
                    self.img_data, self.label_data, self.original_label_data = result[:3]
                    if len(result) > 3:
                        self.transform_results.update(result[3])
            
    def __len__(self):
        return self.samples_num_remaining
    def __stats__(self):
        return self.cnt, self.h0, self.h1, self.h2, self.h3, self.h4
    def __getitem__(self, idx):
        if self.preload:
            img = self.remaining_img_data[idx]
            img_id = self.remaining_img_id[idx]
            
            lbl = None if self.mode == 'test' else self.remaining_label_data[idx]
            or_lbl = None if self.mode == 'test' else self.remaining_original_label_data[idx]
        else:
            s = self.samples[idx]
            img = imageio.imread(s[0])
            img_id = s[0][-15:-5]
            lbl = None if self.mode == 'test' else s[self.label_idx]
            or_lbl = None if self.mode == 'test' else s[self.original_label_idx]
        sample = {'id': img_id, 'image': img, 'label': lbl, 'original_label': or_lbl}
        
        return sample['id'], sample['image'], sample['label'], sample['original_label']

