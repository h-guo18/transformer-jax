import os
import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp
from jax import random
import flaxmodels
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import CIFAR100
from dataset import SetAnomalyDataset, numpy_collate

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "./"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "./"
main_rng = random.PRNGKey(42)


def image_to_numpy(img):
    img = np.array(img, dtype=np.float32)
    img = img / 255.   # Normalization is done in the ResNet
    return img


def extract_features(dataset, save_file):
    if not os.path.isfile(save_file):
        data_loader = data.DataLoader(dataset, batch_size=128, shuffle=False, drop_last=False,
                                      collate_fn=lambda batch: np.stack([b[0] for b in batch], axis=0))
        extracted_features = []
        for imgs in tqdm(data_loader):
            feats = apply_resnet(imgs)
            # Average pooling on the last conv features to obtain a image-level feature vector
            feats = feats['block4_2'].mean(axis=(1, 2))
            extracted_features.append(feats)
        extracted_features = jnp.concatenate(extracted_features, axis=0)
        extracted_features = jax.device_get(extracted_features)
        np.savez_compressed(save_file, feats=extracted_features)
    else:
        extracted_features = np.load(save_file)['feats']
    return extracted_features


# Resize to 224x224, and map to JAX
transform = transforms.Compose([transforms.Resize((224, 224)),
                                image_to_numpy
                                ])
# Loading the training dataset.
train_set = CIFAR100(root=DATASET_PATH, train=True,
                     transform=transform, download=True)

# Loading the test set
test_set = CIFAR100(root=DATASET_PATH, train=False,
                    transform=transform, download=True)

# For later, keep a dictionary mapping class indices to class names
class_idx_to_name = {val: key for key, val in train_set.class_to_idx.items()}
# Pretrained ResNet34 on ImageNet
resnet34 = flaxmodels.ResNet34(
    output='activations', pretrained='imagenet', normalize=True)
main_rng, resnet_rng = random.split(main_rng, 2)
resnet_params = resnet34.init(resnet_rng, jnp.zeros((1, 224, 224, 3)))
# Jit its forward pass for efficiency
apply_resnet = jax.jit(lambda imgs: resnet34.apply(
    resnet_params, imgs, train=False))
train_feat_file = os.path.join(CHECKPOINT_PATH, "train_set_features.npz")
train_set_feats = extract_features(train_set, train_feat_file)

test_feat_file = os.path.join(CHECKPOINT_PATH, "test_set_features.npz")
test_feats = extract_features(test_set, test_feat_file)
# Split train into train+val
# Get labels from train set
labels = np.array(train_set.targets, dtype=np.int32)

# Get indices of images per class
num_labels = labels.max()+1
sorted_indices = np.argsort(labels).reshape(
    num_labels, -1)  # [classes, num_imgs per class]

# Determine number of validation images per class
num_val_exmps = sorted_indices.shape[1] // 10

# Get image indices for validation and training
val_indices = sorted_indices[:, :num_val_exmps].reshape(-1)
train_indices = sorted_indices[:, num_val_exmps:].reshape(-1)

# Group corresponding image features and labels
train_feats, train_labels = train_set_feats[train_indices], labels[train_indices]
val_feats,   val_labels = train_set_feats[val_indices],   labels[val_indices]
SET_SIZE = 10
test_labels = np.array(test_set.targets, dtype=np.int32)

anom_train_dataset = SetAnomalyDataset(
    train_feats, train_labels, np_rng=np.random.default_rng(42), set_size=SET_SIZE, train=True)
anom_val_dataset = SetAnomalyDataset(
    val_feats,   val_labels,   np_rng=np.random.default_rng(43), set_size=SET_SIZE, train=False)
anom_test_dataset = SetAnomalyDataset(
    test_feats,  test_labels,  np_rng=np.random.default_rng(123), set_size=SET_SIZE, train=False)

anom_train_loader = data.DataLoader(
    anom_train_dataset, batch_size=64, shuffle=True,  drop_last=True,  collate_fn=numpy_collate)
anom_val_loader = data.DataLoader(
    anom_val_dataset,   batch_size=64, shuffle=False, drop_last=False, collate_fn=numpy_collate)
anom_test_loader = data.DataLoader(
    anom_test_dataset,  batch_size=64, shuffle=False, drop_last=False, collate_fn=numpy_collate)
