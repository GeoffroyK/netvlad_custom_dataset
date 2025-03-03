import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# Define transformation
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# Brisbane Custom Dataset (Frame Based)
class BrisbaneFrameDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        anchor_image = self.transform(Image.open(self.image_paths[idx]))
        anchor_label = self.labels[idx]

        # Find a positive sample (same label)
        positive_indices = [i for i, label in enumerate(self.labels) if label == anchor_label and i != idx]
        positive_idx = np.random.choice(positive_indices)
        positive_image = self.transform(Image.open(self.image_paths[positive_idx]))

        # Find a negative sample (different label)
        negative_indices = [i for i, label in enumerate(self.labels) if label != anchor_label]
        negative_idx = np.random.choice(negative_indices)
        negative_image = self.transform(Image.open(self.image_paths[negative_idx]))

        return anchor_image, positive_image, negative_image

if __name__ == "__main__":
    paths = np.load("data/images_paths.npy")
    # n_points = len(paths) // 5 # 5 traverses
    # labels = np.tile(np.arange(n_points), 5)
 
    # train_split_data, train_split_labels  = paths[:-n_points], labels[:-n_points]
    # val_split_data, val_split_labels = paths[-n_points:], labels[-n_points:]

    # assert len(train_split_labels) == len(train_split_data)
    # assert len(val_split_labels) == len(val_split_data)

    # train_dataset = BrisbaneFrameDataset(train_split_data, train_split_labels, transform=transform)
    # val_dataset = BrisbaneFrameDataset(val_split_data, val_split_labels, transform=transform)

    # print(train_dataset[0].shape)
