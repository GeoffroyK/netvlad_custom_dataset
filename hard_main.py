import argparse
import torch
import torch.nn as nn
import numpy as np
from dataset import BrisbaneFrameDatasetTM
from netvlad import VGG16NetVLAD
from torch.utils.data import DataLoader
from train_mining import train_model, load_model
from torchvision import transforms
from hard_mining import HardTripletLoss



data_file_path = "/home/geoffroyk/netvlad_custom_dataset/data/images_paths.npy"

# Define transformation
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset_file', type=str, default=data_file_path, help='Numpy array with the images paths')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_clusters', type=int, default=64, help='Number of clusters for NetVLAD')
    # parser.add_argument('--alpha', type=float, default=100, help='Alpha parameter for NetVLAD')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained weights for the VGG16')
    return parser.parse_args()

def init_model_weights(model):
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            torch.nn.init.kaiming_normal_(layer.weight)

def main():
    args = parse_args()
    print(args)
    model = VGG16NetVLAD(args.num_clusters, pretrained=args.pretrained)

    for param in model.backbone.parameters():  
        param.requires_grad = True  # Unfreeze VGG16 if needed

    for name, param in model.netvlad.named_parameters():
        print(name, param.requires_grad, param.grad is None)

    init_model_weights(model)
    criterion = HardTripletLoss(margin=1., squared=True)
    
    #optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    sampled_points = np.load(args.dataset_file)
    
    n_points = len(sampled_points) // 5 # 5 traverses
    labels = np.tile(np.arange(n_points), 5)

    train_split_data, train_split_labels  = sampled_points[:-n_points], labels[:-n_points]
    val_split_data, val_split_labels = sampled_points[-n_points*2:], labels[-n_points*2:]

    train_dataset = BrisbaneFrameDatasetTM(train_split_data, train_split_labels, transform)
    val_dataset = BrisbaneFrameDatasetTM(val_split_data, val_split_labels, transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)


    train_model(model, criterion, optimizer, args.num_epochs, train_loader, val_loader)


if __name__ == "__main__":
    main()
