import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from netvlad import VGG16NetVLAD
from utils.evaluate import recall_at_n
from dataset import BrisbaneFrameDatasetTM
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


traverses_locations = {
    "sunset1": 0,
    "sunset2": 1,
    "morning": 2,
    "sunrise": 3,
    "daytime": 4 
}

def divide_traverses(sampled_points, n_points):
    traverses_paths = {}
    for traverse in traverses_locations:
        idx_beg = traverses_locations[traverse] * n_points
        idx_end = idx_beg + n_points
        traverses_paths[traverse] = sampled_points[idx_beg:idx_end]
    return traverses_paths

def evaluate_model(model, query_loader,  pred_loader):
    # Collect embeddings
    query_embs = []
    pred_embs = []
    pred_labels = []
    query_labels = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f'Inferences on : {device}')

    # Collect queries
    with torch.no_grad():
        for query, label in query_loader:
            query = query.to(device)
            query_emb = model(query)
            query_embs.append(query_emb)
            query_labels.append(label)

        for query, label in pred_loader:
            query = query.to(device)
            query_emb = model(query)
            pred_embs.append(query_emb)
            pred_labels.append(label)
    
    # Convert to np arrays
    query_embs = torch.cat(query_embs, dim=0).cpu().numpy()
    pred_embs = torch.cat(pred_embs, dim=0).cpu().numpy()
    query_labels = torch.cat(query_labels, dim=0).cpu().numpy()
    pred_labels = torch.cat(pred_labels, dim=0).cpu().numpy()

    # Recall@N
    recall_n = []
    for n in range(1,25):
        res = recall_at_n(query_embs, pred_embs, query_labels, pred_labels, n=n)
        recall_n.append(res)
        print(f"Recall@{n}: {res}")
    
    plt.figure()
    plt.plot(recall_n)
    plt.grid()
    plt.xlabel("Top N correlated candidates")
    plt.ylabel("Recall@N")
    plt.title(f"Recall@N - NetVLAD Brisbane {len(query_embs)} queries")
    plt.savefig("recall_at_n.png")

if __name__ == "__main__":
    model = VGG16NetVLAD(num_clusters=64)
    model.load_state_dict(torch.load("vgg16_netvlad_500.pth"))
    data_file_path = "/home/geoffroyk/netvlad_custom_dataset/data/images_paths.npy"

    sampled_points = np.load(data_file_path)
    n_points = len(sampled_points) // 5 # 5 traverses
    labels = np.tile(np.arange(n_points), 5)

   
    divided_traverses = divide_traverses(sampled_points, n_points)
    query_split_data = np.concatenate([divided_traverses["sunset1"], divided_traverses["morning"],  divided_traverses["sunrise"],  divided_traverses["daytime"]], axis=0)
    query_split_labels = np.concatenate([labels[:n_points], labels[2*n_points:3*n_points], labels[3*n_points:4*n_points], labels[4*n_points:]], axis=0)

    pred_split_data = np.concatenate([divided_traverses["sunset1"], divided_traverses["sunset2"]], axis=0)
    pred_split_labels = np.concatenate([labels[n_points:2*n_points], labels[2*n_points:3*n_points]], axis=0)

    query_dataset = BrisbaneFrameDatasetTM(query_split_data, query_split_labels, transform)
    pred_dataset = BrisbaneFrameDatasetTM(pred_split_data, pred_split_labels, transform)
    query_loader = torch.utils.data.DataLoader(query_dataset, batch_size=1, shuffle=False)
    pred_loader = torch.utils.data.DataLoader(pred_dataset, batch_size=1, shuffle=False)

    evaluate_model(model, query_loader, pred_loader)