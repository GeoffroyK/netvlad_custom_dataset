import torch
import torch.nn as nn
from netvlad import VGG16NetVLAD

def save_model(model):
    torch.save(model.state_dict(), "vgg16_netvlad.pth")

def load_model():
    model = VGG16NetVLAD(num_clusters=64, weights=None)
    model.load_state_dict(torch.load("vgg16_netvlad.pth"))
    return model

def train_model(model, criterion, optimizer, epochs, train_loader, val_loader):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # Training step
        for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward passes
            anchor_output = model(anchor)
            positive_output = model(positive)
            negative_output = model(negative)

            # Compute loss
            loss = criterion(anchor_output, positive_output, negative_output)

            # Backward pass and optim step
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    
        running_loss /= batch_idx

        model.eval()
        val_loss = 0.0

        # Validation step
        with torch.no_grad():
            for batch_idx, (anchor, positive, negative) in enumerate(val_loader):
                anchor_output = model(anchor)
                positive_output = model(positive)
                negative_output = model(negative)

                loss = criterion(anchor_output, positive_output, negative_output)
                val_loss += loss.item()
            val_loss /= batch_idx

        print(f"Epoch [{epoch+1}/{epochs}], Training Loss:{running_loss}, Val Loss:{val_loss}")
    save_model(model)


