import torch
import torch.nn as nn
from netvlad import VGG16NetVLAD
import wandb

def save_model(model):
    torch.save(model.state_dict(), "vgg16_netvlad.pth")

def load_model():
    model = VGG16NetVLAD(num_clusters=64, weights=None)
    model.load_state_dict(torch.load("vgg16_netvlad.pth"))
    return model

def train_model(model, criterion, optimizer, epochs, train_loader, val_loader):
    wandb.init(project="netvlad_rgb_brisbane", config={"epochs": epochs})
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    model = model.to(device)
    model.train()
    running_loss = 0.0

    for epoch in range(epochs):
        # Training step
        for batch_idx, (anchor, label) in enumerate(train_loader):
            anchor, label = anchor.to(device), label.to(device)
            optimizer.zero_grad()

            # Forward passes
            anchor_output = model(anchor)

            # Compute loss
            loss = criterion(anchor_output, label)

            # Backward pass and optim step
            loss.backward()
            
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            # Print gradients (optional)
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"Gradient for {name}: {param.grad.norm().item():.4f}")
        
            optimizer.step()
            running_loss += loss.item()
    
        running_loss /= batch_idx

        model.eval()
        val_loss = 0.0

        # Validation step
        with torch.no_grad():
            for batch_idx, (anchor, label) in enumerate(train_loader):
                anchor, label = anchor.to(device), label.to(device)

                anchor_output = model(anchor)

                loss = criterion(anchor_output, label)
                val_loss += loss.item()
            val_loss /= batch_idx

        print(f"Epoch [{epoch+1}/{epochs}], Training Loss:{running_loss}, Val Loss:{val_loss}")
        
        wandb.log({
            "train_loss": running_loss,
            "val_loss": val_loss,
        })
    wandb.finish()
    save_model(model)