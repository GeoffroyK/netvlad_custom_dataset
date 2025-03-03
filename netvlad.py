import torch
import torch.nn as nn
import torchvision.models as models

class NetVLAD(nn.Module):
    '''
    Clusters : Number of centroids, used in the aggregation (distinct visual patterns)
    Alpha : Hyperparameter value that scale the assignment weights of local features
    Dim : Encoding dim of the CNN Backbone
    '''
    def __init__(self, num_clusters=64, dim=512, alpha=100.):
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1,1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
                - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x):
        N, C = x.shape[:2]
        K, _ = self.centroids.shape
        # Soft assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1) # Shape (N,K,H*W)
        soft_assign = torch.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1) # Shape (N, C, H*W)

        # Calculate residuals
        x_flatten = x_flatten.unsqueeze(2) # Shape (N, C, 1, H*W)

        centroids = self.centroids.view(C, K).unsqueeze(0).unsqueeze(-1) # Shape (1, C, K, 1)
        residual = x_flatten - centroids
        
        soft_assign = soft_assign.unsqueeze(1)
        residual *= soft_assign
        vlad = residual.sum(dim=-1) # Shape (N, C, K)

        # Intra norm
        vlad = torch.nn.functional.normalize(vlad, p=2, dim=2)
        
        # L2 norm
        vlad = vlad.view(N, -1) # Shape (N, C * K)
        vlad = torch.nn.functional.normalize(vlad, p=2, dim=1)

        return vlad

class VGG16NetVLAD(nn.Module):
    def __init__(self, num_clusters=64, pretrained=True):
        super(VGG16NetVLAD, self).__init__()

        if pretrained:
            self.backbone = models.vgg16(weights="DEFAULT").features
        else:
            self.backbone = models.vgg16().features
        self.netvlad = NetVLAD(num_clusters=num_clusters, dim=512)

    def forward(self, x):
        x = self.backbone(x)
        x = self.netvlad(x)
        return x

if __name__ == "__main__":
    model = VGG16NetVLAD(pretrained=True)
    input_tensor = torch.randn(1,3,224,224)
    output = model(input_tensor)
    print(output.shape)
