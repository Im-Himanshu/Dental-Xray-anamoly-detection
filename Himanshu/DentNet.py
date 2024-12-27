import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import os
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

import torch
import torch.nn as nn
import torch.nn.functional as F

class DentNetFPN(nn.Module):
    def __init__(self, resnet_backbone, num_classes):
        super(DentNetFPN, self).__init__()

        # Use specific layers from the ResNet backbone for FPN
        self.layer1 = nn.Sequential(*list(resnet_backbone.children())[:5])  # Low-level features
        self.layer2 = nn.Sequential(*list(resnet_backbone.children())[5])  # Mid-level features
        self.layer3 = nn.Sequential(*list(resnet_backbone.children())[6])  # High-level features

        # 1x1 convolutions for feature dimension alignment
        self.conv1x1_1 = nn.Conv2d(256, 256, kernel_size=1)  # Layer1 output channels
        self.conv1x1_2 = nn.Conv2d(512, 256, kernel_size=1)  # Layer2 output channels
        self.conv1x1_3 = nn.Conv2d(1024, 256, kernel_size=1)  # Layer3 output channels

        # 3x3 convolutions for refining upsampled features
        self.conv3x3_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3x3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # Final 1x1 convolution for class predictions
        self.final_conv = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.size()[2:]  # Store input size for dynamic resizing

        # Extract features from different levels
        low_level_feat = self.layer1(x)  # Layer1 features
        mid_level_feat = self.layer2(low_level_feat)  # Layer2 features
        high_level_feat = self.layer3(mid_level_feat)  # Layer3 features

        # FPN-like feature merging
        high_level_up = F.interpolate(self.conv1x1_3(high_level_feat), size=mid_level_feat.size()[2:], mode='bilinear', align_corners=False)
        mid_level_out = self.conv1x1_2(mid_level_feat) + high_level_up
        mid_level_out = self.conv3x3_1(mid_level_out)

        mid_level_up = F.interpolate(mid_level_out, size=low_level_feat.size()[2:], mode='bilinear', align_corners=False)
        low_level_out = self.conv1x1_1(low_level_feat) + mid_level_up
        low_level_out = self.conv3x3_2(low_level_out)

        # Final upsampling to match the input size
        output = F.interpolate(self.final_conv(low_level_out), size=input_size, mode='bilinear', align_corners=False)

        return output




# Custom loss function combining pixel accuracy and cross-entropy loss
def custom_loss(output, target):
    """
    Args:
        output (torch.Tensor): Logits output from the model (B, C, H, W)
        target (torch.Tensor): Ground truth labels (B, H, W)
    """
    # Calculate cross-entropy loss
    ce_loss = F.cross_entropy(output, target, reduction='mean')

    # Calculate pixel accuracy
    _, preds = torch.max(output, dim=1)
    correct_pixels = (preds == target).sum().item()
    total_pixels = target.numel()
    pixel_accuracy = correct_pixels / total_pixels

    # Weighted loss combining cross-entropy and (1 - pixel accuracy)
    loss = ce_loss + (1 - pixel_accuracy)
    return loss


# Dataset class for loading images and masks
class DentalDataset(Dataset):
    def __init__(self, radiograph_dir, segmentation_dir, expert_mask_dir, transform=None):
        self.radiograph_dir = radiograph_dir
        self.segmentation_dir = segmentation_dir
        self.expert_mask_dir = expert_mask_dir
        self.transform = transform
        self.image_filenames = os.listdir(radiograph_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load input image
        image_path = os.path.join(self.radiograph_dir, self.image_filenames[idx])
        image = Image.open(image_path).convert("RGB")

        # Load segmentation mask
        seg_path = os.path.join(self.segmentation_dir, self.image_filenames[idx])
        segmentation_mask = Image.open(seg_path)

        # Load expert mask
        expert_path = os.path.join(self.expert_mask_dir, self.image_filenames[idx])
        expert_mask = Image.open(expert_path)

        # Convert masks to tensors and pick only the first channel
        segmentation_tensor = transforms.ToTensor()(segmentation_mask)[0]  # Pick the first channel
        expert_tensor = transforms.ToTensor()(expert_mask)[0]  # Pick the first channel

        # Combine masks into a single target tensor
        target = torch.stack([segmentation_tensor, expert_tensor], dim=0)

        if self.transform:
            image = self.transform(image)

        return image, target


# Utility function to overlay masks on images
def overlay_masks_on_image(image, masks):
    image = transforms.ToPILImage()(image.cpu().clone().squeeze(0))
    draw = ImageDraw.Draw(image)
    mask_colors = [(255, 0, 0), (0, 255, 0)]  # Colors for the two masks

    for i, mask in enumerate(masks):
        mask = transforms.ToPILImage()(mask.cpu().squeeze(0))
        image = Image.blend(image, mask.convert("RGB"), alpha=0.5)

    return image


# Training script
if __name__ == "__main__":
    # Allow for flexibility in selecting ResNet versions
    resnet_versions = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "resnet152": models.resnet152
    }

    resnet_version = "resnet18"  # Change this to select a different ResNet version
    resnet_backbone = resnet_versions[resnet_version](pretrained=True)

    # Instantiate the DentNet with two output channels (teeth and anomaly segmentation)
    model = DentNetFPN(resnet_backbone, num_classes=2)
    summary(model, input_size=(1, 3, 1615, 840))  # Batch size of 1
    # Directories for data
    radiograph_dir = "../TUFTS-project/Radiographs"
    segmentation_dir = "../TUFTS-project/Segmentation/teeth_mask/"
    expert_mask_dir = "../TUFTS-project/Expert/mask"

    # Transformations for input images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset and dataloader
    dataset = DentalDataset(radiograph_dir, segmentation_dir, expert_mask_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Set up TensorBoard logging
    writer = SummaryWriter()

    # Training loop
    model.train()
    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, (images, targets) in enumerate(dataloader):
            # Move data to GPU if available
            if torch.cuda.is_available():
                model = model.cuda()
                images = images.cuda()
                targets = targets.cuda()

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = custom_loss(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Log results to TensorBoard
            if batch_idx % 10 == 0:
                output_masks = torch.argmax(outputs, dim=1, keepdim=True)  # Predicted masks
                overlay_image = overlay_masks_on_image(images[0], output_masks)
                writer.add_image("Input with Masks", transforms.ToTensor()(overlay_image),
                                 global_step=epoch * len(dataloader) + batch_idx)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")
        writer.add_scalar("Loss/train", total_loss / len(dataloader), epoch)

    # Save the model after training
    save_path = "dentnet_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Close the TensorBoard writer
    writer.close()
