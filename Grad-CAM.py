import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
import json
from tqdm import tqdm
import argparse

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        forward_handle = self.target_layer.register_forward_hook(forward_hook)
        backward_handle = self.target_layer.register_full_backward_hook(backward_hook)
        self.hook_handles = [forward_handle, backward_handle]
    
    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
    
    def generate_cam(self, input_tensor, target_class=None):

        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        

        self.model.zero_grad()
        
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        gradients = self.gradients[0].cpu().numpy()
        activations = self.activations[0].cpu().numpy()
        
        weights = np.mean(gradients, axis=(1, 2))
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        cam = np.maximum(cam, 0)
        
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        
        return cam, target_class, output

def preprocess_image(image_path, size=(224, 224)):
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0)
    return input_tensor, image

def save_gradcam(cam, original_image, save_path, alpha=0.5):
    cam = cv2.resize(cam, (original_image.width, original_image.height))
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    heatmap = Image.fromarray(heatmap)
    
    overlayed = Image.blend(original_image, heatmap, alpha)
    
    overlayed.save(save_path)

def load_custom_convnext_model(model_path, num_classes, device):
    model = models.convnext_base(pretrained=False)
    
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_ftrs, num_classes)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model

def process_test_set(test_dir, output_dir, model, target_layer, class_names, num_images_per_class=5):
    os.makedirs(output_dir, exist_ok=True)
    
    grad_cam = GradCAM(model, target_layer)
    
    classes = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    classes.sort()
    
    for cls in classes:
        os.makedirs(os.path.join(output_dir, cls), exist_ok=True)
    
    for cls in classes:
        print(f"Processing class: {cls}")
        class_dir = os.path.join(test_dir, cls)
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if num_images_per_class > 0:
            image_files = image_files[:num_images_per_class]
        
        for img_file in tqdm(image_files):
            img_path = os.path.join(class_dir, img_file)
            
            try:
                input_tensor, original_image = preprocess_image(img_path)
                
                cam, target_class, output = grad_cam.generate_cam(input_tensor)
                
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence = probabilities[0, target_class].item()
                
                predicted_class = class_names[target_class] if class_names else str(target_class)
                
                base_name = os.path.splitext(img_file)[0]
                save_path = os.path.join(output_dir, cls, f"{base_name}_gradcam.jpg")
                save_gradcam(cam, original_image, save_path)
                
                info_path = os.path.join(output_dir, cls, f"{base_name}_info.txt")
                with open(info_path, 'w') as f:
                    f.write(f"True class: {cls}\n")
                    f.write(f"Predicted class: {predicted_class}\n")
                    f.write(f"Confidence: {confidence:.4f}\n")
                    
                if predicted_class != cls:
                    error_dir = os.path.join(output_dir, "errors", f"{cls}_to_{predicted_class}")
                    os.makedirs(error_dir, exist_ok=True)
                    error_save_path = os.path.join(error_dir, f"{base_name}_gradcam.jpg")
                    save_gradcam(cam, original_image, error_save_path)
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    grad_cam.remove_hooks()


def main():
    parser = argparse.ArgumentParser(description='Grad-CAM visualization for ConvNeXt model')
    
    parser.add_argument('--test_dir', type=str, default='test_dir', 
                       help='Directory containing test images (default: test_dir)')
    parser.add_argument('--output_dir', type=str, default='output_dir',
                       help='Directory to save Grad-CAM results (default: output_dir)')
    parser.add_argument('--model_path', type=str, default='model.pth',
                       help='Path to trained model file (default: model.pth)')
    parser.add_argument('--num_classes', type=int, default=26,
                       help='Number of classes in the model (default: 26)')
    parser.add_argument('--num_images_per_class', type=int, default=10,
                       help='Number of images to process per class (default: 10, use -1 for all)')
    parser.add_argument('--class_names_file', type=str, default=None,
                       help='JSON file containing class names (if not provided, uses default)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda or cpu), if not specified auto-detects')
    
    args = parser.parse_args()
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.class_names_file and os.path.exists(args.class_names_file):
        with open(args.class_names_file, 'r') as f:
            class_names = json.load(f)
    else:
        class_names = [
            "1001", "1002", "1003", "1005", "1006", "1007", "1009", "1010", 
            "1011", "1012", "1013", "1018", "1019", "1020", "1021", "1022", 
            "1024", "1025", "1026", "1027", "1028", "1029", "1030", "1031", 
            "1032", "1034"
        ]
    print(f"Using device: {device}")
    print(f"Test directory: {args.test_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model path: {args.model_path}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Number of images per class: {args.num_images_per_class}")
    print(f"Number of class names provided: {len(class_names)}")
    
    #load models
    model = load_custom_convnext_model(args.model_path, args.num_classes, device)
    model.to(device)
    print("Model loaded successfully.")
    
    #setting layers for Grad-CAM
    target_layer = model.features[-1][-1].block[0]
    
    process_test_set(args.test_dir, args.output_dir, model, target_layer, 
                     class_names, num_images_per_class=args.num_images_per_class)

if __name__ == "__main__":
    main()