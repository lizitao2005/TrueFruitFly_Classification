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
    test_dir = "test_dir"  
    output_dir = "output_dir"  
    model_path = "model.pth"   
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_classes = 26  
    

    class_names = [
        "1001", "1002", "1003","1005","1006","1007","1009","1010","1011","1012","1013","1018","1019","1020","1021","1022","1024","1025","1026","1027","1028","1029","1030","1031","1032","1034"
    ]  

    model = load_custom_convnext_model(model_path, num_classes, device)
    model.to(device)
    print("Model structure:")
    print(model)
    
    target_layer = model.features[-1][-1].block[0] 

    process_test_set(test_dir, output_dir, model, target_layer, class_names, num_images_per_class=10)

if __name__ == "__main__":
    main()