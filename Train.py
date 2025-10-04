import os
import json
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score
import seaborn as sns

# Keep reproducible
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='--FruitFyl---')
    parser.add_argument('--data_dir', type=str, default='/public/home/2023002/zitaoli/resnet/resnet50/Dataset_TrainModel/New_Fruitfly_dataset2_NoDorsalView', help='Dataset root directory')
    parser.add_argument('--model_type', type=str, default='resnet', 
                        choices=['efficientnet', 'resnet', 'swin','mobilenet','inception','vit','convnext'], help='Model type')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=60, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', 
                        help='Training device')
    parser.add_argument('--output_dir', type=str, default='outputs_lr0001/', help='Output directory')
    parser.add_argument('--save_interval', type=int, default=5, 
                        help='Epoch interval to save model checkpoints (default: 5)')  
    return parser.parse_args()

 # Transformations of training data
def get_data_loaders(data_dir, batch_size, model_type='resnet'):
        # Inception requires 299x299 input size
    if model_type == 'inception':
        input_size = 299
    else:
        input_size = 224
    # Train data augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Val and test data transformation
    val_test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_test_transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=val_test_transform)
    
    # Create data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Get class names
    class_names = train_dataset.classes
    num_classes = len(class_names)
    
    return train_loader, val_loader, test_loader, class_names, num_classes

# Get pretrained model and modify the last classification layer
def get_model(model_type, num_classes, device):
    if model_type == 'efficientnet':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    elif model_type == 'resnet':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_type == 'swin':
        model = models.swin_b(weights=models.Swin_B_Weights.DEFAULT)
        model.head = nn.Linear(model.head.in_features, num_classes)
        
    elif model_type == 'mobilenet':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif model_type == 'inception':
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        # main output layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        # auxiliary output layer
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
        
    elif model_type == 'vit':
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        
    elif model_type == 'convnext':
        model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)       
    return model.to(device)

# Train function
def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs, save_dir, save_interval=5,model_type='resnet'):
    since = time.time()
    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Record training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has train and val phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    if model_type == 'inception' and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)
                    
                    # Backward pass + optimization (only in train phase)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Record history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
                # Update learning rate scheduler
                if scheduler is not None:
                    scheduler.step()
                    current_lr = optimizer.param_groups[0]['lr']
                    history['lr'].append(current_lr)
                    print(f'Current learning rate: {current_lr:.6f}')
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                
            # If best val accuracy, save model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # Save best model and delete previous best model
                torch.save(model.state_dict(), best_model_path)
                print(f'New best model saved with accuracy: {best_acc:.4f}')
                
        if save_interval > 0 and (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f'epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Checkpoint saved at epoch {epoch+1} to {checkpoint_path}')
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val accuracy: {best_acc:4f}')
    
    # Save training history
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f)
    
    # Plot training process
    plot_training_history(history, save_dir)
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Plot training history
def plot_training_history(history, save_dir):
    # Set plot style
    plt.style.use('seaborn-v0_8-dark')
    # Try to use Times New Roman font, if not exist, use default font
    try:
        times_font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
        if not os.path.exists(times_font_path):
            times_font_path = fm.findfont('Times New Roman')
        font_prop = fm.FontProperties(fname=times_font_path)
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
    except:
        # If font setting failed, use default font
        font_prop = fm.FontProperties()
    
    # Create square subplot - modify figsize to square ratio
    plt.figure(figsize=(12, 6))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss', fontproperties=font_prop, fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontproperties=font_prop, fontsize=12)
    plt.xlabel('Epoch', fontproperties=font_prop, fontsize=12)
    plt.legend(prop=font_prop)
    plt.grid(True, alpha=0.3)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy', linewidth=2)
    plt.plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy', fontproperties=font_prop, fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontproperties=font_prop, fontsize=12)
    plt.xlabel('Epoch', fontproperties=font_prop, fontsize=12)
    plt.legend(prop=font_prop)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # Save PNG format (high DPI)
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    # Save SVG format
    plt.savefig(os.path.join(save_dir, 'training_history.svg'), format='svg', bbox_inches='tight')
    plt.close()

def evaluate_model(model, test_loader, criterion, device, class_names, save_dir):
    model.eval()
    
    test_loss = 0.0
    test_corrects = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            test_loss += loss.item() * inputs.size(0)
            test_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_corrects.double() / len(test_loader.dataset)
    
    print(f'Test Loss: {test_loss:.4f} Test Accuracy: {test_acc:.4f}')
    
    # Analyze test set class distribution
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    
    print(f"\nTest set class distribution:")
    for i, (label_idx, count) in enumerate(zip(unique_labels, counts)):
        print(f"{class_names[label_idx]}: {count} samples")
    
    # Calculate detailed classification metrics - use weighted average
    print("\n=== Detailed Classification Metrics ===")
    
    # Overall metrics - mainly focus on weighted average
    overall_accuracy = accuracy_score(all_labels, all_preds)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    print(f"\nMain Evaluation Metrics (Weighted Average):")
    print(f"Accuracy: {overall_accuracy:.4f}")
    print(f"Precision (Weighted): {precision_weighted:.4f}")
    print(f"Recall (Weighted): {recall_weighted:.4f}")
    print(f"F1-Score (Weighted): {f1_weighted:.4f}")
    
    # Detailed metrics for each class
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    print(f"\nDetailed Classification Metrics:")
    print(f"{'Class':<25} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'Samples':<10}")
    print("-" * 75)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<25} {precision_per_class[i]:<10.4f} {recall_per_class[i]:<10.4f} "
              f"{f1_per_class[i]:<10.4f} {support_per_class[i]:<10}")
    
    # Calculate precision for each class
    class_correct = list(0. for i in range(len(class_names)))
    class_total = list(0. for i in range(len(class_names)))
    
    for label, pred in zip(all_labels, all_preds):
        if label == pred:
            class_correct[label] += 1
        class_total[label] += 1
    
    # Fix class distribution dictionary creation
    class_distribution = {}
    for label_idx, count in zip(unique_labels, counts):
        class_distribution[class_names[label_idx]] = int(count)
    
    # Save test results
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_acc.item(),
        'dataset_analysis': {
            'class_distribution': class_distribution
        },
        'overall_metrics': {
            'accuracy': overall_accuracy,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted
        },
        'class_metrics': {},
        'class_accuracy': {}
    }
    
    # Save detailed metrics for each class
    for i, class_name in enumerate(class_names):
        results['class_metrics'][class_name] = {
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1_score': float(f1_per_class[i]),
            'support': int(support_per_class[i])
        }
        
        if class_total[i] > 0:
            accuracy = class_correct[i] / class_total[i]
            results['class_accuracy'][class_name] = float(accuracy)
    
    # Generate confusion matrix and visualize
    plot_confusion_matrix(all_labels, all_preds, class_names, save_dir)
    
    # Save detailed classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, 
                                 output_dict=True, zero_division=0)
    results['classification_report'] = report
    
    with open(os.path.join(save_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save a readable text report
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write("=== Moth Image Classification Model Evaluation Report ===\n\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        f.write("Main Evaluation Metrics (Weighted Average):\n")
        f.write(f"Accuracy: {overall_accuracy:.4f}\n")
        f.write(f"Precision (Weighted): {precision_weighted:.4f}\n")
        f.write(f"Recall (Weighted): {recall_weighted:.4f}\n")
        f.write(f"F1-Score (Weighted): {f1_weighted:.4f}\n\n")
        f.write("Detailed Classification Metrics:\n")
        f.write(f"{'Class':<25} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'Samples':<10}\n")
        f.write("-" * 75 + "\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name:<25} {precision_per_class[i]:<10.4f} {recall_per_class[i]:<10.4f} "
                   f"{f1_per_class[i]:<10.4f} {support_per_class[i]:<10}\n")
        f.write(f"\nDetailed Classification Report:\n")
        f.write(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    
    return results

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names, save_dir):
    # Set Times New Roman font
    import matplotlib.font_manager as fm
    
    try:
        times_font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
        times_italic_font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Italic.ttf'
        
        if not os.path.exists(times_font_path):
            times_font_path = fm.findfont('Times New Roman')
            times_italic_font_path = times_font_path
        
        font_prop = fm.FontProperties(fname=times_font_path)
        font_prop_italic = fm.FontProperties(fname=times_italic_font_path)
        
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
    except:
        font_prop = fm.FontProperties()
        font_prop_italic = fm.FontProperties()
    
    # Calculate confusion matrix - original number version
    cm = confusion_matrix(y_true, y_pred)
    
    # Create chart
    plt.figure(figsize=(10, 8))
    
    # Plot heatmap - use original number (not normalized)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Confusion Matrix', fontproperties=font_prop)
    plt.ylabel('True Label', fontproperties=font_prop)
    plt.xlabel('Predicted Label', fontproperties=font_prop)
    # Set tick labels to italic
    plt.xticks(fontproperties=font_prop_italic, rotation=45, ha='right')
    plt.yticks(fontproperties=font_prop_italic)
    
    plt.tight_layout()
    # Save PNG format (high DPI)
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300)
    # Save SVG format
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.svg'), format='svg')
    plt.close()

def main():
    args = parse_args()
    set_seed(42)
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, f"{args.model_type}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)
        
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, class_names, num_classes = get_data_loaders(
        args.data_dir, args.batch_size, args.model_type)
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    # Save class information
    with open(os.path.join(output_dir, 'classes.json'), 'w') as f:
        json.dump(class_names, f)
    
    print(f"Detected {num_classes} classes: {class_names}")
    
    # Initialize model
    print(f"Initializing {args.model_type} model...")
    model = get_model(args.model_type, num_classes, args.device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Use AdamW optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Use CosineAnnealingLR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.lr * 0.01
     )
    
    # use StepLR 
    #scheduler = optim.lr_scheduler.StepLR(
    #	optimizer, step_size = 10, gamma = 0.1, last_epoch = -1
    #)
    
    # Train model
    print("Starting training...")
    model = train_model(model, dataloaders, criterion, optimizer, 
                        scheduler, args.device, args.num_epochs, output_dir,
                        save_interval=args.save_interval, model_type=args.model_type)#增加保存间隔参数
    
    # Test model
    print("Evaluating model on test set...")
    results = evaluate_model(model, test_loader, criterion, args.device, class_names, output_dir)
    
    print(f"All results and models saved to {output_dir}")

if __name__ == "__main__":
    main() 
