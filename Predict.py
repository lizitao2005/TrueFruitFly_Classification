import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image
import argparse
import random
import shutil
import glob

# Parse command line arguments
parser = argparse.ArgumentParser(description='Insect Classification Testing')
parser.add_argument('--model_path', type=str, default="best.pth", required=False, help='Path to trained model (single model)')
parser.add_argument('--model_dir', type=str, default=None, help='Directory containing multiple model files')
parser.add_argument('--model_root', type=str, default=None, help='Root directory containing subdirectories with models')
parser.add_argument('--test_dir', type=str, default="./test_dataset1", required=False, help='Path to test dataset (single dataset)')
parser.add_argument('--test_dirs', type=str, default=None, nargs='+', help='List of test directories for multiple datasets')
parser.add_argument('--test_root', type=str, default=None, help='Root directory containing subdirectories with test datasets')
parser.add_argument('--class_names', nargs='+', default=["1001", "1002", "1003","1005","1006","1007","1009","1010","1011","1012","1013","1018","1019","1020","1021","1022","1024","1025","1026","1027","1028","1029","1030","1031","1032","1034"], required=False, help='List of class names in order')
parser.add_argument('--results_dir', type=str, default='./test_results', help='Directory to save test results')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')
parser.add_argument('--model_pattern', type=str, default='*.pth', help='Pattern for model files (e.g., *.pth, best*.pth)')
parser.add_argument('--model_type', type=str, default='resnet', choices=['auto', 'resnet', 'efficientnet', 'swin','mobilenet','convnext'], 
                    help='Model type to load. Use "auto" to detect from filename')
args = parser.parse_args()

# Check GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set global font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

# Data preprocessing
data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    #transforms.Resize(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Custom test dataset class
class InsectTestDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_names=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = sorted(os.listdir(root_dir)) if class_names is None else class_names
        self.image_paths = []
        self.labels = []
        
        # Check if class names match parameters
        if class_names is not None and set(os.listdir(root_dir)) != set(class_names):
            print(f"Warning: Class names mismatch! Directory: {os.listdir(root_dir)}, Provided: {class_names}")
        
        # Map class names to indices
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}
        
        # Collect all image paths and labels
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])
        
        print(f"Found {len(self.image_paths)} images in {len(self.class_names)} classes")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, img_path

# Load model
def load_model(model_path, num_classes, model_type):

    if model_type == 'resnet':
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    elif model_type == 'efficientnet':

        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif model_type == 'swin':
        model = models.swin_b(weights=None)
        model.head = nn.Linear(model.head.in_features, num_classes)
        
    elif model_type == 'mobilenet':
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)    
        
    elif model_type == 'convnext':
        model = models.convnext_base(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)  
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    if torch.cuda.is_available():
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    return model

# Test function
def test_model(model, test_loader, class_names, results_dir):
    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []
    
    # For storing misclassified samples
    misclassified = []
    
    with torch.no_grad():
        for inputs, labels, paths in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_paths.extend(paths)
            
            # Record misclassified samples
            for i in range(len(preds)):
                if preds[i] != labels[i]:
                    misclassified.append({
                        'path': paths[i],
                        'true_label': labels[i].item(),
                        'pred_label': preds[i].item(),
                        'prob': probs[i][preds[i]].item(),
                        'true_class': class_names[labels[i].item()],
                        'pred_class': class_names[preds[i].item()]
                    })
    
    return all_labels, all_preds, all_probs, all_paths, misclassified

# Visualize misclassified samples
def visualize_misclassified(misclassified, num_samples=20, results_dir=None, class_names=None):
    if not misclassified:
        print("No misclassified samples found!")
        return
        
    if len(misclassified) > num_samples:
        samples = random.sample(misclassified, num_samples)
    else:
        samples = misclassified
        
    plt.figure(figsize=(15, 10))
    
    for i, sample in enumerate(samples):
        img = Image.open(sample['path'])
        plt.subplot(4, 5, i+1)
        plt.imshow(img)
        plt.title(f"True: {sample['true_class']}\nPred: {sample['pred_class']}\nProb: {sample['prob']:.2f}", fontfamily='Times New Roman')
        plt.axis('off')
        
        # 保存错误样本
        if results_dir:
            dest_dir = os.path.join(results_dir, 'misclassified', 
                                   f"true_{sample['true_class']}", 
                                   f"pred_{sample['pred_class']}")
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(sample['path'], dest_dir)
    
    if results_dir:
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'misclassified_samples.png'), dpi=300, bbox_inches='tight')
        plt.close()

def run_single_test(model_path, test_dir, results_dir, class_names, batch_size, num_workers, model_type='auto'):
    # Create result directory
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'misclassified'), exist_ok=True)
    
    # Check if test directory exists
    if not os.path.isdir(test_dir):
        raise NotADirectoryError(f"Test directory not found: {test_dir}")
    
    # Create test dataset and data loader
    test_dataset = InsectTestDataset(test_dir, transform=data_transform, class_names=class_names)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers)
    
    # Load model
    print(f"Loading model from {model_path}")
    model = load_model(model_path, len(class_names), model_type)
    
    # Run test
    print(f"Starting testing on {test_dir}...")
    all_labels, all_preds, all_probs, all_paths, misclassified = test_model(
        model, test_loader, class_names, results_dir
    )
    
    # Save complete test results
    test_results = []
    for i in range(len(all_paths)):
        test_results.append({
            'image_path': all_paths[i],
            'true_label': all_labels[i],
            'pred_label': all_preds[i],
            'true_class': class_names[all_labels[i]],
            'pred_class': class_names[all_preds[i]],
            'confidence': all_probs[i][all_preds[i]],
            'correct': 1 if all_labels[i] == all_preds[i] else 0
        })
    
    test_results_df = pd.DataFrame(test_results)
    test_results_df.to_csv(os.path.join(results_dir, 'test_results.csv'), index=False)
    
    # Calculate overall accuracy
    accuracy = np.mean(np.array(all_labels) == np.array(all_preds))
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Generate classification report
    report = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(results_dir, 'test_classification_report.csv'))
    print("Classification report saved")
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.set(font='Times New Roman')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontfamily='Times New Roman', fontsize=16)
    plt.xlabel('Predicted', fontfamily='Times New Roman', fontsize=14)
    plt.ylabel('True', fontfamily='Times New Roman', fontsize=14)
    plt.xticks(rotation=45, fontfamily='Times New Roman')
    plt.yticks(fontfamily='Times New Roman')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Confusion matrix saved")
    
    # Process misclassified samples
    print(f"Found {len(misclassified)} misclassified samples")
    visualize_misclassified(misclassified, results_dir=results_dir, class_names=class_names)
    print("Misclassified samples visualized")
    
    # Save error classification statistics
    if misclassified:
        misclassified_df = pd.DataFrame(misclassified)
        misclassified_df.to_csv(os.path.join(results_dir, 'misclassified_details.csv'), index=False)
        
        # Group by class and count errors
        error_stats = misclassified_df.groupby('true_class').size().reset_index(name='count')
        error_stats['total'] = error_stats['true_class'].map(
            test_results_df['true_class'].value_counts())
        error_stats['error_rate'] = error_stats['count'] / error_stats['total']
        error_stats.to_csv(os.path.join(results_dir, 'error_statistics.csv'), index=False)
        
        # Plot error rate bar chart
        plt.figure(figsize=(10, 6))
        sns.barplot(x='true_class', y='error_rate', data=error_stats)
        plt.title('Error Rate by Class', fontfamily='Times New Roman', fontsize=16)
        plt.xlabel('Class', fontfamily='Times New Roman', fontsize=14)
        plt.ylabel('Error Rate', fontfamily='Times New Roman', fontsize=14)
        plt.xticks(rotation=45, fontfamily='Times New Roman')
        plt.yticks(fontfamily='Times New Roman')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'error_rate_by_class.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save final report
    with open(os.path.join(results_dir, 'test_summary.txt'), 'w') as f:
        f.write(f"Insect Classification Test Report\n")
        f.write(f"=================================\n\n")
        f.write(f"Test Parameters:\n")
        f.write(f"- Classes: {len(class_names)} ({', '.join(class_names)})\n")
        f.write(f"- Test images: {len(test_dataset)}\n")
        f.write(f"- Batch size: {batch_size}\n")
        f.write(f"- Model: {model_path}\n")
        f.write(f"- Model type: {model_type}\n\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n")
        f.write(f"Number of misclassified samples: {len(misclassified)}\n")
        f.write(f"Error rate: {len(misclassified)/len(test_dataset):.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report_df.to_string())
        if misclassified:
            f.write("\n\nError Statistics:\n")
            f.write(error_stats.to_string(index=False))
    
    print(f"Test results saved to {results_dir}")
    return accuracy, report

def main():
    all_results = []
    model_paths = []
    model_names = []
    model_types = [] 
    
    # Process model root directory
    if args.model_root:
        print(f"Scanning model root directory: {args.model_root}")
        # Iterate over all subdirectories in the model root directory
        for root, dirs, files in os.walk(args.model_root):
            # Find model files in each subdirectory
            model_files = glob.glob(os.path.join(root, args.model_pattern))
            for model_file in model_files:
                model_paths.append(model_file)
                # Create model name: relative path + file name
                rel_path = os.path.relpath(model_file, args.model_root)
                model_name = os.path.splitext(rel_path)[0].replace(os.sep, '-')
                model_names.append(model_name)
                model_types.append(args.model_type)
    
    # Process model directory
    if args.model_dir:
        # Get all model files in the directory
        model_files = glob.glob(os.path.join(args.model_dir, args.model_pattern))
        for model_file in model_files:
            model_paths.append(model_file)
            model_names.append(os.path.splitext(os.path.basename(model_file))[0])
            model_types.append(args.model_type)
    
    # Process single model
    if args.model_path and not args.model_root and not args.model_dir:
        model_paths.append(args.model_path)
        model_names.append(os.path.splitext(os.path.basename(args.model_path))[0])
        model_types.append(args.model_type)
    
    if not model_paths:
        print("No models found! Please specify model paths.")
        return
    
    print(f"Found {len(model_paths)} models for testing")
    
    # Determine test directory list
    test_dirs = []
    test_names = []
    
    # Process test root directory
    if args.test_root:
        print(f"Scanning test root directory: {args.test_root}")
        # Iterate over all subdirectories in the test root directory
        for root, dirs, files in os.walk(args.test_root):
            # Only add directories that contain subdirectories (i.e. test set directories)
            if os.listdir(root) and all(os.path.isdir(os.path.join(root, d)) for d in os.listdir(root)):
                test_dirs.append(root)
                test_names.append(os.path.relpath(root, args.test_root).replace(os.sep, '-'))
    
    # Process multiple test directories
    if args.test_dirs:
        for test_dir in args.test_dirs:
            test_dirs.append(test_dir)
            test_names.append(os.path.basename(test_dir.rstrip('/')))
    
    # Process single test directory
    if args.test_dir and not args.test_root and not args.test_dirs:
        test_dirs.append(args.test_dir)
        test_names.append(os.path.basename(args.test_dir.rstrip('/')))
    
    if not test_dirs:
        print("No test datasets found! Please specify test paths.")
        return
    
    print(f"Found {len(test_dirs)} test datasets for testing")
    
    # Iterate over all model and test directory combinations
    for model_path, model_name, model_type in zip(model_paths, model_names, model_types):
        for test_dir, test_name in zip(test_dirs, test_names):
            # Create result directory name: model_name-pre-test_folder name
            result_dir_name = f"{model_name}-pre-{test_name}"
            results_dir = os.path.join(args.results_dir, result_dir_name)
            
            print(f"\n{'='*50}")
            print(f"Running test: Model={model_name}, Type={model_type}, Dataset={test_name}")
            print(f"Results will be saved to: {results_dir}")
            print(f"{'='*50}\n")
            
            try:
                # Run test
                accuracy, report = run_single_test(
                    model_path=model_path,
                    test_dir=test_dir,
                    results_dir=results_dir,
                    class_names=args.class_names,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    model_type=model_type
                )
                
                # Record results for summary
                all_results.append({
                    'model': model_name,
                    'model_type': model_type,
                    'dataset': test_name,
                    'accuracy': accuracy,
                    'precision_macro': report['macro avg']['precision'],
                    'recall_macro': report['macro avg']['recall'],
                    'f1_macro': report['macro avg']['f1-score'],
                    'results_dir': results_dir
                })
            except Exception as e:
                print(f"Error testing model {model_name} on dataset {test_name}: {str(e)}")
                continue
    
    # Save summary results
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_file = os.path.join(args.results_dir, 'all_results_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"\n{'='*50}")
        print(f"All tests completed. Summary saved to: {summary_file}")
        print(summary_df[['model', 'model_type', 'dataset', 'accuracy']])
        print(f"{'='*50}")
        
        # Plot accuracy comparison chart
        if len(all_results) > 1:
            sns.set(font='Times New Roman')
            rotation = 45 
            
            # Accuracy comparison chart 1
            plt.figure(figsize=(14, 8))
            sns.barplot(x='model', y='accuracy', hue='dataset', data=summary_df)
            plt.title('Model Accuracy Comparison Across Datasets', fontfamily='Times New Roman', fontsize=16)
            plt.xlabel('Model', fontfamily='Times New Roman', fontsize=14)
            plt.ylabel('Accuracy', fontfamily='Times New Roman', fontsize=14)
            plt.ylim(0, 1.0)
            plt.xticks(rotation=rotation, fontfamily='Times New Roman')
            plt.yticks(fontfamily='Times New Roman')
            plt.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left', title_fontproperties={'family': 'Times New Roman'}, prop={'family': 'Times New Roman'})
            plt.tight_layout()
            plt.savefig(os.path.join(args.results_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("Accuracy comparison chart saved")
            
            # Group by model type and plot accuracy comparison
            plt.figure(figsize=(14, 8))
            sns.barplot(x='model_type', y='accuracy', hue='dataset', data=summary_df)
            plt.title('Model Type Accuracy Comparison Across Datasets', fontfamily='Times New Roman', fontsize=16)
            plt.xlabel('Model Type', fontfamily='Times New Roman', fontsize=14)
            plt.ylabel('Accuracy', fontfamily='Times New Roman', fontsize=14)
            plt.ylim(0, 1.0)
            plt.xticks(fontfamily='Times New Roman')
            plt.yticks(fontfamily='Times New Roman')
            plt.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left', title_fontproperties={'family': 'Times New Roman'}, prop={'family': 'Times New Roman'})
            plt.tight_layout()
            plt.savefig(os.path.join(args.results_dir, 'accuracy_by_model_type.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("Accuracy by model type chart saved")
            
            # Save an additional horizontal bar chart
            plt.figure(figsize=(12, max(6, len(summary_df) * 0.3)))
            sns.barplot(y='model', x='accuracy', hue='dataset', data=summary_df, orient='h')
            plt.title('Model Accuracy Comparison Across Datasets', fontfamily='Times New Roman', fontsize=16)
            plt.ylabel('Model', fontfamily='Times New Roman', fontsize=14)
            plt.xlabel('Accuracy', fontfamily='Times New Roman', fontsize=14)
            plt.xlim(0, 1.0)
            plt.xticks(fontfamily='Times New Roman')
            plt.yticks(fontfamily='Times New Roman')
            plt.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left', title_fontproperties={'family': 'Times New Roman'}, prop={'family': 'Times New Roman'})
            plt.tight_layout()
            plt.savefig(os.path.join(args.results_dir, 'accuracy_comparison_horizontal.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("Horizontal accuracy comparison chart saved")

if __name__ == "__main__":
    main()