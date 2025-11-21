import torch
from dataset.knee import KneeXRDataset
from models.predictor import Predictor
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import argparse
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import numpy as np
from sklearn.metrics import roc_auc_score, mean_absolute_error

concept_classes = {
    "XRATTL": 4,
    "XRATTM": 4,
    "XRCYFL": 3,
    "XRCYFM": 3,
    "XRCYTL": 3,
    "XRCYTM": 3,
    "XROSFL": 4,
    "XROSFM": 4,
    "XROSTL": 4,
    "XROSTM": 4,
    "kl_0": 5,
    "kl_1": 5,
    "change": 2
}

regression_classes = ["MCMJSW", "TPCFDS"]

ignore_list = ['bmi', 'change', "time_0", "time_1", "kl_1"]

def regularize_img(x):
    return (x-x.min())/(x.max()-x.min())*2-1

def clip_img(x):
    return x.clamp(-1, 1)

def init_model(args):
    num_of_classes = []
    for key, value in concept_classes.items():
        if key not in ignore_list:
            num_of_classes.append(value)
    model = Predictor(num_of_classes, len(regression_classes))
    return model

def load_data(args):
    train_dataset = KneeXRDataset(pickle_dir=args.data_dir, target_dim=args.target_dim, split='train', binary=args.binary, split_by_site=args.split_by_site)
    val_dataset = KneeXRDataset(pickle_dir=args.data_dir, target_dim=args.target_dim, split='val', binary=args.binary, split_by_site=args.split_by_site)
    test_dataset = KneeXRDataset(pickle_dir=args.data_dir, target_dim=args.target_dim, split='test', binary=args.binary, split_by_site=args.split_by_site)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    return train_dataloader, val_dataloader, test_dataloader


def val_concepts(val_dataloader, model, args, device, best_val_loss):
    model.eval()
    lpips = LearnedPerceptualImagePatchSimilarity('vgg').to(device)
    
    total_concept_loss = 0
    total_lpips_loss = 0
    total_loss_sum = 0
    
    all_class_preds = []
    all_class_targets = []
    all_reg_preds = []
    all_reg_targets = []
    
    with torch.no_grad():
        pbar = tqdm(val_dataloader, desc="Validation")
        for img0, img1, metadata in pbar:
            img0 = img0.to(device)
            img1 = img1.to(device)

            img0 = regularize_img(img0)
            img1 = regularize_img(img1)

            classification = []
            regression = []

            for key in concept_classes:
                if key not in ignore_list:
                    classification.append(metadata[key])
            
            for key in regression_classes:
                regression.append(metadata[key])
            
            classification = torch.stack(classification, dim=1).to(device)
            regression = torch.stack(regression, dim=1).to(device)

            concept_loss, y, batch_class_preds, batch_class_targets, batch_reg_preds, batch_reg_targets = model.validate_concepts(img0, classification, regression)
            
            y = clip_img(y)

            lpips_loss = lpips(img0, y)

            total_loss = 5*concept_loss + lpips_loss
            
            total_concept_loss += concept_loss.item()
            total_lpips_loss += lpips_loss.item()
            total_loss_sum += total_loss.item()
            
            # Append batch results to lists
            # batch_class_preds is a list of tensors (one per classifier)
            if not all_class_preds:
                all_class_preds = [[] for _ in range(len(batch_class_preds))]
                all_class_targets = [[] for _ in range(len(batch_class_targets))]
                all_reg_preds = [[] for _ in range(len(batch_reg_preds))]
                all_reg_targets = [[] for _ in range(len(batch_reg_targets))]
            
            for i in range(len(batch_class_preds)):
                all_class_preds[i].append(batch_class_preds[i].cpu())
                all_class_targets[i].append(batch_class_targets[i].cpu())
                
            for i in range(len(batch_reg_preds)):
                all_reg_preds[i].append(batch_reg_preds[i].cpu())
                all_reg_targets[i].append(batch_reg_targets[i].cpu())
            
    avg_concept_loss = total_concept_loss / len(val_dataloader)
    avg_lpips_loss = total_lpips_loss / len(val_dataloader)
    avg_total_loss = total_loss_sum / len(val_dataloader)
    
    # Calculate AUROC and MAE over the entire validation set
    final_auroc = []
    final_mae = []
    
    # Process Classifiers
    for i in range(len(all_class_preds)):
        preds = torch.cat(all_class_preds[i], dim=0).numpy()
        targets = torch.cat(all_class_targets[i], dim=0).numpy()
        
        try:
            if preds.shape[1] == 2:
                score = roc_auc_score(targets, preds[:, 1])
            else:
                score = roc_auc_score(targets, preds, multi_class='ovr')
        except ValueError:
            score = float('nan')
        final_auroc.append(score)
        
    # Process Regressors
    for i in range(len(all_reg_preds)):
        preds = torch.cat(all_reg_preds[i], dim=0).numpy()
        targets = torch.cat(all_reg_targets[i], dim=0).numpy()
        mae = mean_absolute_error(targets, preds)
        final_mae.append(mae)
    
    print(f"Validation Results - Concept Loss: {avg_concept_loss:.4f}, LPIPS Loss: {avg_lpips_loss:.4f}, Total Loss: {avg_total_loss:.4f}")
    print(f"Average AUROC per task: {final_auroc}")
    print(f"Average MAE per task: {final_mae}")
    
    save_dir = f"./results/{args.save_dir}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save latest model
    torch.save(model.state_dict(), os.path.join(save_dir, "latest_model.pth"))
    
    if avg_total_loss < best_val_loss:
        best_val_loss = avg_total_loss
        torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
        print(f"New best model saved with loss: {best_val_loss:.4f}")
    
    model.train()
    return best_val_loss


def train_concepts(train_dataloader, val_dataloader, model, args, device):
    model.train()
    lpips = LearnedPerceptualImagePatchSimilarity('vgg').to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    save_dir = f"./results/{args.save_dir}"
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_loss = float('inf')

    for epoch in range(args.num_epochs):
        # Start Training epoch
        pbar = tqdm(train_dataloader)
        for img0, img1, metadata in pbar:
            img0 = img0.to(device)
            img1 = img1.to(device)

            img0 = regularize_img(img0)
            img1 = regularize_img(img1)

            # bmi = metadata['bmi'].to(device)
            # change = metadata['change'].to(device)
            # time_0 = metadata['time_0'].to(device)
            # time_1 = metadata['time_1'].to(device)
            # kl_0 = metadata['kl_0'].to(device)
            # kl_1 = metadata['kl_1'].to(device)

            classification = []
            regression = []

            for key in concept_classes:
                if key not in ignore_list:
                    classification.append(metadata[key])
            
            for key in regression_classes:
                regression.append(metadata[key])
            
            classification = torch.stack(classification, dim=1).to(device)
            regression = torch.stack(regression, dim=1).to(device)

            concept_loss, y = model.train_concepts(img0, classification, regression)
            
            y = clip_img(y)

            lpips_loss = lpips(img0, y)

            total_loss = 5*concept_loss + lpips_loss
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            pbar.set_description(f"Epoch {epoch+1}/{args.num_epochs}")
            pbar.set_postfix({"concept_loss": concept_loss.item(), "lpips_loss": lpips_loss.item(), "total_loss": total_loss.item()})

        
        # Validation at the end of epoch
        best_val_loss = val_concepts(val_dataloader, model, args, device, best_val_loss)
    
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader, val_dataloader, test_dataloader = load_data(args)
    model = init_model(args)
    model.to(device)

    train_concepts(train_dataloader, val_dataloader, model, args, device)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/mnt/sdg2/seungho/Dataset/Forecasting/list_dict_reduced.pkl")
    parser.add_argument("--target_dim", type=tuple, default=(256, 256))
    parser.add_argument("--binary", type=bool, default=False)
    parser.add_argument("--split_by_site", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--regressors", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", type=str, default="concept")
    args = parser.parse_args()
    
    main(args)