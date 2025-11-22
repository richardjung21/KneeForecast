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
from sklearn.metrics import f1_score, mean_absolute_error, accuracy_score

concept_classes = {
    "XRATTL": 4,
    "XRATTM": 4,
    "XRCYFL": 3,
    "XRCYFM": 3,
    "XRCYTL": 3,
    "XRCYTM": 3,
    "XRJSL": 4,
    "XRJSM": 4,
    "XROSFL": 4,
    "XROSFM": 4,
    "XROSTL": 4,
    "XROSTM": 4,
    "kl_0": 5,
    "kl_1": 5,
    "change": 2
}

regression_classes = ["MCMJSW", "TPCFDS"]

ignore_list = ['bmi', 'change', "time_0", "time_1", "kl_1", "XRATTL", "XRATTM", "XRCYFL", "XRCYFM", "XRCYTL", "XRCYTM"]

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


def val_concepts(val_dataloader, model, args, device, best_val_loss, epoch):
    model.eval()
    total_concept_loss = 0
    save_dir = f"./results/{args.save_dir}"
    images_dir = os.path.join(save_dir, "images", f"epoch_{epoch}")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    tp = [0, 0, 0, 0, 0, 0, 0]
    fp = [0, 0, 0, 0, 0, 0, 0]
    fn = [0, 0, 0, 0, 0, 0, 0]
    tn = [0, 0, 0, 0, 0, 0, 0]
    regression_error = [0, 0]
    
    with torch.no_grad():
        pbar = tqdm(val_dataloader, desc="Validation")
        for i, (img0, img1, metadata) in enumerate(pbar):
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

            concept_loss, reconstruction, batch_class_preds, batch_class_targets, reg_preds, reg_targets = model.validate_concepts(img0, classification, regression)

            reconstruction = torch.clamp((reconstruction[0]+1)/2, 0, 1)
            img0 = torch.clamp((img0[0]+1)/2, 0, 1)
            
            if i % (len(pbar)//10) == 0:
                gt_image_path = os.path.join(images_dir, f"Iteration_{i}_gt.png")
                recon_image_path = os.path.join(images_dir, f"Iteration_{i}_recon.png")
                plt.imsave(gt_image_path, img0.permute(1, 2, 0).cpu().numpy())
                plt.imsave(recon_image_path, reconstruction.permute(1, 2, 0).cpu().numpy())
            
            total_concept_loss += concept_loss.item()

            predicted_classes = []
            for class_preds in batch_class_preds:
                predicted_classes.append(torch.argmax(class_preds, dim=1))

            predicted_classes = torch.stack(predicted_classes, dim=1)
            class_targets = torch.stack(batch_class_targets, dim=1).long()

            for i in range(predicted_classes.shape[1]):
                tp[i] += torch.sum(predicted_classes[:, i] == class_targets[:, i]).item()
                fp[i] += torch.sum(predicted_classes[:, i] != class_targets[:, i]).item()
                fn[i] += torch.sum(predicted_classes[:, i] != class_targets[:, i]).item()
                tn[i] += torch.sum(predicted_classes[:, i] == class_targets[:, i]).item()

            regression_error[0] += torch.sum(torch.abs(reg_preds[:, 0] - reg_targets[:, 0])).item()
            regression_error[1] += torch.sum(torch.abs(reg_preds[:, 1] - reg_targets[:, 1])).item()
            
    avg_concept_loss = total_concept_loss / len(val_dataloader)
    regression_error = [regression_error[0]/len(val_dataloader), regression_error[1]/len(val_dataloader)]
    specificity = [0, 0, 0, 0, 0, 0, 0]
    sensitivity = [0, 0, 0, 0, 0, 0, 0]
    f1 = [0, 0, 0, 0, 0, 0, 0]
    acc = [0, 0, 0, 0, 0, 0, 0]
    
    for i in range(len(tp)):
        specificity[i] = tn[i] / (tn[i] + fp[i])
        sensitivity[i] = tp[i] / (tp[i] + fn[i])
        f1[i] = 2 * (specificity[i] * sensitivity[i]) / (specificity[i] + sensitivity[i])
        acc[i] = (tp[i] + tn[i]) / (tp[i] + tn[i] + fp[i] + fn[i])
    
    f1_string = [f"{i:.4f}" for i in f1]
    acc_string = [f"{i:.4f}" for i in acc]
    regression_error_string = [f"{i:.4f}" for i in regression_error]

    print(f"Validation Results - Concept Loss: {avg_concept_loss:.4f}")
    print(f"Average F1 per task: {f1_string}")
    print(f"Average Accuracy per task: {acc_string}")
    print(f"Average MAE per task: {regression_error_string}")
    
    save_dir = f"./results/{args.save_dir}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save latest model
    torch.save(model.state_dict(), os.path.join(save_dir, "latest_model.pth"))
    
    if avg_concept_loss < best_val_loss:
        best_val_loss = avg_concept_loss
        torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
        print(f"New best model saved with loss: {best_val_loss:.4f}")
    
    model.train()
    return best_val_loss


def train_concepts(train_dataloader, val_dataloader, model, args, device):
    model.train()
    lpips = LearnedPerceptualImagePatchSimilarity('vgg').to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    
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
        best_val_loss = val_concepts(val_dataloader, model, args, device, best_val_loss, epoch)
    
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader, val_dataloader, test_dataloader = load_data(args)
    model = init_model(args)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))
    model.to(device)

    best_val_loss = float('inf')
    if args.mode == "train":
        train_concepts(train_dataloader, val_dataloader, model, args, device)
    elif args.mode == "val":
        best_val_loss = val_concepts(val_dataloader, model, args, device, best_val_loss, 0)
    
    

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
    parser.add_argument("--mode", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--checkpoint", type=str, default="")
    args = parser.parse_args()
    
    main(args)