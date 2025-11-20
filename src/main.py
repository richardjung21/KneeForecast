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

ignore_list = ['bmi', 'change', "time_0", "time_1", "kl_0", "kl_1"]

def regularize_img(x):
    return (x-x.min())/(x.max()-x.min())*2-1

def clip_img(x):
    return x.clamp(-1, 1)

def init_model(args):
    num_of_classes = []
    for key, value in concept_classes.items():
        if key not in ignore_list:
            num_of_classes.append(value)
    model = Predictor(num_of_classes, args.regressors)
    return model

def load_data(args):
    train_dataset = KneeXRDataset(pickle_dir=args.data_dir, target_dim=args.target_dim, split='train', binary=args.binary, split_by_site=args.split_by_site)
    val_dataset = KneeXRDataset(pickle_dir=args.data_dir, target_dim=args.target_dim, split='val', binary=args.binary, split_by_site=args.split_by_site)
    test_dataset = KneeXRDataset(pickle_dir=args.data_dir, target_dim=args.target_dim, split='test', binary=args.binary, split_by_site=args.split_by_site)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    return train_dataloader, val_dataloader, test_dataloader


def train_concepts(train_dataloader, model, args, device):
    model.train()
    lpips = LearnedPerceptualImagePatchSimilarity('vgg').to(device)
    
    for epoch in range(args.num_epochs):
        # Start Training epoch
        for img0, img1, metadata in tqdm(train_dataloader):
            img0 = img0.to(device)
            img1 = img1.to(device)

            img0 = regularize_img(img0)
            img1 = regularize_img(img1)

            bmi = metadata['bmi'].to(device)
            change = metadata['change'].to(device)
            time_0 = metadata['time_0'].to(device)
            time_1 = metadata['time_1'].to(device)
            kl_0 = metadata['kl_0'].to(device)
            kl_1 = metadata['kl_1'].to(device)

            classification = []
            regression = []
            for key, value in metadata.items():
                if key not in ignore_list and key in concept_classes:
                    classification.append(value)
                elif key not in ignore_list:
                    regression.append(value)
            
            classification = torch.stack(classification, dim=1).to(device)
            regression = torch.stack(regression, dim=1).to(device)

            concept_loss, y = model.train_concepts(img0, classification, regression)
            
            y = clip_img(y)

            lpips_loss = lpips(img0, y)

            
    
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader, val_dataloader, test_dataloader = load_data(args)
    model = init_model(args)
    model.to(device)

    train_concepts(train_dataloader, model, args, device)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/mnt/sdg2/seungho/Dataset/Forecasting/list_dict_reduced.pkl")
    parser.add_argument("--target_dim", type=tuple, default=(256, 256))
    parser.add_argument("--binary", type=bool, default=False)
    parser.add_argument("--split_by_site", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--regressors", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=100)
    args = parser.parse_args()
    
    main(args)