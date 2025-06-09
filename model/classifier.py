import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.amp import autocast, GradScaler 
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix
from config import PIN_MEMORY

class Classifier(nn.Module):
    def __init__(self, in_dim, n_cls, dropout1=0.6, dropout2=0.3):
        super().__init__()
        self.dropout1  = nn.Dropout(dropout1)
        self.fc1 = nn.Linear(in_dim, 256)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout2)
        self.fc2 = nn.Linear(256, n_cls)

    def forward(self, x):
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

class FeatExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature = nn.Sequential(*list(resnet.children())[:-1])
        
    def forward(self, x):
        x = self.feature(x)
        return x.view(x.size(0), -1)

def train_epoch(loader, feature_extractor, classifier, optimizer, class_weights, device_obj):
    is_cuda = (device_obj.type == 'cuda')
    scaler = GradScaler(enabled=is_cuda)
    
    feature_extractor.train()
    classifier.train()
    total_loss, correct, num_samples = 0.0, 0, 0
    
    for x_cpu, y in tqdm(loader, desc="[TrainBatches]", leave=False, ncols=100):
        x = x_cpu.to(device_obj, non_blocking=PIN_MEMORY) 
        y = y.to(device_obj, non_blocking=PIN_MEMORY)
        batch_s = x.size(0)
        num_samples += batch_s
        
        optimizer.zero_grad()
        
        with autocast(device_type=device_obj.type, enabled=is_cuda):
            z = feature_extractor(x)
            out = classifier(z)
            loss = F.cross_entropy(out, y, weight=class_weights)
            
        if is_cuda:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        total_loss += loss.item() * batch_s
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        
    return (total_loss / num_samples if num_samples > 0 else 0), \
           (correct / num_samples if num_samples > 0 else 0)

def validate_epoch(loader, feature_extractor, classifier, device_obj):
    feature_extractor.eval()
    classifier.eval()
    
    total_loss, correct, num_samples = 0.0, 0, 0
    all_preds, all_gts = [], []
    is_cuda = (device_obj.type == 'cuda')
    
    with torch.no_grad():
        for x_cpu, y in tqdm(loader, desc="[ValBatches]", leave=False, ncols=100):
            x = x_cpu.to(device_obj, non_blocking=PIN_MEMORY)
            y = y.to(device_obj, non_blocking=PIN_MEMORY)
            batch_s = x.size(0)
            num_samples += batch_s

            with autocast(device_type=device_obj.type, enabled=is_cuda):
                z = feature_extractor(x)
                out = classifier(z)
                loss = F.cross_entropy(out, y)
                
            total_loss += loss.item() * batch_s
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            
            all_preds.extend(preds.cpu().numpy())
            all_gts.extend(y.cpu().numpy())
            
    avg_loss = total_loss / num_samples if num_samples > 0 else 0
    accuracy = correct / num_samples if num_samples > 0 else 0
    f1 = f1_score(all_gts, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_gts, all_preds) if all_gts and all_preds else None
    
    return avg_loss, accuracy, f1, cm