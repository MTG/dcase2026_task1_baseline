import collections.abc
from collections import defaultdict
import json
import os
from pyexpat import model
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from losses import CrossEntropyLoss
from utils import get_subconfig, set_seed, build_class_to_topclass_mapping, build_class_to_topclass_tensor
from models import BaseClassifier
from dataset_utils import HATRDataset
from evaluate import evaluate_model

# Paths
dataset_name = get_subconfig("active_dataset")
dataset_path = get_subconfig("datasets")[dataset_name]["metadata_csv"]
color_dict_path = get_subconfig("color_dict_path")
top_color_dict_path = get_subconfig("top_color_dict_path")

data_dir = get_subconfig("output_path")
prepared_dataset_path = os.path.join(data_dir, get_subconfig("processed_dataset_csv"))
class_dict_json = os.path.join(data_dir, get_subconfig("class_dict_json"))
top_class_dict_json = os.path.join(data_dir, get_subconfig("top_class_dict_json"))
subclass_json = os.path.join(data_dir, get_subconfig("top_class_subclass_dict_json"))


def init_weights(model):
    if isinstance(model, nn.Conv2d):
        nn.init.kaiming_normal_(model.weight, mode='fan_out')
    elif isinstance(model, nn.Linear):
        nn.init.xavier_uniform_(model.weight)

def make_serializable(obj, decimals=6):
    """Recursively convert tensors, numpy arrays, and numbers to JSON-serializable types with rounding."""
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu().numpy()
        return make_serializable(obj, decimals)
    elif isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return round(float(obj), decimals)
        else:
            return [make_serializable(x, decimals) for x in obj]
    elif isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, int):
        return obj
    elif isinstance(obj, collections.abc.Mapping):
        return {k: make_serializable(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, collections.abc.Iterable) and not isinstance(obj, (str, bytes)):
        return [make_serializable(x, decimals) for x in obj]
    else:
        return obj
    
def train_model(model, train_loader, val_loader, device,
                num_epochs=100, lr=0.001, classification_weight=1.0, classification_criterion=None, 
                output_dir='model_output', scheduler_type='plateau', patience=10, early_stopping_factor=5):
    """
    Train a model with validation, LR scheduling, checkpointing, and early stopping.

    Tracks training loss, validation accuracy, and (if available) attention statistics.
    Saves the best model with a config, and training history to `output_dir`.

    Returns:
        best_accuracy (float), history (dict), model (nn.Module)
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    if scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=patience, verbose=True)
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    else:
        scheduler = None
    
    best_accuracy = 0.0
    epochs_without_improvement = 0
    history = defaultdict(list)

    for epoch in range(num_epochs):
        model.train()
        losses = defaultdict(float)
        total_samples = 0

        attn_audio_epoch = []
        attn_text_epoch = []

        for data in train_loader:
            class_labels = data['class_idx'].to(device)
            audio_emb = data.get('audio_embedding', None)
            text_emb = data.get('text_embedding', None)
            
            if audio_emb is not None:
                audio_emb = audio_emb.to(device)
            if text_emb is not None:
                text_emb = text_emb.to(device)

            optimizer.zero_grad()
            
            z, class_logit, attn_scores = model(audio_emb, text_emb)
            
            # collect batch attention once per batch
            if attn_scores is not None:
                attn_audio_epoch.append(attn_scores[:, 0].detach().cpu())
                attn_text_epoch.append(attn_scores[:, 1].detach().cpu())

            total_loss = 0.0
            
            batch_size = class_labels.size(0)
            total_samples += batch_size

            if classification_criterion is not None:
                cls_loss = classification_criterion(class_logit, class_labels)
                losses['cls'] += cls_loss.item() * batch_size
                total_loss += classification_weight * cls_loss

            total_loss.backward()
            optimizer.step()
            losses['total'] += total_loss.item() * batch_size

        # per-epoch attention summary
        if attn_audio_epoch:
            attn_audio_epoch = torch.cat(attn_audio_epoch, dim=0)
            attn_text_epoch = torch.cat(attn_text_epoch, dim=0)
            history["attention_audio"].append(attn_audio_epoch.mean(0).numpy())
            history["attention_text"].append(attn_text_epoch.mean(0).numpy())

        num_batches = len(train_loader)
        for k in losses:
            history[f'train_{k}_loss'].append(losses[k] / total_samples)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                labels = data['class_idx'].to(device)
                audio_emb = data.get('audio_embedding', None)
                text_emb = data.get('text_embedding', None)
                
                if audio_emb is not None:
                    audio_emb = audio_emb.to(device)
                if text_emb is not None:
                    text_emb = text_emb.to(device)

                _, class_logit, _ = model(audio_emb, text_emb)
                    
                _, predicted = torch.max(class_logit.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        history['val_accuracy'].append(val_accuracy)

        with open(os.path.join(output_dir, "history.json"), "w") as f:
            json.dump(make_serializable(history), f, indent=2)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Val acc: {val_accuracy:.2f}%")
        # for k in losses:
        #     if losses[k] > 0:
        #         print(f"  {k.capitalize()} loss: {losses[k] / total_samples:.4f}")
        # print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        if scheduler:
            if scheduler_type == 'plateau':
                scheduler.step(val_accuracy)
            else:
                scheduler.step()

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            model_config = {'hidden_size': hidden_size, 'num_classes': len(class_dict),
                'emb_size_audio': emb_size_audio, 'emb_size_text': emb_size_text, 
                'dropout': dropout, 'use_batch_norm': True,'mode': mode,
            }

            torch.save({
                'model_state': model.state_dict(),
                'config': model_config,
            }, os.path.join(output_dir, "best_model.pth"))

            print(f"  New best model saved")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience * early_stopping_factor:
                print("Early stopping triggered.")
                break

    return best_accuracy, history, model


if __name__ == "__main__":
    seed = set_seed()  # For reproducibility

    with open(class_dict_json, 'r') as f:
        class_dict = json.load(f)
    with open(top_class_dict_json, 'r') as f:
        top_class_dict = json.load(f)

    modes = ['both', 'audio']  # Change this to test single modalities if needed
    model_output = './model_output'  # Base directory for model outputs  # f'model_output_{current_time}'

    batch_size = 64
    num_epochs = 100
    learning_rate = 0.001
    classification_weight = 1
    scheduler_type = 'step'
    patience = 5
    early_stopping_factor = 3
    k_folds = 5

    full_df = pd.read_csv(prepared_dataset_path)
    # Filter by confidence if needed (note that this need numbers in the confidence column)
    # full_df_with_conf= pd.read_csv(dataset_path)  # Annotation confidence is only in the original metadata
    # filtered_conf_df = full_df_with_conf[full_df_with_conf['confidence'] >= 0]
    # filtered_conf_df['sound_id'] = filtered_conf_df['sound_id'].astype(str)
    # full_df = full_df[full_df['index'].isin(filtered_conf_df['sound_id'])]

    datasets = {
        f'{dataset_name} full': {'df': full_df}
    }

    for dataset, dataset_info in datasets.items():
        print(f"\n=== Dataset: {dataset} ===")
        database = dataset_info['df']
        labels = database["class_idx"].tolist()

        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

        for mode in modes:
            print(f"\n=== Running experiments: Dataset={dataset} | Mode={mode} ===")

            for fold, (trainval_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
                print(f"\n==== Fold {fold} ====")

                trainval_labels = [labels[i] for i in trainval_idx]
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                train_idx_rel, val_idx_rel = next(sss.split(np.zeros(len(trainval_labels)), trainval_labels))
                train_idx = [trainval_idx[i] for i in train_idx_rel]
                val_idx = [trainval_idx[i] for i in val_idx_rel]

                train_df = database.iloc[train_idx].reset_index(drop=True)
                val_df = database.iloc[val_idx].reset_index(drop=True)
                test_df = database.iloc[test_idx].reset_index(drop=True)
                print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

                train_dataset = HATRDataset(train_df, aug=True, mask_pct=0.7)
                val_dataset = HATRDataset(val_df, aug=False)
                test_dataset = HATRDataset(test_df, aug=False)

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=4,
                    pin_memory=torch.cuda.is_available()
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=torch.cuda.is_available()
                )
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=torch.cuda.is_available()
                )

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                emb_size_audio = 512 if mode in ['audio', 'both'] else 0
                emb_size_text = 512 if mode in ['text', 'both'] else 0

                hidden_size = 128
                dropout = 0.1
                use_batch_norm = True

                model = BaseClassifier(
                    hidden_size=128,
                    num_classes=len(class_dict),
                    emb_size_audio=emb_size_audio,
                    emb_size_text=emb_size_text,
                    dropout=dropout,
                    use_batch_norm=use_batch_norm,
                    mode=mode
                ).to(device)

                classification_criterion = CrossEntropyLoss()

                output_dir = os.path.join(
                    model_output,
                    mode, f"fold_{fold}"
                )
                os.makedirs(output_dir, exist_ok=True)

                model_path = os.path.join(output_dir, "best_model.pth")

                init_weights(model)

                best_accuracy, history, trained_model = train_model(
                    model, train_loader, val_loader, device,
                    num_epochs=num_epochs, lr=learning_rate,
                    classification_weight=classification_weight,
                    classification_criterion=classification_criterion,
                    output_dir=output_dir,
                    scheduler_type=scheduler_type, patience=patience, early_stopping_factor=early_stopping_factor
                )
                print(f"Best validation accuracy: {best_accuracy:.2f}%")

                # Save splits for reproducibility
                splits_df = pd.concat([
                    train_df[['index']].assign(split='train'),
                    val_df[['index']].assign(split='val'),
                    test_df[['index']].assign(split='test')
                ])
                splits_df.to_csv(os.path.join(output_dir, "splits.csv"), index=False)

                # Save updated history with model info
                history['model_info'] = {
                    'model_class': trained_model.__class__.__name__,
                    'hidden_size': hidden_size,
                    'num_classes': len(class_dict),
                    'emb_size_audio': emb_size_audio,
                    'emb_size_text': emb_size_text,
                    'dropout': dropout,
                    'use_batch_norm': True,
                    'mode': mode,
                    'num_folds': k_folds,
                    'fold_id': fold,
                    'batch_size': batch_size,
                    'random_seed': seed,
                }
                
                history_path = os.path.join(output_dir, "history.json")
                with open(history_path, "w") as f:
                    json.dump(make_serializable(history), f, indent=2)

                # Testing
                class_to_top_class = build_class_to_topclass_mapping(class_dict, top_class_dict)
                subclass_to_topclass_tensor = build_class_to_topclass_tensor(class_dict, top_class_dict, device)

                metrics = evaluate_model(
                    BaseClassifier,
                    model_path,
                    test_loader,
                    device,
                    class_to_top_class,
                    output_dir=output_dir,
                    fold_id=fold,
                    class_dict=class_dict,
                )

                print("\n===== Fold Results =====")
                print(f"Final model accuracy: {metrics['accuracy']:.2f}%")
                print(f"Final model top-level accuracy: {metrics['top_accuracy']:.2f}%")
                print("========================")

    print("All experiments done!")
