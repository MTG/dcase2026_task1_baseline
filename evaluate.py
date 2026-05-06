import os
import torch
import pandas as pd
import numpy as np 

from utils import build_class_to_topclass_mapping, build_id_to_class_mapping, extend_subcat, intersection, get_top_level


def hierarchical_accuracy(subcat, predictions_gt, lambda_param=0.5):
  # Compute custom hierarchical accuracy for a given category.
  # This is a custom defined metric, which assigns different score if the prediction is correct at both taxonomy levels or only at the top level
  # This penalized score (when prediction is only correct at top level), can be adjusted with the lambda parameter

  prediction_scores = []
  for count, (prediction, gt) in enumerate(predictions_gt):
    if subcat == gt:
        preditcion_top_level, preditcion_sub_level = extend_subcat(prediction)
        gt_top_level, gt_sub_level = extend_subcat(gt)
        if preditcion_top_level == gt_top_level and preditcion_sub_level == gt_sub_level:
          prediction_scores.append(1) # Prediction is correct at both hierarchy levels
        elif preditcion_top_level == gt_top_level and preditcion_sub_level != gt_sub_level:
          prediction_scores.append(lambda_param) # Prediction is correct only at top level
        else:
          prediction_scores.append(0.0)
          # Note that prediction will never be correct only at the second level as all taxonomy nodes only have one possible parent
  classAcc = sum(prediction_scores)/len(prediction_scores)
  return classAcc


def hierarchical_prf(subcat, predictions_gt):
  # Compute hierarchical precision, recall and f-measure for a given category.
  # It is based on this: https://medium.com/data-science/hierarchical-performance-metrics-and-where-to-find-them-7090aaa07183 (also notation is from this article)
  # Discussed in survey paper: Silla Jr, C. N., & Freitas, A. A. (2011). A survey of hierarchical classification across different application domains. Data mining and knowledge discovery, 22(1), 31-72. (Page 59)
  # The original paper: Kiritchenko, Svetlana, Stan Matwin, and A. Fazel Famili. "Functional annotation of genes using hierarchical text categorization." Proc. of the ACL Workshop on Linking Biological Literature, Ontologies and Databases: Mining Biological Semantics. Vol. 76. 2005.

  hPP = []
  hRR = []
  for count, (prediction, gt) in enumerate(predictions_gt):
    pi = extend_subcat(prediction)
    ti = extend_subcat(gt)
    pi_intersection_ti = intersection(pi, ti)

    if subcat == prediction:  # precision: include false positives
        hP = len(pi_intersection_ti) / len(pi)
        hPP.append(hP)

    if subcat == gt:  # recall: only true positives + false negatives (all actual instances of the class)
        hR = len(pi_intersection_ti) / len(ti)
        hRR.append(hR)

  classP = sum(hPP)/len(hPP)
  classR = sum(hRR)/len(hRR)
  if classR == 0 and classP == 0:
    classF = 0
  else:
    classF = 2*classP*classR/(classP+classR)
  return classP, classR, classF


def hierarchical_prf_weighted(subcat, predictions_gt, lambda_param=0.75):
  # Compute hierarchical weighted precision, recall and f-measure for a given category.
  # Simlar to "hierarchical_prf", but including the lambda_param to control how much we value the correct top-level prediction
  # when second-level is wrong. "hierarchical_prf" is the same as "hierarchical_prf_weighted" when lambda_param=1.0.

  hPP = []
  hRR = []
  for count, (prediction, gt) in enumerate(predictions_gt):
    pi = extend_subcat(prediction)
    ti = extend_subcat(gt)
    pi_intersection_ti = intersection(pi, ti)

    if subcat == prediction:  # precision: include false positives
        w = 1 if prediction == gt else (lambda_param if get_top_level(prediction) == get_top_level(gt) else 0)
        hP = (w * len(pi_intersection_ti)) / len(pi)
        hPP.append(hP)

    if subcat == gt:  # recall: only true positives + false negatives
        w = 1 if prediction == gt else (lambda_param if get_top_level(prediction) == get_top_level(gt) else 0)
        hR = (w * len(pi_intersection_ti)) / len(ti)
        hRR.append(hR)

  classP = sum(hPP)/len(hPP)
  classR = sum(hRR)/len(hRR)
  if classR == 0 and classP == 0:
    classF = 0
  else:
    classF = 2*classP*classR/(classP+classR)
  return classP, classR, classF

def evaluate_model(model_class, model_path, data_loader, device, class_to_topclass, 
               output_dir, fold_id, class_dict=None):
    # -------------------- Setup --------------------
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint["config"]
    model = model_class(**config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    model_name = model.__class__.__name__

    predictions = {"sound_id": [], "gt": [], "pred": [], "pred_score": []}

    # -------------------- Inference --------------------
    with torch.no_grad():
        for data in data_loader:
            labels = data['class_idx'].to(device)
            sound_ids = data['sound_id']

            audio_emb = data.get('audio_embedding', None)
            text_emb = data.get('text_embedding', None)

            if audio_emb is not None:
                audio_emb = audio_emb.to(device)
            if text_emb is not None:
                text_emb = text_emb.to(device)

            _, class_logits, _ = model(audio_emb, text_emb)
            probs = torch.softmax(class_logits, dim=1)

            # Top-1 prediction
            # Note: expand with topk if needed
            top1 = torch.argmax(probs, dim=1)
            max_probs = probs.gather(1, top1.unsqueeze(1)).squeeze(1)

            # Store all predictions and scores
            for i in range(labels.size(0)):
                sid = sound_ids[i]
                if isinstance(sid, torch.Tensor):
                    sid = sid.item()

                predictions["sound_id"].append(sid)
                predictions["gt"].append(labels[i].item())
                predictions["pred"].append(top1[i].item())
                predictions["pred_score"].append(float(max_probs[i]))

    # -------------------- Metrics --------------------
    def compute_metrics(predictions, class_to_topclass, class_dict):
        ''' 
        Compute standard accuracy, top-level accuracy, hierarchical accuracy, and hierarchical weighted precision/recall/f1.
        Standard accuracies are micro, but hierarchical metrics are macro (average per class)
        '''
        total = len(predictions["gt"])

        preds = predictions["pred"]
        gts = predictions["gt"]

        id_to_class = build_id_to_class_mapping(class_dict)
        pred_labels = [id_to_class.get(p, str(p)) for p in preds]
        gt_labels = [id_to_class.get(gt, str(gt)) for gt in gts]

        pred_gt_pairs = list(zip(pred_labels, gt_labels))
        classes = list(set(gt_labels))  # evaluate per class (macro)

        # ---------------- Standard accuracy (micro) ----------------
        correct = sum(p == gt for p, gt in zip(preds, gts))

        top_correct = sum(
            class_to_topclass.get(gt) == class_to_topclass.get(p)
            for p, gt in zip(preds, gts)
            if class_to_topclass.get(gt) is not None
            and class_to_topclass.get(p) is not None
        )

        # ---------------- Standard accuracy (macro) ----------------
        class_accs = []
        class_top_accs = []
        for c in classes:
            class_indices = [i for i, gt in enumerate(gt_labels) if gt == c]
            if not class_indices:
                continue
            class_correct = sum(1 for i in class_indices if preds[i] == gts[i])
            class_accs.append(class_correct / len(class_indices))

            class_top_correct = sum(
                1
                for i in class_indices
                if class_to_topclass.get(gts[i]) == class_to_topclass.get(preds[i])
            )
            class_top_accs.append(class_top_correct / len(class_indices))

        macro_accuracy = np.mean(class_accs) if class_accs else 0
        macro_top_accuracy = np.mean(class_top_accs) if class_top_accs else 0

        # ---------------- Hierarchical accuracy ----------------
        h_accs = []
        for c in classes:
            try:
                acc = hierarchical_accuracy(c, pred_gt_pairs, lambda_param=0.5)
                if not np.isnan(acc):
                    h_accs.append(acc)
            except:
                continue

        hierarchical_acc = np.mean(h_accs) if h_accs else 0

        # ---------------- Hierarchical weighted PRF ----------------
        hPs, hRs, hFs = [], [], []

        for c in classes:
            try:
                p, r, f = hierarchical_prf_weighted(
                    c, pred_gt_pairs, lambda_param=0.75
                )
                if not (np.isnan(p) or np.isnan(r) or np.isnan(f)):
                    hPs.append(p)
                    hRs.append(r)
                    hFs.append(f)
            except:
                continue

        hP = np.mean(hPs) if hPs else 0
        hR = np.mean(hRs) if hRs else 0
        hF = np.mean(hFs) if hFs else 0

        return {
            "accuracy": 100 * correct / total if total > 0 else 0,
            "top_accuracy": 100 * top_correct / total if total > 0 else 0,
            "macro_accuracy": 100 * macro_accuracy if total > 0 else 0,
            "macro_top_accuracy": 100 * macro_top_accuracy if total > 0 else 0,
            "hierarchical_accuracy": 100 * hierarchical_acc if total > 0 else 0,
            "hierarchical_precision": 100 * hP if total > 0 else 0,
            "hierarchical_recall": 100 * hR if total > 0 else 0,
            "hierarchical_f1": 100 * hF if total > 0 else 0,
        }

    metrics = compute_metrics(predictions, class_to_topclass, class_dict)

    # -------------------- Save outputs --------------------
    id_to_class = {v: k for k, v in class_dict.items()} if class_dict else {}

    df = pd.DataFrame({
        "sound_id": predictions["sound_id"],
        "ground_truth": [id_to_class.get(lbl, str(lbl)) for lbl in predictions["gt"]],
        "prediction": [id_to_class.get(lbl, str(lbl)) for lbl in predictions["pred"]],
        "prediction_score": [round(float(x), 4) for x in predictions["pred_score"]],
    })

    pred_path = os.path.join(output_dir, "evaluation", "predictions.csv")
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    df.to_csv(pred_path, index=False)

    # Misclassified
    # mis_df = df[df["ground_truth"] != df["prediction"]]
    # mis_path = os.path.join(output_dir, "evaluation", "misclassified_samples.csv")
    # mis_df.to_csv(mis_path, index=False)

    metrics_to_log = {
        "Accuracy": "accuracy",
        "Top class accuracy": "top_accuracy",
        "Macro accuracy": "macro_accuracy",
        "Macro top class accuracy": "macro_top_accuracy",
        "Hierarchical accuracy": "hierarchical_accuracy",
        "Hierarchical precision": "hierarchical_precision",
        "Hierarchical recall": "hierarchical_recall",
        "Hierarchical F1": "hierarchical_f1",
    }

    for label, key in metrics_to_log.items():
        print(f"[{model_name} | Fold {fold_id}] {label}: {metrics[key]:.2f}%")

    with open(os.path.join(output_dir, "evaluation", "results.txt"), "w") as f:
        for label, key in metrics_to_log.items():
            f.write(f"{key}: {metrics[key]:.2f}%\n")

    return metrics


if __name__ == "__main__":
    import json
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    from torch.utils.data import DataLoader
    from dataset_utils import HATRDataset
    from models import BaseClassifier
    from utils import get_subconfig

    # ---- define evaluation setup ----
    fold_id = 0  # hardcoded for now, can loop if needed
    model_folder = f"./model_output/both/fold_{fold_id}"  # Model path to evaluate. Change if needed
    evaluation_ouput_folder = os.path.join(model_folder, "evaluation_only")  # Save evaluation outputs separately to avoid overwriting training outputs
    os.makedirs(evaluation_ouput_folder, exist_ok=True)
    # ---------------------------------

    dataset_name = get_subconfig("active_dataset")
    dataset_path = get_subconfig("datasets")[dataset_name]["metadata_csv"]  # Change to evaluate on a different dataset than the one for training
    color_dict_path = get_subconfig("color_dict_path")
    top_color_dict_path = get_subconfig("top_color_dict_path")
    data_dir = get_subconfig("output_path")
    prepared_dataset_path = os.path.join(data_dir, get_subconfig("processed_dataset_csv"))  # it should be the one used for training for recreating the evaluation
    class_dict_json = os.path.join(data_dir, get_subconfig("class_dict_json"))
    top_class_dict_json = os.path.join(data_dir, get_subconfig("top_class_dict_json"))
    subclass_json = os.path.join(data_dir, get_subconfig("top_class_subclass_dict_json"))

    with open(class_dict_json, 'r') as f:
        class_dict = json.load(f)
    with open(top_class_dict_json, 'r') as f:
        top_class_dict = json.load(f)

    prepared_df = pd.read_csv(prepared_dataset_path)
    confidence_df = pd.read_csv(dataset_path)
    filtered_conf = confidence_df[confidence_df['confidence'] >= 0]
    prepared_df = prepared_df[prepared_df['index'].isin(filtered_conf['sound_id'])]
    database = prepared_df.reset_index(drop=True)

    model_path = os.path.join(model_folder, "best_model.pth")
    history_path = os.path.join(model_folder, "history.json")  # Retrieve saved history.json 
    with open(history_path, 'r') as f:
        history = json.load(f)
    seed = history['model_info']['random_seed']
    k_folds = history['model_info']['num_folds']

    # Recreate the fold split as in train_test.py
    labels = database["class_idx"].tolist()
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    
    for fold, (trainval_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        if fold == fold_id:
            test_df = database.iloc[test_idx].reset_index(drop=True)
            break

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Change evaluated dataset here, if necessary
    test_dataset = HATRDataset(test_df, aug=False) 
    data_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())
    
    evaluate_model(
        model_class=BaseClassifier,
        model_path=model_path,
        data_loader=data_loader,
        device=device,
        class_to_topclass=build_class_to_topclass_mapping(class_dict, top_class_dict),
        output_dir=evaluation_ouput_folder,
        fold_id=fold_id,
        class_dict=class_dict
    )
