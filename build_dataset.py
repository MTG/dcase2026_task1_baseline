import os
import pandas as pd
import json
from utils import get_subconfig

"""
The dataset is built from existing embedding (.npy) files identified by sound_id.
A sample is included only if both audio and text embeddings exist and a matching metadata 
entry is found. The metadata is used to assign class and top-class labels, which are also saved as JSON files.
By default, we exclude top-level classes and classes that belong to the "-other" category.
"""

# --- Filepaths ---
dataset_name = get_subconfig("active_dataset")
metadata_csv = get_subconfig("datasets")[dataset_name]["metadata_csv"]
audio_emb_folder = get_subconfig("datasets")[dataset_name]["audio_emb_folder"]
text_emb_folder = get_subconfig("datasets")[dataset_name]["text_emb_folder"]

output_path = get_subconfig("output_path")
os.makedirs(output_path, exist_ok=True)
processed_dataset_csv = os.path.join(output_path, get_subconfig("processed_dataset_csv"))
class_dict_json = os.path.join(output_path, get_subconfig("class_dict_json"))
top_class_dict_json = os.path.join(output_path, get_subconfig("top_class_dict_json"))
top_class_subclass_dict_json = os.path.join(output_path, get_subconfig("top_class_subclass_dict_json"))

# --- Load metadata and build class mappings ---
df = pd.read_csv(metadata_csv)
df['sound_id'] = df['sound_id'].astype(str).str.strip()

print(f"Examining original data from {dataset_name}:")
print(f"  Total rows: {len(df)}")
print(f"  Unique classes: {df['class'].nunique()}")

# Discard top-level classes and classes that belong to "-other" category
s = df['class_idx'].astype(str)
df = df[~((s.str.len() == 3) & (s.str.endswith('99') | s.str.endswith('00')))].copy()
print("After filtering:", len(df))

df['original_class_idx'] = df['class_idx']

# --- Map class_idx → 0..N for training ---
original_indices = sorted(df['original_class_idx'].unique())
index_mapping = {orig: new for new, orig in enumerate(original_indices)}
df['class_idx'] = df['original_class_idx'].map(index_mapping)

# --- top class ---
df['class_top'] = df['class'].apply(lambda x: x.split('-')[0] if isinstance(x, str) else None)

df_sorted = df.sort_values('original_class_idx')

top_classes = df_sorted['class_top'].drop_duplicates()
class_top_dict = {cls: i for i, cls in enumerate(top_classes)}

df['top_class_idx'] = df['class_top'].map(class_top_dict)

# --- class dict ---
class_dict = dict(zip(df['class'], df['class_idx']))

# --- subclass dict ---
class_top_subclass_dict = {
    top_class: {
        subclass: idx
        for idx, subclass in enumerate(
            df[df['class_top'] == top_class]
            .sort_values('original_class_idx') 
            ['class']
            .drop_duplicates()
        )
    }
    for top_class in class_top_dict.keys()
}

with open(class_dict_json, 'w') as f:
    json.dump(class_dict, f, indent=4)
print(f"Saved class dictionary to {class_dict_json}")

with open(top_class_dict_json, 'w') as f:
    json.dump(class_top_dict, f, indent=4)
print(f"Saved top class dictionary to {top_class_dict_json}")

with open(top_class_subclass_dict_json, 'w') as f:
    json.dump(class_top_subclass_dict, f, indent=4)
print(f"Saved top class subclass dictionary to {top_class_subclass_dict_json}")

records = []

df.set_index('sound_id', inplace=True)
for sound_id in df.index:
    file = f"{sound_id}.npy"

    audio_emb_filepath = os.path.abspath(os.path.join(audio_emb_folder, file))
    text_emb_filepath = os.path.abspath(os.path.join(text_emb_folder, file))

    if not os.path.isfile(audio_emb_filepath):
        print(f"Missing audio embedding for sound_id {sound_id}")
        continue

    if not os.path.isfile(text_emb_filepath):
        print(f"Missing text embedding for sound_id {sound_id}")
        continue

    match = df.loc[sound_id]

    class_top = match['class_top']
    class_top_idx = class_top_dict.get(class_top, -1)
    class_name = match['class']
    class_idx = int(match['class_idx'])

    records.append({
        "index": sound_id,
        "audio_emb_filepath": audio_emb_filepath,
        "text_emb_filepath": text_emb_filepath,
        "top_class": class_top,
        "top_class_idx": class_top_idx,
        "class": class_name,
        "class_idx": class_idx,
    })

db_df = pd.DataFrame(records)
db_df.to_csv(processed_dataset_csv, index=False)
print(f"Saved embedding dataframe to {processed_dataset_csv}")
print(f"Dataset built with {len(db_df)} samples.")