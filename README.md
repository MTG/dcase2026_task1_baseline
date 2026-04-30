# DCASE2026 Challenge: Task 1 - Baseline System

Contact: **Panagiota Anastasopoulou** (panagiota.anastasopoulou@upf.edu), *Music Technology Group, Universitat Pompeu Fabra, Barcelona*

## Heterogeneous Audio Classi1cation

This task focuses on heterogeneous audio classification using the Broad Sound Taxonomy (BST), which comprises 5 top-level and 23 second-level sound categories. The goal of this task is to evaluate sound classification models on diverse, real-world audio that varies widely in its nature, including duration and recording conditions. To that end, two complementary Freesound-based datasets are provided: a curated set, BSD10k-v1.2, and a larger, noisier, crowd-sourced collection, BSD35k-CS, which reflects real-world labeling variability. Participants are encouraged to explore audio-based and multimodal approaches to sound classification, as well as to leverage hierarchical relationships between taxonomy categories.

For a detailed description of the challenge and this task visit the [DCASE website](https://dcase.community/challenge2026/task-heterogeneous-audio-classification).

## Baseline System

This repository contains the code for the baseline systems of the DCASE 2026 Challenge Task 1. 
It provides a full pipeline for training and evaluating an audio classification model using precomputed audio and text embeddings.

As a baseline system, we use variations of the [HATR model](https://github.com/allholy/hatr) presented at the DCASE Workshop 2025 [1]. 
We include a **multimodal** and an **audio-only** version, both non-hierachical models trained on audio (and text) representation vectors extracted using the `630k-audioset-fusion-best.pt` checkpoint of the pretrained [LAION-CLAP](https://github.com/LAION-AI/CLAP/) model.

The model is characterized by:
- **Multimodality:** Supports both audio and text input (embeddings) with separate encoders
- **Attention-based fusion:** Learns to weight modalities dynamically
- **Residual-based classifier:** Stacked residual blocks
- **Data augmentation:** Gaussian noise and random masking

For the evaluation phase, apart from standard accuracy (micro and macro in both levels), we additionally compute hierarchical metrics (accuracy, precision, recall, F1) as part of the challenge's rules and ranking system.


## Quick Start

1. Clone this repository.

2. Create and activate a [conda](https://docs.anaconda.com/free/miniconda/index.html) environment:

```
conda create -n hac python=3.13
conda activate hac
```

3. Install requirements:

```
pip3 install -r requirements.txt
```

You can edit the [PyTorch](https://pytorch.org/get-started/previous-versions/) version if necessary to suit your system.

4. Download and extract the datasets: [BSD10k-v1.2](https://zenodo.org/records/17233904) and [BSD35k-CS](https://zenodo.org/records/19187099). Their file structure is described in their READMEs.

7. Specify the input and output paths in `config.yaml`. Make sure all paths point to the correct directories or files before running the model. By default, all generated files for internal model use are stored in the `data/` directory. We also assume that datasets are placed or symlinked into this directory.

8. Run the data preparation script:

```
python build_dataset.py
```

7. Train the model:

```
python train_test.py
```

This script includes k-fold training and evaluation of the models on their respective test sets.

You can run only evaluation with `evaluate.py`. 

8. Summarize the results:

```
python summarize_results.py
```

This script summarizes results for each model across all 5 folds.

> **Note:** You can skip steps 6-8 and simply run:
>     python main.py


## Baseline Results

| Dataset    | Mode        | Hier. Accuracy       | Hier. F1            |
|------------|-------------|----------------------|---------------------|
| BSD10k     | Audio       | 77.36% ± 0.71%       | 76.11% ± 0.45%      |
| BSD10k     | Multimodal  | 79.71% ± 0.82%       | **78.76% ± 0.79%**  |
| BSD35k-CS  | Audio       | 70.90% ± 0.45%       | 70.19% ± 0.93%      |
| BSD35k-CS  | Multimodal  | 80.63% ± 1.02%       | 79.77% ± 0.60%      |

We report hierarchical accuracy and hierarchical F1 (both macro-averaged), where the later is used for ranking systems. For additional metrics, such as hierarchical precision, recall, standard micro and macro accuracy and top-level accuracy, you can run the baseline on the datasets yourself.

> Note: These results may reflect internal biases or inaccuracies within each dataset, and training in combination or cross-evaluation may significantly change the results. BSD10k is annotated by experts and therefore expected to be more representative. BSD35k-CS is crowdsourced and may exhibit stronger biases, for example due to uneven contribution patterns from frequent contributors or noise introduced by inconsistencies in user annotations. Thus, higher scores on the evaluation dataset may require careful dataset selection, post-processing, or validation before adoption.

## Citations
[1] Panagiota Anastasopoulou, Jessica Torrey, Xavier Serra, and Frederic Font. Heterogeneous sound classification with the Broad Sound Taxonomy and Dataset. In Proc. Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE). 2024.
