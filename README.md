# VesselReID

Maritime vessel re-identification: given a query image of a vessel, retrieve images of
the same vessel from a gallery of unseen images (different pose, lighting, weather,
time). Built on the VesselReID dataset with a ResNet-50 backbone, trained in three
stages of increasing sophistication (baseline -> global+part -> strong-baseline v2),
evaluated with a from-scratch CMC/mAP implementation.

## Results

Evaluated on the held-out test split (495 identities, 1 query + rest as gallery per ID).

| Model | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---|---|---|---|
| Baseline (ResNet-50, CE + triplet) | 0.509 | 0.742 | 0.860 | 0.895 |
| Global + Part (3 stripes) | 0.637 | 0.866 | 0.927 | 0.939 |
| Global + Part + re-ranking + flip-TTA | 0.740 | 0.868 | 0.919 | 0.933 |
| **Global + Part v2** (GeM, BNNeck, PK-sampler, label smoothing, random erasing, warmup+cosine, AMP) | 0.769 | 0.917 | 0.957 | 0.966 |
| **Global + Part v2 + re-ranking + flip-TTA** | 0.819 | 0.915 | 0.945 | 0.964 |
| **Global + Part v2, 384px input** (same recipe, `--img-size 384`, 30 epochs) | 0.796 | 0.935 | 0.967 | 0.974 |
| **Global + Part v2, 384px + re-ranking + flip-TTA** | **0.836** | 0.921 | 0.953 | 0.962 |

mAP roughly **+0.33** and Rank-1 **+0.18** over the original reported baseline. The 384px
input resolution is the single biggest lever after the v2 recipe itself - it adds another
+0.017 mAP and +0.006 Rank-1 over the 256px v2 model (both +rerank+flip-TTA), on top of the
+0.31/+0.17 gained from fixing the eval protocol and the strong-baseline training recipe -
no architecture change beyond GeM/BNNeck on the same global+part topology. The 384px model
is now the default for `--model global_part_v2_384` in `eval.py`.

See `figures/retrieval_global_part_v2_384_rerank_flip.png` for example query -> top-5
retrievals from the best/final model (green border = correct vessel, red = incorrect).
`figures/retrieval_global_part_v2_rerank_flip.png` and `figures/retrieval_global_part_rerank_flip.png`
use the same six queries with the earlier 256px v2 and original global+part models
respectively, and `figures/retrieval_baseline.png` with the baseline, for a direct
before/after comparison across all four generations.

Note on evaluation protocol: this dataset's `camera_id`/`seq_id` fields (parsed from the
`<vessel_id>_c1s1_<frame>_01.jpg` filename convention) are constant across every image -
there's no real multi-camera signal to hold out on, so matching is done by vessel ID only
across the whole gallery (see `eval.py` docstring for details).

## Architecture

- **Backbone**: ResNet-50 (ImageNet pretrained), stride-32 feature map, FC/avgpool removed.
- **Baseline** (`models/baseline_resnet.py`): global average pool -> 512-d embedding ->
  L2-normalize -> linear classifier. Trained with CE + batch-hard triplet loss.
- **Global + Part** (`models/global_part_resnet.py: GlobalPartReIDResNet`): adds 3
  horizontal-stripe part branches (à la PCB), each with its own reduction + classifier;
  inference feature is the L2-normalized concat of global + all part embeddings (2048-d).
- **Global + Part v2** (`models/global_part_resnet.py: GlobalPartReIDResNetV2`): same
  global+part topology, upgraded with the "Bag of Tricks" strong-baseline recipe
  (Luo et al., 2019):
  - **GeM pooling** instead of plain average pooling (`models/layers.py`)
  - **BNNeck**: triplet loss sees the pre-BN embedding, the classifier and retrieval
    both use the post-BN embedding - decouples the two competing objectives
  - **PK-sampler** (`samplers.py`): P identities x K images per batch, so batch-hard
    triplet loss actually has positives to mine (plain shuffling over ~1500 IDs almost
    never puts 2 images of the same vessel in one batch)
  - Label smoothing, random erasing, warmup+cosine LR schedule, mixed precision (AMP)

At eval time, `eval.py` optionally applies **k-reciprocal re-ranking** and **flip
test-time augmentation** - both free (no retraining) and each add meaningful mAP.

## Repo layout

```
datasets.py               # Vessel2258Dataset (train), Vessel2258EvalDataset (query/gallery)
samplers.py                # PK identity sampler for triplet-loss training
models/
  baseline_resnet.py       # GlobalReIDResNet
  global_part_resnet.py    # GlobalPartReIDResNet, GlobalPartReIDResNetV2
  layers.py                 # GeM pooling, BNNeck
losses/reid_loss.py        # CE (label smoothing) + batch-hard triplet
train_baseline.py          # original baseline training run
train_global_part.py       # original global+part training run
train_v2.py                # strong-baseline v2 training run
eval.py                    # unified eval: CMC/mAP, --rerank, --flip-tta
visualize_retrieval.py     # saves query/top-K retrieval figure to figures/
app.py                     # interactive Streamlit retrieval demo
make_metadata.py           # builds vessel2258_meta_split.csv from image folders
demo.py                    # downloads the official VesselReID.json dataset (see below)
run_train_v2_resilient.ps1 # Windows watchdog: auto-restarts train_v2.py --auto-resume after a crash
```

## Reproducing

```bash
pip install -r requirements.txt

# Train
python train_v2.py --epochs 30 --warmup-epochs 5 --batch-size 64 --num-instances 4

# Evaluate (add --rerank / --flip-tta for the free accuracy boost)
python eval.py --model global_part_v2 --rerank --flip-tta

# Visualize retrievals
python visualize_retrieval.py --model global_part_v2 --rerank --flip-tta

# Interactive demo (pick a query vessel, see live top-K retrieval)
streamlit run app.py
```

## About the dataset

This repo's working dataset (`REID_2258/`) is a locally-curated vessel image set of
2,469 identities / 124,932 images (60/20/20 train/val/test split by identity, built by
`make_metadata.py`) - larger than, and independent from, the official VesselReID
release described below.

### VesselReID Dataset Introduction
To facilitate the research of vessel re-identification, we build the VesselReID dataset that consists of 30,587 images of 1,248 vessels. Each vessel in VesselReID has a number of images captured at various times, places, and viewpoints as well as under different weather conditions.

Following the normal settings in most popular re-ID datasets, we split the entire dataset into training and testing sets. We randomly select 624 vessel identities (half of the vessel identities) for training, while images of the other 624 vessels are used for testing. Therefore, there is no validation set in the original VesselReID dataset, and the re-ID models are supposed to be evaluated on the test set during the training. Moreover, we additionally divide the VesselReID dataset into training, validating and testing sets as an extra split, where there are 550 vessel identities in both training and testing sets while the remaining 148 vessel identities are utilised for validation.

See more details in the original paper (cited below).

### Dataset Availability
To encourage the research in terms of vessel re-identification, the dataset authors are pleased to provide the VesselReID dataset according to request.
If you want to use the dataset for research, contact *info@vsislab.com*. They will make the dataset available by providing the file needed for a download.
When contacting them, state full name and affiliation.
This information is requested only to make sure the dataset is used for non-commercial purposes.

### Download

```python
conda create -n demo python=3.6
conda activate demo
pip install -r requirements.txt
python demo.py --pth '$path$/VesselReID.json' --save_pth '$path$/datasets/VesselReID'    # pth is the address of the annotations file and save_pth is the address where the data set is downloaded and saved
```
The needed file titled 'VesselReID.json' can only be obtained by email.

### Citation
If you find the VesselReID Dataset useful in your work, please cite the original paper:

```
@ARTICLE{10046401,
  author={Zhang, Qian and Zhang, Mingxin and Liu, Jinghe and He, Xuanyu and Song, Ran and Zhang, Wei},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  title={Unsupervised Maritime Vessel Re-Identification With Multi-Level Contrastive Learning},
  year={2023},
  volume={24},
  number={5},
  pages={5406-5418},
  doi={10.1109/TITS.2023.3243591}}
```
