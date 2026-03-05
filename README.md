# OpenSplat3D

[[`Paper`](https://arxiv.org/abs/2506.07697)] &nbsp; | &nbsp; [[`Project Page`](https://jenspiek.github.io/opensplat3d)] &nbsp; | &nbsp; [[`BibTeX`](#-Citation)]

**CVPRW 2025 (OpenSUN3D)**

Official implementation of the paper "OpenSplat3D: Open-Vocabulary 3D Instance Segmentation using Gaussian Splatting".

## Setup

```bash
git clone https://github.com/VisualComputingInstitute/opensplat3d.git --recursive
```

If using _just_, run the following:

```bash
just setup
```

otherwise run these steps manually

```bash
# # load CUDA which is required to build the renderer
# module load cuda/12.4

# # if required, define your compute capabilities
# export TORCH_CUDA_ARCH_LIST="8.6 8.9"

uv sync
uv sync --extra compile # compile packages that require no-build-isolation
```

### Checkpoints

To create the checkpoints folder and download the required checkpoints, _just_ run the following unless `just setup` is already being used:

```bash
just download_ckpts
```

If _just_ is not desired, manually execute the corresponding commands from the [justfile](justfile).

## Data Structure and Preprocessing

### Scene Directory Structures

- COLMAP-style (3DGS default):
  - Expects a `./sparse/0` subfolder with the model.
- Nerfstudio-style (ScanNet++):
  - Expects a `transform.json` file.
- Blender:
  - Expects a `transforms_train.json` file.

### Custom Dataset

The easiest way to use custom data is to create a scene folder `{source_path}` and put the image collection into a `./input` subfolder.

```
{source_path}
├── input
│   ├── {image 0}
│   ├── {image 1}
│   ├── ...
```

### Preprocessing

If using _just_, run the following to apply the default SfM and mask extraction procedure:

```bash
just source_path="$source_path" preprocess
```

#### Structure-from-Motion

To reconstruct the cameras and sparse structure based on the images in the `{source_path}/input` directory, using either colmap or glomap, run the following command:

```bash
uv run python opensplat3d/data/preprocessing/sfm.py "$source_path"
```

Our script uses the best model in `{source_path}/distorted/sparse/` based on the number of reconstructed cameras and points.

#### Mask Extraction

To extract the SAM masks run:

```bash
uv run python opensplat3d/masks/extract_sam_masks.py "$source_path" --sort score
```

this creates a subfolder `./sam` in the source directory with a `.npy/.npz` file per frame.

## Pipeline

### Optimize Gaussians with Instance Embeddings

For joint optimization of Gaussians with instance embeddings run:

```bash
uv run python opensplat3d/train.py model.source_path="$source_path" model.model_path="$model_path" --config configs/XXX.yaml
```

You can write your own config or use one from [configs](./configs).
More configuration parameters can be found in [params.py](opensplat3d/params.py).
The config parameters can be overritten by CLI arguments.

> The clustering and language embeddings can be computed directly after optimizing the Gaussians by enabling it in the config or by CLI.

### Clustering

To manually cluster the features of the Gaussians use the following command:

```bash
uv run python opensplat3d/cluster/hdbscan.py "$model_path" --min-size 32 --min-samples 16
```

### Language Embeddings

To manually compute the language embeddings use the following command:

```bash
uv run python opensplat3d/language/embed.py "$model_path" --lang-model masqclip --dynamic-ratio
```

### Output Directory Structure

> The `model_id` is automatically generated to avoid clashing run, avoid manual handling of existing runs and ensure easy identification. It has the format `{yyyymmddHHMMSS-uid}`.

#### Single Scene

```
{model_path}/{model_id}
├── clustering/
│   ├── config.yaml              # Clustering configuration
│   ├── labels.npy               # Clustering labels
│   └── stats.json               # Clustering statistics (e.g., noise ratio)
├── point_cloud/
│   └── iteration_{iterations}/
│       └── point_cloud.ply      # Gaussian point cloud at iteration {iterations}
├── cameras.json                 # Camera parameters
├── command.txt                  # Command used to launch the run
├── config.yaml                  # Model configuration
├── input.ply                    # Input point cloud (possibly downsampled)
├── {lang_model}_embeddings.pth  # Language model embeddings (e.g., CLIP, MasQCLIP)
├── overrides.yaml               # CLI override parameters
└── wandb/                       # Weights & Biases logging directory (if enabled)
```

#### Evaluation on Multiple Scenes

For evaluating multiple scenes it is important to have a `scenes` subfolder in the output directory.
This is accomplished by using `model.model_path="{output_path}/scenes/{scenes_id}"`, so that the output will be organized as:

```
{output_path}/
├── scenes/
│   └── {scene_id}/{model_id}/    # Per-scene model directory (see above)
├── eval_predictions/             # Exported predictions (e.g., ScanNet++)
└── eval_results/                 # Aggregated evaluation results
```

## Interactive Demo

```bash
uv run python demo.py "$model_path" --cameras --language "$lang_model"
```

where the language model can be `clip`, `siglip` or `masqclip`, depending on the computed embeddings.

## Evaluation

In the following, the provided `$exp_path` should point to a single or eval experiment directory.
If the directory is an eval directory, the script will automatically evaluate all of the scenes in the `scenes` directory.

### LERF-Mask

```bash
uv run python opensplat3d/eval/eval_lerf_mask.py "$exp_path"
```

Use `--help` or see the [eval_lerf_mask.py](./opensplat3d/eval/eval_lerf_mask.py) for more configurations.

### LERF-OVS

Set the `LERF_OVS_LABEL_PATH` environment variable to point to the label directory of LeRF-OVS.

```bash
uv run python opensplat3d/eval/eval_lerf_ovs.py "$exp_path"
```

Use `--help` or see the [eval_lerf_ovs.py](./opensplat3d/eval/eval_lerf_ovs.py) for more configurations.

### ScanNet++

To prepare the ScanNet++ dataset, you need to preprocess it first with the official documentation, i.e. undistort the DSLR images and prepare the semantic/instance ground-truth pth files for evaluation.
To convert these files into the dataformat for our Gaussian Splatting pipeline, use the following command:

```bash
uv run python opensplat3d/data/preprocessing/scannetpp.py "$dataset_dir" data/scannetpp "$split" --pth-dir "$pth_dir"
```

where `$dataset_dir` points to the root directory of the ScanNet++ dataset containing `data` and `splits` subdirectories.

When you want to use the segments based on the Felzenswalb and Huttenlocher's algorithm, use the official [segmentator](https://github.com/ScanNet/ScanNet/tree/master/Segmentator) of ScanNet.
The segments.json files provided by ScanNet++ do not have the correct segments.

Set the `SCANNETPP_ROOT_PATH` environment variable to the root directory of the ScanNet++ dataset containing the actual data.
Set the `SCANNETPP_PATH` environment variable to the processed ScanNet++ dataset directory containing the `pth/val`, `sem_gt/val`, and `inst_gt/val` subdirectories, as well as the optional `segments` directory.

The prediction format that is required for the evaluation are automatically exported after training if configured, but can be also manually exported using the following script:

```bash
uv run python opensplat3d/eval/scannetpp/export_scannetpp.py "$exp_path"
```

> **NOTE**: We implemented custom wrapper methods to evaluate our method, including class-agnostic evaluation. The necessary ScanNet++ source files were vendored and packaged for easier use, see [here](submodules/scannetpp/README.md) for more details.

#### Instance Evaluation

To run the standard instance evaluation use:

```bash
uv run python opensplat3d/eval/scannetpp/eval_instance.py "$exp_path"
```

For a class-agnostic evaluation run:

```bash
uv run python opensplat3d/eval/scannetpp/eval_instance.py "$exp_path" --class-agnostic
```

#### Semantic Evaluation

```bash
uv run python opensplat3d/eval/scannetpp/eval_semantic.py "$exp_path"
```

## ℹ️ Disclaimer

This software is a research prototype only and suitable only for test purposes.
It has been published solely for use in research applications; it is not permitted to use this software in any kind of improper, disrespectful, defamatory, obscene, military or otherwise harmful application.
This software is not suitable for use in or for products and/or services and in particular not in or for safety-relevant areas.
It was solely developed for and published as part of the publication "OpenSplat3D: Open-Vocabulary 3D Instance Segmentation using Gaussian Splatting" and will neither be maintained nor monitored in any way.

The research and development of this software by RWTH Aachen has been supported by Robert Bosch GmbH under the project "Context Understanding for Autonomous Systems".

## ⚖️ License

The code is released under the Gaussian-Splatting License.
See [LICENSE](LICENSE) for more details.

## 🎓 Citation

f you use our work in your research, please use the following BibTeX entry.

```
@InProceedings{piekenbrinck2025opensplat3d,
  title     = {{OpenSplat3D: Open-Vocabulary 3D Instance Segmentation using Gaussian Splatting}},
  author    = {Piekenbrinck, Jens and Schmidt, Christian and Hermans, Alexander and Vaskevicius, Narunas and Linder, Timm and Leibe, Bastian},
  booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages     = {5246--5255},
  year      = {2025}
}
```
