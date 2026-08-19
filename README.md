<a id="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">SHARP：Hierarchical Ambiguity Resolution with Small Language Models In Domestic HRI</h3>

  ![Framework Overview](docs/fig2_from_svg.pdf)

</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#downloading-datasets-and-models">Downloading Datasets and Models</a>
    </li>
    <li>
      <a href="#starting-services">Starting Services</a>
    </li>
    <li>
      <a href="#evaluation">Evaluation</a>
    </li>
    <li>
      <a href="#other-baseline-methods-evaluation">Other Baseline Methods Evaluation</a>
    </li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

The **SHARP** framework is a hierarchical ambiguity resolution pipeline designed specifically for resource-constrained, on-device processing in domestic human-robot interaction (HRI). It integrate[...]

The SLM initially infers user intent at the category level and selectively invokes the VLM to ground visual attributes. The system then performs symbolic reasoning over a pruned scene graph to res[...]

**Key Dependencies**
This project builds upon the amazing works of the following open-source communities:
* [![GroundingDINO](https://img.shields.io/badge/GroundingDINO-IDEA_Research-blue)](https://github.com/IDEA-Research/GroundingDINO): Used for open-vocabulary object detection.
* [![SAM](https://img.shields.io/badge/Segment_Anything-Meta_FAIR-purple)](https://github.com/facebookresearch/segment-anything): Used for precise instance segmentation.


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

*   **Conda**: Ensure you have Anaconda or Miniconda installed.
*   **Python**: Version 3.10 is required.

### Installation

1.  **Configure Environment**
    ```sh
    conda create -n SHARP python=3.10
    conda activate SHARP
    ```

2.  **Install PyTorch**
    ```sh
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

3.  **Install GroundingDINO**
    ```sh
    cd GroundingDINO
    pip install -e . --no-build-isolation
    cd ..
    ```

4.  **Install SAM**
    ```sh
    pip install -e sam/segmentation/SAM/
    ```

5.  **Install Requirements**
    ```sh
    pip install -r requirements.txt
    ```

6.  **Install flash_attn**
    *Ensure you install the `flash_attn` package as required.*

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- DOWNLOADING DATASETS AND MODELS -->
## Downloading Datasets and Models

All the following commands should be executed from the root directory of the project (where this README is located).

### 1. Download Datasets (ModelScope)
We provide two benchmarks for evaluation:
-   **Ambiguity-200**: Real-world environment dataset.
-   **Ambiguity-1k**: Simulation environment dataset.

```bash
# Download the real-world dataset
modelscope download --dataset fdy13672387986/Ambiguity-200 --local_dir data/Ambiguity-200

# Download the simulation dataset
modelscope download --dataset fdy13672387986/Ambiguity-1k --local_dir data/Ambiguity-1k
```

### 2. Download SHARP Weights (ModelScope)
Download the Qwen-based SLM weights for the standard SHARP evaluation.

```bash
# For standard SHARP evaluation (default)
modelscope download --model fdy13672387986/full --local_dir weights/full
```

### 3. Download SmolVLM Weights (ModelScope)
We provide two fine-tuned variants of the SmolVLM model. Download them according to the dataset you plan to evaluate:
*   **smolvlm-real**: Fine-tuned for real-world scenarios.
*   **smolvlm-sim**: Fine-tuned for simulation environments.

Please download the models and place them in the following paths:
```text
weights/smolvlm-real
weights/smolvlm-sim
```
> **Note:** By default, the service reads from `weights/smolvlm-real`. You can specify a different path when starting the service if evaluating the simulation dataset.

### 4. Download GroundingDINO Weights (Official)
Download `groundingdino_swinb_cogcoor.pth` and place it in:
```text
GroundingDINO/weights/groundingdino_swinb_cogcoor.pth
```

### 5. Download SAM Weights (Official)
Download `sam_vit_h_4b8939.pth` and place it in:
```text
sam/checkpoints/sam_vit_h_4b8939.pth
```

### Final Directory Structure Check (Key Files)
After preparation, ensure at least the following paths exist:
```text
data/Ambiguity-200/
data/Ambiguity-1k/          # If downloaded
weights/full/
weights/smolvlm-real/       # Real-world weights
weights/smolvlm-sim/        # Simulation weights
GroundingDINO/weights/groundingdino_swinb_cogcoor.pth
sam/checkpoints/sam_vit_h_4b8939.pth
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- STARTING SERVICES -->
## Starting Services

To run the SHARP evaluation pipeline, several background services must be initialized. These include the **GroundingDINO** background service for object detection and a **Dual-Model** service hos[...]

### 1. Start GroundingDINO Service
```bash
cd ambiguity
python utils/dino_service.py
```

### 2. Start Dual Models (Default Configuration - Real World)
If your model weights are in the default paths (`weights/full` and `weights/smolvlm-real`), run the following to start evaluation services configured for real-world contexts:
```bash
python start_dual_model_services.py
```

### 3. Start Dual Models for Simulation or Custom Paths
If you are evaluating the simulation dataset (`Ambiguity-1k`), you need to specify the simulation weights for the VLM (`smolvlm-sim`). 

If your models are located elsewhere, specify them via command-line arguments:
```bash
python start_dual_model_services.py \
    --qwen-model-path weights/full \
    --smol-model-path weights/smolvlm-sim
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- EVALUATION -->
## Evaluation

Use the `run_evaluate.py` script as the unified entry point for evaluations.

### Basic Usage
The primary entry point for executing the evaluation framework is using `run_evaluate.py`. You can evaluate either the real-world dataset (`Ambiguity-200`) or the simulation dataset (`Ambiguity-1[...]

```bash
python run_evaluate.py <benchmark_folder> --eval <eval_type> [other_arguments]
```
*   `<benchmark_folder>`: Target dataset directory (e.g., `Ambiguity-200` or `Ambiguity-1k`).
*   `<eval_type>`: The specific pipeline configuration. Use `sharp` for SHARP evaluation.

### Examples

You can replace `Ambiguity-200` with `Ambiguity-1k` to evaluate on the simulation dataset.

1.  **SHARP Evaluation**
    ```bash
    python run_evaluate.py Ambiguity-200 --eval sharp
    ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- OTHER BASELINE METHODS EVALUATION -->
## Baseline Evaluation

We also provide scripts to evaluate popular baseline methods. You can specify either `Ambiguity-200` for real-world scenarios or `Ambiguity-1k` for simulation.

### FreeGrasp Evaluation
To evaluate using FreeGrasp:
```bash
python FreeGrasp_code/evaluate_freegrasp.py <benchmark_folder>
```
*Example (Real-world):*
```bash
python FreeGrasp_code/evaluate_freegrasp.py Ambiguity-200
```

### AmbResVLM Evaluation
To evaluate using AmbResVLM:
```bash
python AmbResVLM/scripts/evaluate_ambres_benchmark.py <benchmark_folder>
```
*Example (Simulation):*
```bash
python AmbResVLM/scripts/evaluate_ambres_benchmark.py Ambiguity-1k
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>
