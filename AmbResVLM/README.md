Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and proprietary rights in and to this software, related documentation and any modifications thereto. Any use, reproduction, disclosure or distribution of this software and related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.

# Robotic Task Ambiguity Resolution via Natural Language Interaction
[**arXiv**](https://arxiv.org/abs/2504.17748) | [**website**](https://ambres.cs.uni-freiburg.de/) 

Repository providing the source code for the paper 
>Robotic Task Ambiguity Resolution via Natural Language Interaction<br/>
>[Eugenio Chisari](https://rl.uni-freiburg.de/people/chisari), [Jan Ole von Hartz](https://rl.uni-freiburg.de/people/hartz), [Fabien Despinoy](https://www.toyota-europe.com/about-us/toyota-in-europe/toyota-motor-europe), [Abhinav Valada](https://rl.uni-freiburg.de/people/valada)

Please cite the paper as follows:
```text
@article{chisari2025robotic,
  title={Robotic Task Ambiguity Resolution via Natural Language Interaction},
  author={Chisari, Eugenio and von Hartz, Jan Ole and Despinoy, Fabien and Valada, Abhinav},
  journal = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2025}
}
```

## Installation

```bash
conda create --name ambres_env python=3.10
conda activate ambres_env
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

#### SAM2

```bash
bash bash/download_weights.sh
pip install git+https://github.com/facebookresearch/sam2.git
```

## Dataset download

Download the dataset from [this link](https://ambres.cs.uni-freiburg.de/download/ambres.zip). Then extract it under the folder `~/datasets/`.

## Pre-trained checkpoints

To download the pre-trained checkpoints run

```bash
bash bash/download_ckpts.sh
```

## Evaluation

Note that inference will require a GPU with at least 20 GB of RAM. To reproduce all results reported in Table I of the paper, run the following evaluations:

```bash
python scripts/evaluate_ambres.py --env real --model_type prompt
python scripts/evaluate_ambres.py --env sim --model_type prompt
python scripts/evaluate_ambres.py --env real --model_type finetune
python scripts/evaluate_ambres.py --env sim --model_type finetune
python scripts/evaluate_knowno.py --env real
python scripts/evaluate_knowno.py --env sim
```

## Fine-tuning your own models

Note that training will require GPUs with at least 46 GB of RAM. To start a training run use the following command:

```bash
deepspeed --include localhost:0,1 scripts/train.py --env real
```

- The `--include localhost:0,1` flag is used to limit training to GPUs 0 and 1. Leave this out if you wish to use all GPUs available. See [this doc](https://www.deepspeed.ai/getting-started/#resource-configuration-single-node) for more information.
- The `--env` flag can take argument `sim` or `real`. This flag determines which dataset you will train the model on: either on the simplified simulated images, or the real world images.
- If someone else is using distributed training, you will need to change the port:

    ```bash
    deepspeed --master_port 12344 --include localhost:0,1 scripts/train.py
    ```

- Note that to evaluate your own models, you will have to change the checkpoint name at `ambiguity_resolution/ambres/__init__.py`, in the `CKPT` class.

## License
See the [LICENSE](./LICENSE) file for details about the license under which this code is made available.
