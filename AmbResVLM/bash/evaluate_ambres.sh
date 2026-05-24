#!/usr/bin/env bash

python scripts/evaluate_ambres.py --env real --model_type prompt
python scripts/evaluate_ambres.py --env real --model_type finetune
python scripts/evaluate_ambres.py --env sim --model_type prompt
python scripts/evaluate_ambres.py --env sim --model_type finetune