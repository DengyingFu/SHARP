"""
Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and proprietary rights in and to this software, 
related documentation and any modifications thereto. Any use, reproduction, disclosure or distribution of this software and 
related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
"""

import os
import shortuuid
from peft import LoraConfig
from transformers import TrainingArguments
from ambres import ASSETS_DIR


os.environ["WANDB_PROJECT"] = "ambres"
run_name = shortuuid.uuid()

TRAINING_ARGS = TrainingArguments(
    optim="adamw_torch",
    bf16=True,
    tf32=True,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,  # Not supported
    learning_rate=1e-4,
    weight_decay=0.1,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    remove_unused_columns=False,
    # dataloader_num_workers=4,
    deepspeed=ASSETS_DIR.ROOT / "zero3_offload.json",
    logging_steps=1,
    eval_strategy="steps",
    eval_steps=40,
    output_dir="ckpt/" + run_name,
    run_name=run_name,
    report_to="wandb",
    save_strategy="epoch",
    save_total_limit=1,
    save_only_model=True,
)

LORA_CONFIG = LoraConfig(
    r=4,  # Rank
    lora_alpha=8,  # Scaling factor
    lora_dropout=0.05,  # Dropout probability
    bias="none",
    target_modules=["att_proj", "ff_proj", "attn_out", "ff_out"],
    task_type="CAUSAL_LM",
)

# # Not supported
# BNB_CONFIG = BitsAndBytesConfig(
#     load_in_8bit=True,  # Set to True for 8-bit, or use load_in_4bit=True for 4-bit
#     bnb_8bit_compute_dtype=torch.bfloat16,  # Use bfloat16 for stable training
#     bnb_8bit_use_double_quant=True  # Double quantization to save even more memory
# )
