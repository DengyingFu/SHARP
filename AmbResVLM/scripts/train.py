"""
Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and proprietary rights in and to this software, 
related documentation and any modifications thereto. Any use, reproduction, disclosure or distribution of this software and 
related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
"""

import argparse
from peft import get_peft_model
from transformers import Trainer, AutoProcessor, AutoModelForCausalLM
from ambres.training.data import SupervisedDataset, DataCollatorForSupervisedDataset
from ambres.training.params import TRAINING_ARGS, LORA_CONFIG


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    return


def main(env: str):
    assert env in ["sim", "real"]
    print(f"Training on {env} data")
    model_id = "allenai/Molmo-7B-D-0924"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype="bfloat16"
    )
    model = get_peft_model(model, LORA_CONFIG)
    print_trainable_parameters(model)
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype="bfloat16"
    )

    train_dataset = SupervisedDataset(processor, env, "train")
    eval_dataset = SupervisedDataset(processor, env, "test")
    collate_fn = DataCollatorForSupervisedDataset(processor.tokenizer.pad_token_id)

    trainer = Trainer(
        model=model,
        processing_class=processor.tokenizer,
        data_collator=collate_fn,
        args=TRAINING_ARGS,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    print(f"Output directory: {trainer.args.output_dir}")
    trainer.train()

    # Save the model
    # trainer.model.save_pretrained(trainer.args.output_dir + "/" + trainer.args.run_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--env", choices=["sim", "real"], default="real")
    args = parser.parse_args()
    main(args.env)
