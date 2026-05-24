"""
Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and proprietary rights in and to this software, 
related documentation and any modifications thereto. Any use, reproduction, disclosure or distribution of this software and 
related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
"""

"""Adapted from https://github.com/2U1/Molmo-Finetune"""

import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor
from ambres import ASSETS_DIR, DATA_DIR
from ambres.dataset import Sample


def process_msg_list(
    messages: list[dict], images: list[Image.Image], processor: AutoProcessor, append_eos=False
):
    """
    IMPORTANT: This function assumes that the image placeholder is always in the first message!!
    """
    eos_token_id = processor.tokenizer.eos_token_id
    all_input_ids = []
    all_labels = []
    for i, message in enumerate(messages):
        role = message["role"]
        if role == "user":
            assert i % 2 == 0
            if i == 0:
                user_prompt = message["content"]
                inputs = processor.process(text=user_prompt, images=images)
                prompt_input_ids = inputs["input_ids"]
            else:
                user_prompt = f" User: {message['content']} Assistant:"
                prompt_input_ids = processor.tokenizer(
                    user_prompt, add_special_tokens=False, padding=False, return_tensors="pt"
                )["input_ids"].squeeze(0)
            all_input_ids.append(prompt_input_ids)
            all_labels.append(torch.tensor([-100] * len(prompt_input_ids)))
        if role == "assistant":
            assert i % 2 == 1
            assistant_prompt = f" {message['content']}"
            assistant_input_ids = processor.tokenizer(
                assistant_prompt, add_special_tokens=False, padding=False, return_tensors="pt"
            )["input_ids"].squeeze(0)
            all_input_ids.append(assistant_input_ids)
            all_labels.append(assistant_input_ids)

    if append_eos:
        all_input_ids.append(torch.tensor([eos_token_id]))  # eos token id
        all_labels.append(torch.tensor([eos_token_id]))  # eos token id
    input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
    labels = torch.cat(all_labels, dim=0).to(torch.long)
    data_dict = dict(
        input_ids=input_ids,
        labels=labels,
        images=inputs["images"],
        image_input_idx=inputs["image_input_idx"],
        image_masks=inputs["image_masks"],
    )
    ## Debugging
    # print(processor.tokenizer.decode(input_ids, skip_special_tokens=True))
    return data_dict


class SupervisedDataset(Dataset):
    def __init__(self, processor, env="real", mode="train"):
        assert env in ["sim", "real"]
        assert mode in ["train", "test"]
        self.image_folder = DATA_DIR.get_dir(env, mode)
        self.processor = processor
        with open(ASSETS_DIR.get_dir(env, mode) / "data.json", "r") as f:
            loaded_data = json.load(f)
        self.data = [Sample(**d).as_messages() for d in loaded_data]
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        image_files = example["image"]
        image_path = self.image_folder / image_files
        image = Image.open(image_path).convert("RGB")
        image = image.reduce(4)
        images = [image]
        data_dict = process_msg_list(example["messages"], images, self.processor, append_eos=True)
        return data_dict


class DataCollatorForSupervisedDataset:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_images = []
        batch_image_input_idx = []
        batch_image_mask = []
        for example in examples:
            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])
            batch_images.append(example["images"])
            batch_image_input_idx.append(example["image_input_idx"])
            batch_image_mask.append(example["image_masks"])

        input_ids = self.stack_with_padding(
            batch_input_ids, padding_side="right", padding_value=self.pad_token_id
        )
        labels = self.stack_with_padding(batch_label_ids, padding_side="right", padding_value=-100)
        attention_mask = (input_ids != self.pad_token_id).to(torch.long)
        images = torch.stack(batch_images)
        image_input_idx = torch.stack(batch_image_input_idx)
        image_masks = torch.stack(batch_image_mask)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "images": images,
            "image_input_idx": image_input_idx,
            "image_masks": image_masks,
        }

    def stack_with_padding(self, sequences, padding_side="right", padding_value=0):
        """
        Pad a list of sequences to the same length.
        sequences: list of tensors in [seq_len, *] shape
        """
        assert padding_side in ["right", "left"]
        batch_size = len(sequences)
        max_len = max(len(seq) for seq in sequences)
        trailing_dims = sequences[0].shape[1:]
        output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
        for i, seq in enumerate(sequences):
            length = seq.size(0)
            if padding_side == "right":
                output.data[i, :length] = seq
            else:
                output.data[i, -length:] = seq
        return output


if __name__ == "__main__":
    from transformers import AutoProcessor

    model_id = "allenai/Molmo-7B-D-0924"
    molmo_processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype="bfloat16",
        device_map="cuda",
    )
    dataset = SupervisedDataset(molmo_processor)
    sample = dataset[0]
    print(sample)
