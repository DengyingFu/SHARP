import argparse
import os
import re
import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

def load_model():
    print("Loading Molmo model...")
    processor = AutoProcessor.from_pretrained(
        'allenai/Molmo-7B-D-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )
    model = AutoModelForCausalLM.from_pretrained(
        'allenai/Molmo-7B-D-0924',
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    return processor, model

def extract_points(molmo_output, image_w, image_h):
    points = []
    seen = set()
    # 正则表达式匹配 Molmo 输出的坐标格式: x="23.5" y="45.1"
    for match in re.finditer(r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"', molmo_output):
        x, y = float(match.group(1)), float(match.group(2))
        # 坐标是 normalized 到 0-100 的，需要转换回像素坐标
        if x <= 100 and y <= 100:
            pixel_x = int((x / 100) * image_w)
            pixel_y = int((y / 100) * image_h)
            if (pixel_x, pixel_y) not in seen:
                points.append((pixel_x, pixel_y))
                seen.add((pixel_x, pixel_y))
    return points

def run_inference(processor, model, image, prompt):
    image_w, image_h = image.size
    
    inputs = processor.process(images=[image], text=prompt)
    # Cast float tensors to the model's dtype to avoid mixed precision mismatches.
    model_dtype = next(model.parameters()).dtype
    inputs = {
        k: (v.to(model.device, dtype=model_dtype) if torch.is_floating_point(v) else v.to(model.device)).unsqueeze(0)
        for k, v in inputs.items()
    }
    
    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(
                max_new_tokens=500,
                do_sample=False,
                # Do not use an empty stop string; it halts generation immediately.
                stop_strings=None
            ),
            tokenizer=processor.tokenizer,
            use_cache=False
        )
        
    generated_tokens = output[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"Generated Output: {generated_text}")
    
    points = extract_points(generated_text, image_w, image_h)
    # 返回 (id, x, y) 列表。ID 从 1 开始。
    points_with_ids = [(i + 1, x, y) for i, (x, y) in enumerate(points)]
    return points_with_ids, generated_text

def annotate_image(image_path, output_dir, processor, model, prompt):
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)

    # Check if output files already exist
    output_image_path = os.path.join(output_dir, f"{name}_annotated.png")
    text_path = os.path.join(output_dir, f"{name}_id.txt")
    if os.path.exists(output_image_path) and os.path.exists(text_path):
        print(f"Skipping {image_path} as annotated files already exist.")
        return

    print(f"Processing image: {image_path}")
    try:
        original_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Failed to open image {image_path}: {e}")
        return

    points_with_ids, generated_text = run_inference(processor, model, original_image, prompt)

    # Save text result
    with open(text_path, "w") as f:
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Generated Text: {generated_text}\n\n")
        f.write("Molmo_ID X Y\n")
        for obj_id, x, y in points_with_ids:
            f.write(f"{obj_id} {x} {y}\n")
            
    # Save annotated image
    plt.figure(figsize=(10, 8))
    plt.imshow(original_image)

    for obj_id, x, y in points_with_ids:
        plt.text(
            x, y, str(obj_id),
            color="yellow", fontsize=10, fontweight="bold",
            ha="center", va="center", bbox=dict(facecolor="black", alpha=0.5, edgecolor="none")
        )
        
    plt.axis("off")
    plt.savefig(output_image_path, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()

    print(f"Saved annotated image to: {output_image_path}")
    print(f"Saved coordinates to: {text_path}")

def main():
    parser = argparse.ArgumentParser(description="Annotate objects in PNG images using Molmo.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input directory containing subfolders.")
    parser.add_argument("--prompt", "-p", type=str, default="point to all objects", help="Prompt for Molmo.")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input path {args.input} does not exist.")
        return

    processor, model = load_model()
    
    if os.path.isfile(args.input):
        # Handle single file case
        output_dir = os.path.join(os.path.dirname(args.input), "annotated")
        os.makedirs(output_dir, exist_ok=True)
        annotate_image(args.input, output_dir, processor, model, args.prompt)

    elif os.path.isdir(args.input):
        found_count = 0
        for root, dirs, files in os.walk(args.input):
            if "final_rgb.png" in files:
                image_path = os.path.join(root, "final_rgb.png")
                output_dir = os.path.join(root, "annotated")
                os.makedirs(output_dir, exist_ok=True)
                
                print(f"Found final_rgb.png in {root}. Processing...")
                annotate_image(image_path, output_dir, processor, model, args.prompt)
                found_count += 1
        
        if found_count == 0:
            print(f"No 'final_rgb.png' files found in any subdirectory of: {args.input}")

if __name__ == "__main__":
    main()
