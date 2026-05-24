from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "./model"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
).to(device)

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
print(text)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=38912,
    temperature=0.6,        # 控制随机性：0.6适中
    top_p=0.95,             # 核采样：保留累计概率0.95的token
    top_k=20,               # 限制候选token数为20
    min_p=0.0,              # 最小概率阈值：0表示不限制
    do_sample=True,         # 启用采样（必须设置为True才能生效temperature/top_p等参数）
    pad_token_id=tokenizer.eos_token_id  # 避免padding警告
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)