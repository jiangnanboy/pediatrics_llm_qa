from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch

model = AutoModelForCausalLM.from_pretrained("/Qwen2-1.5B-instruct", torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, "/checkpoint-lora")
# 模型合并存储
merged_model = model.merge_and_unload()
# 将权重保存为safetensors格式的权重, 且每个权重文件最大不超过4GB(4096MB)
merged_model.save_pretrained("/output_qwen_merged", max_shard_size="4096MB", safe_serialization=True)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "/Qwen2-1.5B-instruct",
    trust_remote_code=True
)

# 将tokenizer也保存到 merge_model_dir
tokenizer.save_pretrained("/output_qwen_merged")