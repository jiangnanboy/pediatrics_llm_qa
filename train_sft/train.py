import torch
import pandas as pd
import os
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
# from accelerate.utils import DistributedType

#os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_dir = '/Qwen2-1.5B-instruct'
# dataset = load_dataset("csv", data_files="data/all_data.csv", split="train")
# dataset = dataset.filter(lambda x: x["output"] is not None)
# datasets = dataset.train_test_split(test_size=0.1)

df = pd.read_csv('/data_process/all_data.csv', delimiter='@#@')
ds = Dataset.from_pandas(df, split='train')
# 查看前三条数据集
print(ds[:3])
datasets = ds.train_test_split(test_size=0.1)

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

def process_func(example):
    MAX_LENGTH = 512
    instruction = tokenizer(
        f"<|im_start|>system\n{example['instruction']}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (instruction["attention_mask"] + response["attention_mask"] + [1])
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_ds = datasets['train'].map(process_func, remove_columns=['instruction', 'input', 'output'])
tokenized_ts = datasets['test'].map(process_func, remove_columns=['instruction', 'input', 'output'])

model = AutoModelForCausalLM.from_pretrained(model_dir,
                                             device_map='auto',
                                             torch_dtype=torch.bfloat16)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

# 创建LoRA配置
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1,  # Dropout 比例
)

model = get_peft_model(model, config)

args = TrainingArguments(
    output_dir="/weights",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    logging_steps=6,
    # save_steps=5,
    save_strategy="epoch",
    num_train_epochs=20,
    learning_rate=1e-4,
    # deepspeed='/opt/doctor_llm_qa/train/ds_zero_2.json'
)
# args.distributed_state.distributed_type = DistributedType.DEEPSPEED

model.enable_input_require_grads()

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds,
    eval_dataset=tokenized_ts,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
)

trainer.train()

# 设置模型保存路径
peft_model_id = "/llm_qa_lora"

# 保存训练好的模型到指定路径
trainer.model.save_pretrained(peft_model_id)

# 保存对应的分词器到指定路径
tokenizer.save_pretrained(peft_model_id)
