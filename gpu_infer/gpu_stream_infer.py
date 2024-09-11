from threading import Thread
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

Model_PATH = '/Qwen2-1.5B-instruct'
LORA_CHECKPOINT_PATH = '/weights/checkpoint-lora'

def load_model_tokenizer():
    # 加载原tokenizer和model
    tokenizer = AutoTokenizer.from_pretrained(Model_PATH, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(Model_PATH, device_map=DEVICE,
                                                 torch_dtype=torch.bfloat16).eval()

    # 加载训练好的Lora模型，将下面的checkpointXXX替换为实际的checkpoint文件名名称
    model = PeftModel.from_pretrained(model, model_id=LORA_CHECKPOINT_PATH)

    model.generation_config.max_new_tokens = 512   # For chat.
    return model, tokenizer

model, tokenizer = load_model_tokenizer()


def chat_stream(query, history):
    conversation = [
        {'role': 'system', 'content': "你是儿科专家问答助手小嘉，你将帮助用户解答基础的医疗问题，下面是用户提出的问题，请根据实际医疗知识进行回答，" \
                              "答案应条理清晰，有层次，不要有任何杜撰的内容，对于无法回答的问题，请用'您的问题我暂时无法回答！'进行回答。"},
    ]
    for query_h, response_h in history:
        conversation.append({'role': 'user', 'content': query_h})
        conversation.append({'role': 'assistant', 'content': response_h})
    conversation.append({'role': 'user', 'content': query})
    inputs = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors='pt',
    )
    inputs = inputs.to(model.device)
    streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True, timeout=60.0, skip_special_tokens=True)
    generation_kwargs = dict(
        input_ids=inputs,
        streamer=streamer,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    for new_text in streamer:
        yield new_text

def gc():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main(query):
    response = ''
    for new_text in chat_stream(query, history=[]):
        print(new_text, end='', flush=True)
        response += new_text

if __name__ == '__main__':
    while True:
        query = input('输入问题：')
        if query == 'exit':
            break
        main(query)
