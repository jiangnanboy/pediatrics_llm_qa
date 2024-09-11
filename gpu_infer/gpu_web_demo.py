from threading import Thread
import os
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

Model_PATH = '/Qwen2-1.5B-instruct'
LORA_CHECKPOINT_PATH = '/weights/checkpoint-lora'

def load_model_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(Model_PATH, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(Model_PATH, device_map=DEVICE,
                                                 torch_dtype=torch.bfloat16).eval()

    # 加载训练好的Lora模型，将下面的checkpointXXX替换为实际的checkpoint文件名名称
    model = PeftModel.from_pretrained(model, model_id=LORA_CHECKPOINT_PATH)

    model.generation_config.max_new_tokens = 512   # For chat.

    return model, tokenizer

def chat_stream(model, tokenizer, query, history):
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

def launch_demo(model, tokenizer):

    def predict(query, chatbot, task_history):
        print(f"User: {query}")
        chatbot.append((query, ""))
        full_response = ""
        response = ""
        for new_text in chat_stream(model, tokenizer, query, history=task_history):
            response += new_text
            chatbot[-1] = (query, response)

            yield chatbot
            full_response = response

        print(f"History: {task_history}")
        task_history.append((query, full_response))
        print(f"Instruct: {full_response}")

    def regenerate(chatbot, task_history):
        if not task_history:
            yield chatbot
            return
        item = task_history.pop(-1)
        chatbot.pop(-1)
        yield from predict(item[0], chatbot, task_history)

    def reset_user_input():
        return gr.update(value="")

    def reset_state(chatbot, task_history):
        task_history.clear()
        chatbot.clear()
        gc()
        return chatbot

    with gr.Blocks() as demo:
        gr.Markdown(
            """\
<center><font size=3>本项目是儿科问诊对话机器人, developed by Jiangnanboy.</center>""")
        chatbot = gr.Chatbot(label='XiaoJiaBot', elem_classes="control-height")
        query = gr.Textbox(lines=2, label='Input')
        task_history = gr.State([])

        with gr.Row():
            empty_btn = gr.Button("🧹 Clear History (清除历史)")
            submit_btn = gr.Button("🚀 Submit (发送)")
            regen_btn = gr.Button("🤔️ Regenerate (重试)")

        submit_btn.click(predict, [query, chatbot, task_history], [chatbot], show_progress=True)
        submit_btn.click(reset_user_input, [], [query])
        empty_btn.click(reset_state, [chatbot, task_history], outputs=[chatbot], show_progress=True)
        regen_btn.click(regenerate, [chatbot, task_history], [chatbot], show_progress=True)

        gr.Markdown("""\
<font size=2>Note: This demo is governed by the original license of Jiangnanboy. \
We strongly advise users not to knowingly generate or allow others to knowingly generate harmful content, \
including hate speech, violence, pornography, deception, etc. \
(注：本演示受Jiangnanboy的许可协议限制。我们强烈建议，用户不应传播及不应允许他人传播以下内容，\
包括但不限于仇恨言论、暴力、色情、欺诈相关的有害信息。)""")

    demo.queue().launch(
        share=False,
        inbrowser=False,
        server_port=8000,
        server_name='0.0.0.0',
    )

def main():
    model, tokenizer = load_model_tokenizer()
    launch_demo(model, tokenizer)

if __name__ == '__main__':
    main()