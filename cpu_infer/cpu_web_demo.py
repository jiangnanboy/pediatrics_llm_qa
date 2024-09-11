from threading import Thread
import gradio as gr
import torch
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer, TextIteratorStreamer, AutoConfig, StoppingCriteriaList, StoppingCriteria

model_dir = '/output_qwen_merged-ov'

def load_model_tokenizer():
    ov_config = {"PERFORMANCE_HINT": "LATENCY",
                 "NUM_STREAMS": "1", "CACHE_DIR": ""}
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    print("====Compiling model====")
    ov_model = OVModelForCausalLM.from_pretrained(
        model_dir,
        device='CPU',
        ov_config=ov_config,
        config=AutoConfig.from_pretrained(model_dir),
        trust_remote_code=True,
    )

    return ov_model, tokenizer

class StopOnTokens(StoppingCriteria):
    def __init__(self, token_ids):
        self.token_ids = token_ids

    def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_id in self.token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

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
        max_new_tokens=512,
        streamer=streamer,
        stopping_criteria=StoppingCriteriaList([StopOnTokens([151643, 151645])]),
        pad_token_id=151645,
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