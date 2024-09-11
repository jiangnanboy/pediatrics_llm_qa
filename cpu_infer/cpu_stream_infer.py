from threading import Thread
import torch
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer, TextIteratorStreamer, AutoConfig, StoppingCriteriaList, StoppingCriteria

model_dir = '/Qwen2-1.5B-Instruct-ov'

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
