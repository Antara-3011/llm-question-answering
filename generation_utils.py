

from transformers import AutoTokenizer
from openvino.runtime import Core
import json
from config import SUPPORTED_LLM_MODELS
import numpy as np
from optimum.intel.openvino import OVModelForCausalLM
import torch
from threading import Thread
from time import perf_counter
from typing import List
from transformers import AutoTokenizer, TextIteratorStreamer
import gradio as gr
import asyncio


model_dir = r"C:\Users\KIIT\openvino_notebooks\notebooks\llm-question-answering\phi-2\INT4_compressed_weights"

model_name = "susnato/phi-2"
model_configuration = SUPPORTED_LLM_MODELS["phi-2"]
ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}

tok = AutoTokenizer.from_pretrained(model_name)
ov_model = OVModelForCausalLM.from_pretrained(model_dir, device="CPU", ov_config=ov_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer_kwargs = model_configuration.get("toeknizer_kwargs", {})

response_key = model_configuration.get("response_key")
tokenizer_response_key = None

def get_special_token_id(tokenizer: AutoTokenizer, key: str) -> int:
    token_ids = tokenizer.encode(key)
    if len(token_ids) > 1:
        raise ValueError(f"Expected only a single token for '{key}' but found {token_ids}")
    return token_ids[0]

if response_key is not None:
    tokenizer_response_key = next(
        (token for token in tokenizer.additional_special_tokens if token.startswith(response_key)),
        None,
    )

end_key_token_id = None
if tokenizer_response_key:
    try:
        end_key = model_configuration.get("end_key")
        if end_key:
            end_key_token_id = get_special_token_id(tokenizer, end_key)
    except ValueError:
        pass

prompt_template = model_configuration.get("prompt_template", "{instruction}")
end_key_token_id = end_key_token_id or tokenizer.eos_token_id
pad_token_id = end_key_token_id or tokenizer.pad_token_id

def estimate_latency(current_time: float, current_perf_text: str, new_gen_text: str, per_token_time: List[float], num_tokens: int):
    num_current_toks = len(tokenizer.encode(new_gen_text))
    num_tokens += num_current_toks
    per_token_time.append(num_current_toks / current_time)
    if len(per_token_time) > 10 and len(per_token_time) % 4 == 0:
        current_bucket = per_token_time[:-10]
        return (
            f"Average generation speed: {np.mean(current_bucket):.2f} tokens/s. Total generated tokens: {num_tokens}",
            num_tokens,
        )
    return current_perf_text, num_tokens

def run_generation(user_text: str, top_p: float, temperature: float, top_k: int, max_new_tokens: int, perf_text=str):
    prompt_text = prompt_template.format(instruction=user_text)
    model_inputs = tokenizer(prompt_text, return_tensors="pt", **tokenizer_kwargs)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        temperature=float(temperature),
        top_k=top_k,
        eos_token_id=end_key_token_id,
        pad_token_id=pad_token_id,
    )
    t = Thread(target=ov_model.generate, kwargs=generate_kwargs)
    t.start()

    model_output = ""
    per_token_time = []
    num_tokens = 0
    start = perf_counter()
    for new_text in streamer:
        current_time = perf_counter() - start
        model_output += new_text
        perf_text, num_tokens = estimate_latency(current_time, perf_text, new_text, per_token_time, num_tokens)
        yield model_output, perf_text
        start = perf_counter()
    print(model_output)    
    return model_output, perf_text

def reset_textbox(instruction: str, response: str, perf: str):
    return "", "", ""

def main():
    examples = [
    "Give me a recipe for pizza with pineapple",
    "Write me a tweet about the new OpenVINO release",
    "Explain the difference between CPU and GPU",
    "Give five ideas for a great weekend with family",
    "Do Androids dream of Electric sheep?",
    "Who is Dolly?",
    "Please give me advice on how to write resume?",
    "Name 3 advantages to being a cat",
    "Write instructions on how to become a good AI engineer",
    "Write a love letter to my best friend",
]
    with gr.Blocks() as demo:
        gr.Markdown(
            "# Question Answering with Model and OpenVINO.\n"
            "Provide instruction which describes a task below or select among predefined examples and model writes response that performs requested task."
        )
        with gr.Row():
            with gr.Column(scale=4):
                user_text = gr.Textbox(
                    placeholder="Write an email about an alpaca that likes flan",
                    label="User instruction",
                )
                model_output = gr.Textbox(label="Model response", interactive=False)
                performance = gr.Textbox(label="Performance", lines=1, interactive=False)
                with gr.Column(scale=1):
                    button_clear = gr.Button(value="Clear")
                    button_submit = gr.Button(value="Submit")
                gr.Examples(examples, user_text)
            with gr.Column(scale=1):
                max_new_tokens = gr.Slider(
                    minimum=1,
                    maximum=1000,
                    value=256,
                    step=1,
                    interactive=True,
                    label="Max New Tokens",
                )
                top_p = gr.Slider(
                    minimum=0.05,
                    maximum=1.0,
                    value=0.92,
                    step=0.05,
                    interactive=True,
                    label="Top-p (nucleus sampling)",
                )
                top_k = gr.Slider(
                    minimum=0,
                    maximum=50,
                    value=0,
                    step=1,
                    interactive=True,
                    label="Top-k",
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=5.0,
                    value=0.8,
                    step=0.1,
                    interactive=True,
                    label="Temperature",
                )

        user_text.submit(
            run_generation,
            [user_text, top_p, temperature, top_k, max_new_tokens, performance],
            [model_output, performance],
        )
        button_submit.click(
            run_generation,
            [user_text, top_p, temperature, top_k, max_new_tokens, performance],
            [model_output, performance],
        )
        button_clear.click(
            reset_textbox,
            [user_text, model_output, performance],
            [user_text, model_output, performance],
        )

    demo.queue()
    demo.launch(height=800)
if __name__ == "__main__":
    main()    