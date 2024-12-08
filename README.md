Fine-tuning Large Language Models for Improving Factuality in Legal Question Answering

模型链接: https://pan.baidu.com/s/14oh-_j2xJXWMuiQ_B0fxdw?pwd=mh8p 提取码: mh8p


```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os
import json

input_json_path = "Benchmark.json"
output_json_path = "Output.json"


def get_completion(prompts, model, tokenizer=None):
    stop_token_ids = [151329, 151336, 151338]
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.9,
        top_k=50,
        max_tokens=1024,
        length_penalty=1.0,
        frequency_penalty=0.1,
        repetition_penalty=1.2,
        stop_token_ids=stop_token_ids,
        stop=["<|endoftext|>", "<|im_end|>", "```python", "<|im_sep|>", "```html"])

    llm = LLM(model=model,
              tokenizer=tokenizer,
              max_model_len=2500,
              trust_remote_code=True)

    outputs = llm.generate(prompts, sampling_params)
    return outputs


if __name__ == "__main__":
    model = "model_path"  # 指定模型路径
    tokenizer = None

    with open(input_json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    input_list = [item['instruction'] + item['problem'] for item in data]
    response_list = get_completion(input_list, model, tokenizer=tokenizer)

    data_item_all = []
    for i in range(len(response_list)):
        new_item = {
            "instruction": data[i]['instruction'],
            "laws": data[i]['laws'],
            "problem": data[i]['problem'],
            "ans": data[i]['ans'],
            "Output": response_list[i].outputs[0].text
        }
        data_item_all.append(new_item)

    with open(output_json_path, 'w', encoding='utf-8') as output_file:
        json.dump(data_item_all, output_file, ensure_ascii=False, indent=4)
        output_file.write("\n")
