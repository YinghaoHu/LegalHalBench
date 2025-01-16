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
```

评测集样例展示：
```python
{
        "instruction": "请你作为一名法官，给我相关的建议和明确的法条。",
        "laws": [
            "《民法典 侵权责任编》第1169条规定：“教唆、帮助他人实施侵权行为的，应当与行为人承担连带责任。教唆、帮助无民事行为能力人、限制民事行为能力人实施侵权行为的，应当承担侵权责任；该无民事行为能力人、限制民事行为能力人的监护人未尽到监护职责的，应当承担相应的责任。“"
        ],
        "problem": "假如我无意间借给了一位朋友一些工具，而他之后用这些工具损坏了别人的财产，我在完全不知情的情况下，是否需要承担法律责任？",
        "ans": "咨询者想知道如果他无意间借给朋友一些工具，而朋友用这些工具损坏了别人的财产，在自己不知情的情况下是否需要承担法律责任。《民法典 侵权责任编》第1169条规定：“教唆、帮助他人实施侵权行为的，应当与行为人承担连带责任。教唆、帮助无民事行为能力人、限制民事行为能力人实施侵权行为的，应当承担侵权责任；该无民事行为能力人、限制民事行为能力人的监护人未尽到监护职责的，应当承担相应的责任。“根据您的描述，您是无意间借出工具给朋友的，并且您对朋友使用这些工具损害他人财产的行为完全不知情。根据《民法典 侵权责任编》第1169条的规定，仅有在教唆或帮助他人实施侵权行为的情况下，才需要与行为人承担连带责任。在您的情况下，没有证据表明您教唆或帮助了朋友实施侵权行为，因此您应不需要承担侵权责任。\n尽管您可能不需要承担法律责任，为防止类似情况再次发生，建议在借出任何物品时与借用者明确使用条件和范围，或者采取书面形式记录。这样可以为您提供更多的安全保障，并减少未来潜在的法律风险。同时，保持对借出物品的跟踪和管理也是一个良好的预防措施。"
}
```
