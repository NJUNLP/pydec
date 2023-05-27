# LLaMA: Open and Efficient Foundation Language Models
https://arxiv.org/abs/2302.13971

## Introduction
LLaMA is an auto-regressive language model, based on the transformer architecture. The model comes in different sizes: 7B, 13B, 33B and 65B parameters.

This implementation is based on **[Hugging Face LLaMA](https://huggingface.co/docs/transformers/model_doc/llama)**, and we have made few modifications to the code to generate the decomposition of the output.

## Example usage
To use it, first prepare the model checkpoints and configuration information.

##### decompose LLaMA in generation:

We occupy the 'attentions' field of the output to store the decomposition results

```python
import torch
from dec_llama import LlamaForCausalLM, LlamaTokenizer

model_dir = "/path/to/llama"
tokenizer = LlamaTokenizer.from_pretrained(model_dir)
model = LlamaForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,  # recommended to save memory
    device_map="auto",
)

inputs = tokenizer(
    "Instruction:\nAs a language model, tell me about your name.\n\n### Response:\n",
    return_tensors="pt",
).to("cuda")
generate_output = model.generate(
    inputs.input_ids,
    max_length=500,
    decompose=True,
    use_cache=True,
    return_dict_in_generate=True,
    output_attentions=True,  # use to store the decomposition results
)
generate_results = tokenizer.batch_decode(
    generate_output["sequences"],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]

print("Generated results:\n{}".format(generate_results))

print("The decomposition for each token:")
input_len = inputs.input_ids.size(1)
for step, composition in enumerate(
    generate_output["attentions"]
):  # We occupy the 'attentions' field of the output to store the decomposition results
    print(
        "Generated token: '{}'".format(
            tokenizer.decode(generate_output["sequences"][0, input_len + step])
        ),
    )
    print("Decomposition:")
    for token_idx, score in enumerate(composition):
        print(
            "'{}': {}".format(
                tokenizer.decode(generate_output["sequences"][0, token_idx]), score
            )
        )
    print("\n")
```

## To reduce memory usage
TODO