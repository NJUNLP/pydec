# LLaMA: Open and Efficient Foundation Language Models
https://arxiv.org/abs/2302.13971

## Introduction
LLaMA is an auto-regressive language model, based on the transformer architecture. The model comes in different sizes: 7B, 13B, 33B and 65B parameters.

This implementation is based on **[Hugging Face LLaMA](https://huggingface.co/docs/transformers/model_doc/llama)**, and we have made few modifications to the code to generate the decomposition of the output.

## Example usage
#### Setup
Make sure you have Hugging Face Transformers installed, read [here](https://huggingface.co/docs/transformers/installation) to install it.

To use it, first prepare the model checkpoints and configuration information. [Here](https://huggingface.co/yahma/llama-7b-hf) is an example, as a ready-to-use resource.


#### Decompose LLaMA in generation

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

print("\n\nThe decomposition for each token:")
input_len = inputs.input_ids.size(1)
for step, composition in enumerate(
    generate_output["attentions"]
):  # We occupy the 'attentions' field of the output to store the decomposition results
    print(
        "\n\nGenerated token: '{}'".format(
            tokenizer.decode(generate_output["sequences"][0, input_len + step])
        ),
    )

    pred = composition.recovery.argmax(dim=0)
    print("Decomposition:")
    print("{:<16}| {}".format("Word piece", "Score"))
    for token_idx, score in enumerate(composition[pred].components):
        print(
            "{:<16}| {:.2f}".format(
                ascii(tokenizer.decode([generate_output["sequences"][0, token_idx]])),
                score.item(),
            )
        )

```

Output:
```
Generated results:
Instruction:
As a language model, tell me about your name.

### Response:
My name is Vicuna, and I'm a language model developed by Large Model Systems Organization (LMSYS).


The decomposition for each token:


Generated token: 'My'
Decomposition:
Word piece      | Score
'<s>'           | 7.36
'Inst'          | -0.00
'ruction'       | -0.07
':'             | 0.79
'\n'            | 0.24
'As'            | 0.49
'a'             | 0.04
'language'      | 0.26
'model'         | 0.12
','             | 0.03
'tell'          | 0.52
'me'            | 0.46
'about'         | 0.66
'your'          | 0.91
'name'          | 1.47
'.'             | 0.27
'\n'            | 0.02
'\n'            | 0.10
'##'            | 0.50
'#'             | 0.12
'Response'      | 1.21
':'             | -0.06
'\n'            | 0.21

...
```


## To reduce memory usage
[Here](https://doublevii.github.io/pydec/#/reduce-memory-overhead) are some suggestions to reduce memory overhead.

The [8bit quantization](https://huggingface.co/docs/transformers/v4.30.0/en/perf_infer_gpu_one#bitsandbytes-integration-for-int8-mixedprecision-matrix-decomposition) is not currently supported.