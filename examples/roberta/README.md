# RoBERTa: A Robustly Optimized BERT Pretraining Approach

https://arxiv.org/abs/1907.11692

## Introduction

RoBERTa iterates on BERT's pretraining procedure, including training the model longer, with bigger batches over more data; removing the next sentence prediction objective; training on longer sequences; and dynamically changing the masking pattern applied to the training data. See the associated paper for more details.

We provide two implementations for [Fairseq](https://github.com/facebookresearch/fairseq) and [Hugging Face transformers](https://github.com/huggingface/transformers), respectively.

## Example usage
### Run decomposition with Fairseq
#### Setup
To obtain the RoBERTa pre-trained models, please refer to [here](https://github.com/facebookresearch/fairseq/tree/main/examples/roberta).

To finetune RoBERTa on the sentence classification tasks, please refer to [Finetuning RoBERTa on GLUE tasks](https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.glue.md) or [Finetuning RoBERTa on a custom classification task](https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.custom_classification.md).

To make Fairseq load a checkpoint and run on our code, we need to change the architecture name of the checkpoint. That is, change `roberta_base` to `dec_roberta_base` or `roberta_large` to `dec_roberta_large`.

We provide a script for changing the architecture name:
```bash
python change_fairseq_architecture.py $PATH_TO_CHECKPOINT
```
This will change the architecture name of the entered checkpoint and save it to `checkpoints/checkpoint.dec.pt`.

#### Decompose RoBERTa in sentence classification tasks
```python
import torch
from dec_roberta_for_fairseq import RobertaModel

roberta = RobertaModel.from_pretrained(
    "checkpoints/",
    checkpoint_file="checkpoint.dec.pt",
    data_name_or_path="SST-2-bin",
)

label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.label_dictionary.nspecial]
)

roberta.cuda()
roberta.eval()
with open("/home/yangs/data/glue_data/SST-2/dev.tsv") as fin:
    fin.readline()  # table header
    line = fin.readline() # first sample
    sent, label = line.strip().split("\t")
    tokens = roberta.encode(sent)
    with torch.no_grad():
        c_logits = roberta.predict(
            "sentence_classification_head",
            tokens,
            return_logits=True,
            decompose=True,
        )
    logits = c_logits.c_sum()
    prediction = logits.argmax().item()
    prediction_label = label_fn(prediction)
    print("Label: {}, Prediction: {}".format(label, prediction_label))
    print("Logits of prediction:")
    print(logits[0, prediction])
    print("Decompositoin of the logits:")
    print(c_logits[0, prediction])
```


### Run decomposition with Hugging Face transformers
Not ready yet, coming soon.
