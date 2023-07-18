# RoBERTa: A Robustly Optimized BERT Pretraining Approach

https://arxiv.org/abs/1907.11692

## Introduction

RoBERTa iterates on BERT's pretraining procedure, including training the model longer, with bigger batches over more data; removing the next sentence prediction objective; training on longer sequences; and dynamically changing the masking pattern applied to the training data. See the associated paper for more details.

We provide two implementations for [Fairseq](https://github.com/facebookresearch/fairseq) and [Hugging Face transformers](https://github.com/huggingface/transformers), respectively.

## Example usage
### Run decomposition with Fairseq
#### Setup
Make sure you have Fairseq installed, read [here](https://github.com/facebookresearch/fairseq#requirements-and-installation) to install it.

To obtain the RoBERTa pre-trained models, please refer to [here](https://github.com/facebookresearch/fairseq/tree/main/examples/roberta).

To finetune RoBERTa on the sentence classification tasks, please refer to [Finetuning RoBERTa on GLUE tasks](https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.glue.md) or [Finetuning RoBERTa on a custom classification task](https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.custom_classification.md).

To make Fairseq load a checkpoint and run on our code, we need to change the architecture name of the checkpoint. That is, change `roberta_base` to `dec_roberta_base` or `roberta_large` to `dec_roberta_large`.

We provide a script for changing the architecture name:
```bash
python change_fairseq_architecture.py $PATH_TO_CHECKPOINT
```
This will change the architecture name of the input checkpoint and save it to `checkpoints/checkpoint.dec.pt`.

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

label2str = lambda label: "Positive" if label == "1" else "Negative"

roberta.cuda()
roberta.eval()
with open("glue_data/SST-2/dev.tsv") as fin:
    fin.readline()  # table header
    line = fin.readline()  # first sample
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
    print("Input: {}".format(sent))
    print(
        "Label: {}, Prediction: {}".format(
            label2str(label), label2str(prediction_label)
        )
    )
    print("Logits of prediction:")
    print(logits[0, prediction])
    print("Decompositoin of the logits:")
    print(c_logits[0, prediction])

    # print the score for each word piece
    word_pieces = [
        roberta.bpe.decode(item) for item in roberta.bpe.encode(sent).split()
    ]
    word_pieces = ["<cls>"] + word_pieces + ["<eos>"]
    print("\n{:<16}| {}".format("Word piece", "Score"))
    for word_piece, score in zip(word_pieces, c_logits[0, prediction].components):
        print("{:<16}| {:.2f}".format('"' + word_piece + '"', score.item()))
```

Output:
```
Input: it 's a charming and often affecting journey . 
Label: Positive, Prediction: Positive
Logits of prediction:
tensor(3.3450, device='cuda:0')
Decompositoin of the logits:
composition{
  components:
    tensor(0.2433),
    tensor(0.1477),
    tensor(0.0596),
    tensor(0.1328),
    tensor(0.1651),
    tensor(1.0371),
    tensor(0.2327),
    tensor(0.3910),
    tensor(0.2959),
    tensor(0.4994),
    tensor(0.0403),
    tensor(0.0095),
    tensor(0.0023),
  residual:
    tensor(0.0884),
  device='cuda:0'}

Word piece      | Score
"<cls>"         | 0.24
"it"            | 0.15
" '"            | 0.06
"s"             | 0.13
" a"            | 0.17
" charming"     | 1.04
" and"          | 0.23
" often"        | 0.39
" affecting"    | 0.30
" journey"      | 0.50
" ."            | 0.04
" "             | 0.01
"<eos>"         | 0.00
```

### Run decomposition with Hugging Face transformers
Not ready yet, coming soon.
