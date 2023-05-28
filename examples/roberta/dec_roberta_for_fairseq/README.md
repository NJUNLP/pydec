# Source code for dec RoBERTa
## Noteworthy modifications
* Initialize the Compositions after getting the input embeddings.
* Convert query and key into tensor to linearize the attention layer.