# Source code for dec LLaMA
We have modified the `modeling_llama.py` file, the other files are unchanged.

## Noteworthy modifications
* Initialize the Compositions after getting the input embeddings.
* Convert query and key into tensor to linearize the attention layer.
* Convert the standard deviation to tensor to linearize the LN layer.
* Convert SiLU activation to tensor to linearize SwiGLU.