# convert_lfm2_to_rknn

## why have this
Of course, this is not a Causal LM use case; it's just a demo to show how you can transform the LFM2 model to Rknn. If you are doing some fine-tuning, e.g., transfer the model from LM to the emb model(And do some CLS task or VectorDatabase), this could help.

## performance
In LFM2-350M
With a fixed lens of 128, test CPU interface time: 2995ms, and NPU interface time: 321ms
