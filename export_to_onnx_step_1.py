import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import onnx
import onnx.checker
import numpy as np
import warnings

# You could change to 700M 1.2B Here, I see no problem
model_id = "LiquidAI/LFM2-350M"

# Load model and tokenizer with float32 to avoid bfloat16 issues 
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu",  # Load on CPU for export
    torch_dtype=torch.float32,  # Use float32 to avoid bfloat16 issues
    trust_remote_code=True,
    attn_implementation="eager",  # Use eager attention (worked for ComplexDouble)
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Set model to evaluation mode
model.eval()

# Custom wrapper to control forward pass
class Lfm2ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, position_ids):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,  # Disable caching
            return_dict=True,
        )
        return outputs.last_hidden_state  # Return only last_hidden_state

# Wrap the model
wrapped_model = Lfm2ModelWrapper(model.model)

# Define fixed input length, also you cloud change, or may use dyn axis, but in most cls cases, a fixed lens should work.
fixed_length = 128

# Prepare sample input for ONNX export
prompt = "Who are you"
inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True,
    return_tensors="pt",
    tokenize=True,
    return_dict=True,
    max_length=fixed_length,
    padding="max_length",
)
input_ids = inputs['input_ids'].to(torch.int64)  # Ensure int64 for ONNX
attention_mask = inputs['attention_mask'].to(torch.int64)
position_ids = torch.arange(fixed_length, dtype=torch.int64).unsqueeze(0)

# Export the wrapped model to ONNX
onnx_path = "lfm_model_float32.onnx"
try:
    torch.onnx.export(
        wrapped_model,
        (input_ids, attention_mask, position_ids),
        onnx_path,
        input_names=["input_ids", "attention_mask", "position_ids"],
        output_names=["last_hidden_state"],
        # dynamic_axes={
        #     "input_ids": {0: "batch_size", 1: "sequence_length"},
        #     "attention_mask": {0: "batch_size", 1: "sequence_length"},
        #     "position_ids": {0: "batch_size", 1: "sequence_length"},
        #     "last_hidden_state": {0: "batch_size", 1: "sequence_length", 2: "hidden_size"},
        # },
        opset_version=17,
        do_constant_folding=True,
        verbose=False,
    )
    print(f"Model exported to {onnx_path}")

    # Validate the ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed")
except Exception as e:
    print(f"ONNX export failed: {e}")
    exit(1)

# Auxiliary code for inference
import onnxruntime as ort

def prepare_inputs(prompt, tokenizer, max_length=128):
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
        max_length=max_length,
        padding="max_length",
    )
    input_ids = inputs['input_ids'].to(torch.int64).numpy()
    attention_mask = inputs['attention_mask'].to(torch.int64).numpy()
    position_ids = torch.arange(max_length, dtype=torch.int64).unsqueeze(0).numpy()
    return input_ids, attention_mask, position_ids

# Run inference with ONNX model
try:
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_ids, attention_mask, position_ids = prepare_inputs("Who are you", tokenizer, fixed_length)
    outputs = session.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    })
    last_hidden_state = outputs[0]
    print(f"ONNX model output shape: {last_hidden_state.shape}")

    # Process output
    last_non_padded_idx = np.where(attention_mask[0] != 0)[0][-1]
    logits = last_hidden_state[0][last_non_padded_idx]
    argmax_ids = np.argmax(logits, axis=-1).tolist()
    decoded_text = tokenizer.decode([argmax_ids])
    print(f"Argmax ID: {argmax_ids}")
    print(f"Decoded text: {decoded_text}")
except Exception as e:
    print(f"Inference failed: {e}")
