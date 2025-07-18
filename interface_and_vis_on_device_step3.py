import os
import time
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as ort
from transformers import AutoTokenizer
from rknn.api import RKNN
import traceback

# --- Configuration ---
ONNX_MODEL_PATH = "lfm_model_float32.onnx"
RKNN_MODEL_PATH = "./lfm2-350.rknn"
MODEL_ID = "LiquidAI/LFM2-350M"
MAX_SEQ_LENGTH = 128  # Must match the length used for ONNX export/RKNN conversion

# Directory for saving visualization plots
SAVE_DIR = "embedding_plots"
FILENAME_PREFIX_ONNX = "onnx_embedding_heatmap"
FILENAME_PREFIX_RKNN = "rknn_embedding_heatmap"

# --- Visualization Function ---
def visualize_embedding_heatmap(embedding_vector, title="Embedding Heatmap", filename_prefix="embedding_heatmap", vmin=None, vmax=None, cmap='viridis'):
    """
    Visualizes a 1D embedding vector as a 2D heatmap.
    
    Args:
        embedding_vector (np.ndarray): 1D NumPy array.
        title (str): Title for the plot.
        filename_prefix (str): Prefix for the saved filename.
        vmin (float, optional): Minimum value for the color scale.
        vmax (float, optional): Maximum value for the color scale.
        cmap (str): Matplotlib colormap name.
    """
    if not isinstance(embedding_vector, np.ndarray):
        print("Error: Input must be a NumPy array.")
        return

    embedding_vector = embedding_vector.flatten()
    dim = embedding_vector.shape[0]
    
    # Reshape to 2D for visualization (adjust dimensions as needed)
    target_height = 24
    target_width = int(np.ceil(dim / target_height))
    if dim <= target_height * target_width:
        shape_2d = (target_height, target_width)
        print(f"Visualizing '{title}' (dim: {dim}, reshaped to: {shape_2d})")
    else:
        print(f"Warning: Dimension {dim} too large. Visualizing as 1x{dim}.")
        shape_2d = (1, dim)
    
    try:
        reshaped_embedding = np.pad(embedding_vector, (0, target_height * target_width - dim), 'constant').reshape(shape_2d)
    except ValueError as e:
        print(f"Error: Cannot reshape dimension {dim} to {shape_2d}. {e}")
        return

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"Created directory: {SAVE_DIR}")

    plt.figure(figsize=(target_width / 3, target_height / 3))
    im = plt.imshow(reshaped_embedding, cmap=cmap, aspect='equal', interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.title(title, fontsize=10)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(im, fraction=0.046, pad=0.04)
    
    heatmap_path = os.path.join(SAVE_DIR, f"{filename_prefix}.png")
    try:
        plt.savefig(heatmap_path, dpi=150)
        print(f"Heatmap saved to: {heatmap_path}")
    except Exception as e:
        print(f"Error saving heatmap: {e}")
    plt.close()

# --- Load Tokenizer ---
print(f"--> Loading tokenizer: {MODEL_ID}")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    traceback.print_exc()
    exit(1)

# --- Prepare Input Data ---
print('--> Preparing input data')
input_text = "Who are you"  # Test prompt
print(f"Processing text: '{input_text}'")

tokenized_input = tokenizer.apply_chat_template(
    [{"role": "user", "content": input_text}],
    add_generation_prompt=True,
    return_dict=True,
    tokenize=True,
    max_length=MAX_SEQ_LENGTH,
    padding="max_length",
    return_tensors="np",
)

input_ids_np = tokenized_input['input_ids'].astype(np.int64)
attention_mask_np = tokenized_input['attention_mask'].astype(np.int64)
position_ids_np = np.arange(MAX_SEQ_LENGTH, dtype=np.int64).reshape(1, MAX_SEQ_LENGTH)

print(f"Input IDs shape: {input_ids_np.shape}, dtype: {input_ids_np.dtype}")
print(f"Attention Mask shape: {attention_mask_np.shape}, dtype: {attention_mask_np.dtype}")
print(f"Position IDs shape: {position_ids_np.shape}, dtype: {position_ids_np.dtype}")

expected_shape = (1, MAX_SEQ_LENGTH)
if (input_ids_np.shape != expected_shape or 
    attention_mask_np.shape != expected_shape or 
    position_ids_np.shape != expected_shape):
    print(f"Error: Input shapes do not match expected {expected_shape}")
    exit(1)

# --- ONNX Inference ---
print('--> Running ONNX inference on CPU')
try:
    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
    start_time_onnx = time.perf_counter()
    onnx_outputs = session.run(None, {
        "input_ids": input_ids_np,
        "attention_mask": attention_mask_np,
        "position_ids": position_ids_np,
    })
    end_time_onnx = time.perf_counter()
    onnx_time = (end_time_onnx - start_time_onnx) * 1000  # Convert to ms
    print(f"ONNX inference completed in {onnx_time:.4f} ms")
    
    onnx_embedding = onnx_outputs[0]  # Shape: [1, seq_length, hidden_size]
    print(f"ONNX output shape: {onnx_embedding.shape}")
    
    # Extract last non-padded token's embedding
    last_non_padded_idx = np.where(attention_mask_np[0] != 0)[0][-1]
    onnx_embedding_vector = onnx_embedding[0, last_non_padded_idx]
    print(f"ONNX embedding vector shape: {onnx_embedding_vector.shape}")
    
    # Visualize ONNX embedding
    visualize_embedding_heatmap(
        onnx_embedding_vector,
        title="ONNX Embedding Heatmap",
        filename_prefix=FILENAME_PREFIX_ONNX
    )
except Exception as e:
    print(f"ONNX inference failed: {e}")
    traceback.print_exc()
    exit(1)

# --- RKNN Inference ---
print('--> Loading RKNN model')
rknn = RKNN(verbose=True)

print('--> config model')
rknn.config(target_platform='rk3588')
print('done')

ret = rknn.load_rknn(RKNN_MODEL_PATH)
if ret != 0:
    print(f"Error loading RKNN model: {RKNN_MODEL_PATH}")
    exit(ret)
print('done')

print('--> Initializing RKNN runtime')
ret = rknn.init_runtime(core_mask=RKNN.NPU_CORE_0_1_2, target='rk3588')
if ret != 0:
    print('Error initializing RKNN runtime!')
    rknn.release()
    exit(ret)
print('done')

print('--> Running RKNN inference on NPU')
try:
    start_time_rknn = time.perf_counter()
    rknn_outputs = rknn.inference(inputs=[input_ids_np, attention_mask_np, position_ids_np], data_format=['nchw', 'nchw', 'nchw'])
    end_time_rknn = time.perf_counter()
    rknn_time = (end_time_rknn - start_time_rknn) * 1000  # Convert to ms
    print(f"RKNN inference completed in {rknn_time:.4f} ms")
    
    rknn_embedding = rknn_outputs[0]  # Shape: [1, seq_length, hidden_size]
    print(f"RKNN output shape: {rknn_embedding.shape}")
    
    # Extract last non-padded token's embedding
    rknn_embedding_vector = rknn_embedding[0, last_non_padded_idx]
    print(f"RKNN embedding vector shape: {rknn_embedding_vector.shape}")
    
    # Visualize RKNN embedding
    visualize_embedding_heatmap(
        rknn_embedding_vector,
        title="RKNN Embedding Heatmap",
        filename_prefix=FILENAME_PREFIX_RKNN,
        vmin=onnx_embedding_vector.min(),  # Use same scale as ONNX for comparison
        vmax=onnx_embedding_vector.max()
    )
except Exception as e:
    print(f"RKNN inference failed: {e}")
    traceback.print_exc()
    rknn.release()
    exit(1)

# --- Compare Outputs ---
print('--> Comparing ONNX and RKNN outputs')
diff = np.abs(onnx_embedding_vector - rknn_embedding_vector)
mean_diff = np.mean(diff)
max_diff = np.max(diff)
print(f"Mean absolute difference: {mean_diff:.6f}")
print(f"Max absolute difference: {max_diff:.6f}")

# Visualize difference heatmap
visualize_embedding_heatmap(
    diff,
    title="Absolute Difference (ONNX vs RKNN)",
    filename_prefix="difference_heatmap",
    cmap='hot'
)

# --- Release RKNN context ---
print('--> Releasing RKNN context')
rknn.release()
print('done')

# --- Print Timing Comparison ---
print(f"\nTiming Comparison:")
print(f"ONNX (CPU): {onnx_time:.4f} ms")
print(f"RKNN (NPU): {rknn_time:.4f} ms")
print(f"Speedup (ONNX/RKNN): {onnx_time/rknn_time:.2f}x")
