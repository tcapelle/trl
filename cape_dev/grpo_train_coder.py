import torch
import requests
from io import BytesIO
import re
from dataclasses import dataclass
from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
import wandb
import accelerate
import simple_parsing as sp
from datasets import load_dataset, Dataset
from functools import lru_cache
import hashlib

# ===== Configuration =====

SYSTEM_PROMPT = """
# CUDA Kernel Optimization Task

## Objective
Your task is to optimize PyTorch models by replacing standard PyTorch operators with custom CUDA kernels. You should:
- Choose which operators to replace with custom implementations
- Consider operator fusion opportunities (e.g., combining matmul+relu)
- Explore algorithmic optimizations (e.g., online softmax)
- Rename your optimized implementation as "ModelNew"

## Key GPU Programming Concepts
- Thread: A thread is a single execution unit that can run a single instruction at a time.
- Thread Block: A thread block is a group of threads that can cooperate with each other.
- Warp: A warp is a group of threads that are scheduled together and execute in parallel.
- Shared Memory: Shared memory is a memory space that can be accessed by all threads in a thread block.
- Register: A register is a small memory space that can be accessed by a single thread.
- Memory Hierarchy: Memory hierarchy is a pyramid of memory types with different speeds and sizes.
- Memory Bandwidth: Memory bandwidth is the rate at which data can be read from or stored into memory.
- Cache: Cache is a small memory space that stores frequently accessed data.
- HBM: HBM is a high-bandwidth memory technology that uses 3D-stacked DRAM.

## Best Practices
- Find ways to parallelize sequential code.
- Minimize data transfers between the host and the device.
- Adjust kernel launch configuration to maximize device utilization.
- Ensure that global memory accesses are coalesced.
- Minimize redundant accesses to global memory whenever possible.
- Avoid long sequences of diverged execution by threads within the same warp.
- Use specialized instructions based on the specific GPU architecture

## Example: Original Model
```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return a + b


def get_inputs():
    # randomly generate input tensors based on the model architecture
    a = torch.randn(1, 128).cuda()
    b = torch.randn(1, 128).cuda()
    return [a, b]


def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return []

```

## Example: Optimized Model with Custom CUDA
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for element-wise addition
elementwise_add_source = \"\"\"
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
\"\"\"

elementwise_add_cpp_source = (
    "torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for element-wise addition
elementwise_add = load_inline(
    name="elementwise_add",
    cpp_sources=elementwise_add_cpp_source,
    cuda_sources=elementwise_add_source,
    functions=["elementwise_add_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.elementwise_add = elementwise_add

    def forward(self, a, b):
        return self.elementwise_add.elementwise_add_cuda(a, b)

```
"""


@dataclass
class Args:
    benchmark_server_url: str = "https://tcapelle--kernel-benchmark-server-benchmarkservice-fastapi-app.modal.run/benchmark"
    debug: bool = False
    model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct"  # Use a smaller Qwen model

args = sp.parse(Args)

# Benchmark server parameters
BENCHMARK_SERVER_PARAMS = {
    "timeout": "60",  # 60 seconds timeout for benchmark
    "device": "cuda",
    "repeats": "10",  # Number of benchmark repeats
}

accelerator = accelerate.Accelerator()

# ===== Utility Functions =====

def extract_python_code(text):
    """Extract Python code from markdown code blocks or text"""
    # Look for Python code blocks
    python_blocks = re.findall(r'```(?:python|py)?\s*([\s\S]*?)```', text)
    if python_blocks:
        return python_blocks[0].strip()
        
    # If no Python blocks, extract anything that looks like Python code
    code_lines = []
    in_code = False
    for line in text.split('\n'):
        if line.strip().startswith('```'):
            in_code = not in_code
            continue
        if in_code:
            code_lines.append(line)
    
    return '\n'.join(code_lines) if code_lines else text

# Helper function to ensure content is a string
def ensure_string(content):
    """Convert content to string if it's not already"""
    if isinstance(content, list):
        return ensure_string(content[0])
    elif not isinstance(content, str):
        return str(content)
    return content

# Helper function to create a hash of content for caching
def hash_content(content):
    """Create a hash of content for use as a cache key"""
    content_str = ensure_string(content)
    return hashlib.md5(content_str.encode()).hexdigest()

# Create a cache for benchmark results to avoid duplicate calls
# Using strings as keys instead of tuples with lists
benchmark_cache = {}

# Cache for storing content based on hash
content_cache = {}

# ===== Benchmark Functions =====

def call_benchmark_server(
    ref_pytorch_code,
    optimized_code,
    benchmark_server_url=args.benchmark_server_url,
    benchmark_server_params=BENCHMARK_SERVER_PARAMS,
):
    """Call the benchmark server to evaluate the optimized code"""
    # Ensure inputs are strings
    ref_pytorch_code = ensure_string(ref_pytorch_code)
    optimized_code = ensure_string(optimized_code)
    
    # Create in-memory file objects
    ref_file = BytesIO(ref_pytorch_code.encode("utf-8"))
    kernel_file = BytesIO(optimized_code.encode("utf-8"))
    if args.debug and accelerator.is_main_process:
        print(f"Ref file: {ref_pytorch_code}")
        print(f"Kernel file: {optimized_code}")

    # Prepare the files for the request
    files = {
        "ref_file": ("ref_file.py", ref_file),
        "kernel_file": ("kernel_file.py", kernel_file),
    }

    try:
        # Make the request with both files and data
        response = requests.post(
            benchmark_server_url, files=files, data=benchmark_server_params
        )

        # Add debugging info
        if args.debug and accelerator.is_main_process:
            print("="*100)
            print(f"Status code: {response.status_code}")
            print(f"Response content: {response.content[:500]}")  # Showing first 500 chars
            print("-"*100)

        # Check for successful response before parsing JSON
        if response.status_code != 200:
            return {
                "error": f"Server error: {response.status_code}",
                "content": str(response.content),
            }

        # Try to parse JSON with better error handling
        try:
            return response.json()
        except requests.exceptions.JSONDecodeError:
            return {"error": "Invalid JSON response", "content": str(response.content)}
    except Exception as e:
        return {"error": f"Exception calling benchmark server: {str(e)}"}

# Use LRU cache with a high maxsize to store results during training
@lru_cache(maxsize=10000)
def call_benchmark_server_cached(ref_pytorch_code_hash, optimized_code_hash):
    """Cached version of call_benchmark_server to avoid duplicate calls"""
    # Look up the actual content from the content cache
    ref_pytorch_code = content_cache.get(ref_pytorch_code_hash, "")
    optimized_code = content_cache.get(optimized_code_hash, "")
    return call_benchmark_server(ref_pytorch_code, optimized_code)

def score_kernel(output, ref_code):
    """Score the generated kernel based on benchmark results"""
    # Ensure inputs are strings
    output = ensure_string(output)
    ref_code = ensure_string(ref_code)
    
    extracted_code = extract_python_code(output)
    
    # Create hashes for the content
    ref_hash = hash_content(ref_code)
    code_hash = hash_content(extracted_code)
    
    # Store the actual content in the content cache
    content_cache[ref_hash] = ref_code
    content_cache[code_hash] = extracted_code
    
    # Use the cached version of benchmark server call to avoid duplicate calls
    benchmark_result = call_benchmark_server_cached(ref_hash, code_hash)
    
    error = benchmark_result.get("error", None)
    if error is not None:
        return {
            "compiled": False,
            "correctness": False,
            "speedup_vs_compile": 0,
            "speedup_vs_eager": 0,
            "error": benchmark_result.get("content", str(error)),
        }

    # Handle missing keys safely with .get() and provide defaults
    kernel_result = benchmark_result.get("kernel_result", {})
    return {
        "compiled": kernel_result.get("compiled", False),
        "correctness": kernel_result.get("correctness", False),
        "speedup_vs_compile": benchmark_result.get("speedup_vs_compile", 0),
        "speedup_vs_eager": benchmark_result.get("speedup_vs_eager", 0),
        "error": benchmark_result.get("error", None),
    }

def get_cached_score(response, ref_code):
    """Get or compute kernel score, using a cache to avoid duplicate benchmark calls"""
    # Ensure inputs are strings for consistent hashing
    response = ensure_string(response)
    ref_code = ensure_string(ref_code)
    
    # Create a cache key based on the response and reference code - using hashes to ensure hashability
    cache_key = (hash_content(response), hash_content(ref_code))
    
    # Check if we've already computed this result
    if cache_key in benchmark_cache:
        return benchmark_cache[cache_key]
    
    # If not, compute the result and store it
    result = score_kernel(response, ref_code)
    benchmark_cache[cache_key] = result
    return result

def clear_benchmark_cache():
    """Clear the benchmark cache between batches to avoid memory leaks"""
    benchmark_cache.clear()

# ===== Reward Functions =====

def reward_compilation(completions, ref_code=None, **kwargs):
    """Reward function based on whether the generated code compiles"""
    responses = [completion[0]['content'] for completion in completions]
    
    # Extract ref_code from kwargs if needed
    ref_codes = []
    if ref_code is None and 'ref_code' in kwargs:
        ref_codes = kwargs['ref_code']
    else:
        ref_codes = [ref_code] * len(responses)
    
    rewards = []
    for response, ref in zip(responses, ref_codes):
        try:
            result = get_cached_score(response, ref)
            # Binary reward: 1.0 if compiled, -1.0 if not
            rewards.append(1.0 if result["compiled"] else -1.0)
        except Exception as e:
            if accelerator.is_main_process and args.debug:
                print(f"Error in reward_compilation: {e}")
            rewards.append(-1.0)  # Assign negative reward on error
    
    return rewards

def reward_correctness(completions, ref_code=None, **kwargs):
    """Reward function based on whether the generated code produces correct results"""
    responses = [completion[0]['content'] for completion in completions]
    
    # Extract ref_code from kwargs if needed
    ref_codes = []
    if ref_code is None and 'ref_code' in kwargs:
        ref_codes = kwargs['ref_code']
    else:
        ref_codes = [ref_code] * len(responses)
    
    rewards = []
    for response, ref in zip(responses, ref_codes):
        try:
            result = get_cached_score(response, ref)
            # Binary reward: 1.0 if correct, -1.0 if not
            rewards.append(1.0 if result["correctness"] else -1.0)
        except Exception as e:
            if accelerator.is_main_process and args.debug:
                print(f"Error in reward_correctness: {e}")
            rewards.append(-1.0)  # Assign negative reward on error
    
    return rewards

def reward_speedup(completions, ref_code=None, **kwargs):
    """Reward function based on the speedup achieved"""
    responses = [completion[0]['content'] for completion in completions]
    
    # Extract ref_code from kwargs if needed
    ref_codes = []
    if ref_code is None and 'ref_code' in kwargs:
        ref_codes = kwargs['ref_code']
    else:
        ref_codes = [ref_code] * len(responses)
    
    rewards = []
    for response, ref in zip(responses, ref_codes):
        try:
            result = get_cached_score(response, ref)
            # Reward based on speedup vs eager execution
            # Scale between -1.0 and 10.0 based on performance
            speedup = result["speedup_vs_eager"]
            if speedup <= 0:
                reward = -1.0
            else:
                # Cap the reward at 10.0 for very high speedups
                reward = min(10.0, speedup)
            rewards.append(reward)
        except Exception as e:
            if accelerator.is_main_process and args.debug:
                print(f"Error in reward_speedup: {e}")
            rewards.append(-1.0)  # Assign negative reward on error
    
    # After computing all rewards for this batch, clear the cache to free memory
    clear_benchmark_cache()
    
    return rewards

# ===== Dataset Preparation =====

def get_dataset(dataset_name="cape-team/cuda-optimized-models", split="train", max_samples=100):
    """
    Load the preprocessed dataset and format it for training.
    
    Args:
        dataset_name: The name or path to the preprocessed dataset
        split: The dataset split to use ('train' or 'test')
        max_samples: Maximum number of samples to use (None for all)
    
    Returns:
        Processed dataset ready for training
    """
    # Load the dataset - this is expected to be preprocessed by prepare_dataset.py
    data = load_dataset(dataset_name)[split]
    
    # Take a subset of the dataset for faster training/testing
    if max_samples and len(data) > max_samples:
        data = data.select(range(max_samples))
    
    def format_example(example):
        # Format the prompt with the preprocessed code
        pytorch_code = example["code"]
        
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"# CUDA Kernel Optimization Task\n\n## Your Task: Optimize This Model\n```python\n{pytorch_code}\n```\n\nImplement an optimized version called \"ModelNew\" with custom CUDA operators."}
            ],
            "ref_code": pytorch_code  # Save the reference code for the reward functions
        }
    
    # Format each example in the dataset
    formatted_data = data.map(format_example)
    
    if args.debug and accelerator.is_main_process:
        print(f"Loaded and formatted {len(formatted_data)} examples from {split} split")
        
    return formatted_data

# ===== Model and Training Setup =====

# Load the model and tokenizer
model = args.model_name
tokenizer = AutoTokenizer.from_pretrained(model)

# Prepare the dataset
dataset = get_dataset()

# Configure training arguments
training_args = GRPOConfig(
    use_vllm=False,
    model_init_kwargs={
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
        "device_map": "auto",
    },
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    beta=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_torch",
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_generations=2,
    max_prompt_length=1024,
    max_completion_length=2048,
    max_steps=100,
    save_steps=50,
    max_grad_norm=0.1,
    log_completions=True,
    report_to="wandb",
    output_dir="grpo_cuda_output",
    # Set reward weights to give more importance to speedup
    reward_weights=[0.2, 0.3, 0.5],  # Weights for compilation, correctness, speedup
)

if accelerator.is_main_process:
    wandb.init(project="grpo-cuda-optimization", config=training_args)

# Custom callback to clear benchmark cache after each batch
class BenchmarkCacheCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # Ensure we clear the cache after each step
        clear_benchmark_cache()
        # Also periodically clear the content cache to avoid memory leaks
        if state.global_step % 10 == 0:
            content_cache.clear()

# Initialize the trainer
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[reward_compilation, reward_correctness, reward_speedup],
    args=training_args,
    train_dataset=dataset,
    callbacks=[BenchmarkCacheCallback()],
)

# Start training
trainer.train()