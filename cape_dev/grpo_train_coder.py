import asyncio
import torch
import requests
from io import BytesIO
import re
from dataclasses import dataclass
from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
import wandb
import weave
import accelerate
import simple_parsing as sp
from datasets import load_dataset
from functools import lru_cache
import random
import litellm
from pydantic import BaseModel, Field

from utils import ensure_string, hash_content, serialize_for_wandb, extract_python_code

# Set litellm logging
litellm.set_verbose = False


# ===== Configuration =====

SYSTEM_PROMPT = """
# CUDA Kernel Optimization Task
You are an expert in PyTorch and CUDA programming. 

## Objective
Your task is to optimize PyTorch models by replacing standard PyTorch operators with custom CUDA kernels. You should:
- Choose which operators to replace with custom implementations
- Consider operator fusion opportunities (e.g., combining matmul+relu)
- Explore algorithmic optimizations (e.g., online softmax)
- Rename your optimized implementation as "ModelNew"
- Reply with a short explanation and the optimized code, nothing else

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
    a = torch.randn(1, 128).cuda()
    b = torch.randn(1, 128).cuda()
    return [a, b]


def get_init_inputs():
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

# LLM evaluation prompt for soft-scoring
LLM_EVAL_SYSTEM_PROMPT = """
You are an expert in CUDA kernel optimization and GPU programming. Your task is to evaluate a given CUDA kernel implementation compared to a reference PyTorch implementation without running the code. Focus your evaluation on:

## 1. Correctness:
- Will the CUDA kernel produce exactly the same outputs as the reference PyTorch implementation?
- Identify any logical errors, indexing issues, off-by-one errors, or data race conditions.

## 2. Code Quality:
- Is the CUDA kernel code clean, readable, maintainable, and well-documented?
- Does the code follow GPU programming best practices (e.g., avoiding divergence, coalesced memory accesses, efficient shared memory use)?
- Is the kernel launch configuration (block/grid sizes) appropriate for good GPU utilization?
- Are there optimization opportunities missed (e.g., operator fusion, reduced memory traffic, improved memory hierarchy usage)?

## Task Context:
You are optimizing PyTorch models by replacing standard operators with custom CUDA kernels. The optimization goals include:
- Operator replacement with efficient CUDA implementations.
- Operator fusion opportunities (e.g., combining operations such as matmul + activation).
- Algorithmic optimizations (e.g., online softmax computation).
- Renaming the optimized implementation to "ModelNew".

## Key GPU Programming Concepts:
- **Thread**: A single execution unit executing one instruction at a time.
- **Thread Block**: A group of threads cooperating via shared memory.
- **Warp**: Group of threads executing concurrently, sharing the instruction stream.
- **Shared Memory**: Fast memory accessible by threads in the same block.
- **Registers**: Fastest memory private to individual threads.
- **Memory Hierarchy**: Arrangement of memory types (register, shared, global memory) with varying speeds and capacities.
- **Memory Bandwidth**: Rate of data transfer to/from GPU memory.
- **Memory Coalescing**: Optimizing memory accesses by aligning them across threads.
- **Cache and HBM**: Caches store frequently used data; High-Bandwidth Memory (HBM) offers rapid data access to the GPU.

## CUDA Kernel Best Practices Checklist:
- Parallelize sequential operations effectively.
- Minimize data transfers between host and device.
- Optimize kernel launch parameters (blocks/grid size) for GPU architecture.
- Ensure coalesced global memory accesses.
- Reduce redundant global memory accesses.
- Avoid warp divergence.
- Leverage specialized GPU instructions (e.g., tensor cores) when beneficial.


## Example:

### Reference PyTorch Implementation:
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

### Generated CUDA kernel:
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

### Response

- "analysis": "Kernel correctly implements element-wise addition. Quality is good, but improved error handling and documentation are recommended."
- "correctness": 0.9
- "code_quality": 0.8

## Format:
Reply in JSON format with the following fields:
- "analysis": "Detailed analysis of the generated CUDA kernel."
- "correctness": "Correctness score from 0.0 to 1.0, where 1.0 is perfectly correct"
- "code_quality": "Code quality score from 0.0 to 1.0, where 1.0 is excellent quality"

"""

class LLMKernelEvaluation(BaseModel):
    """The LLM evaluation response for the Cuda Kernel"""
    analysis: str = Field(description="Detailed analysis of the generated CUDA kernel.")
    correctness: float = Field(description="Correctness score from 0.0 to 1.0, where 1.0 is perfectly correct")
    code_quality: float = Field(description="Code quality score from 0.0 to 1.0, where 1.0 is excellent quality")


@dataclass
class Args:
    benchmark_server_url: str = "https://tcapelle--kernel-benchmark-server-benchmarkservice-fastapi-app.modal.run/benchmark"
    debug: bool = False
    model_name: str = "Qwen/Qwen2.5-Coder-14B-Instruct"  # Use a smaller Qwen model
    dataset_name: str = "tcapelle/cuda-optimized-models"#"GPUMODE/Inductor_Created_Data_Permissive"
    code_column: str = "pytorch_code"
    max_samples: int = None # debug parameter
    hard_score_percentage: float = 0.1  # Percentage of samples to use hard scoring (benchmark server)
    llm_evaluator_model: str = "gpt-4o"  # Model to use for soft scoring
    llm_reward_weight: float = 0.0  # Weight for the LLM-based reward relative to benchmark rewards

args = sp.parse(Args)

# Benchmark server parameters
BENCHMARK_SERVER_PARAMS = {
    "timeout": "60",  # 60 seconds timeout for benchmark
    "device": "cuda",
    "repeats": "10",  # Number of benchmark repeats
}

accelerator = accelerate.Accelerator()



# ===== Caching Strategy =====
# The caching mechanism uses two levels:
# 1. content_cache: A dictionary that maps content hashes to the actual content
#    - Keys: MD5 hashes of the content (ref code or optimized code)
#    - Values: The actual content as strings
#
# 2. benchmark_cache: A dictionary that maps (extracted_code_hash, ref_code_hash) tuples to benchmark results
#    - This avoids redundant benchmark server calls for the same code
#
# 3. call_benchmark_server_cached: An LRU-cached function that uses content hashes to retrieve 
#    the actual content from content_cache and make the benchmark server call
#
# This approach ensures we don't duplicate benchmark calls, which are expensive,
# while also keeping memory usage reasonable by storing just one copy of each content.



# Create a cache for benchmark results to avoid duplicate calls
# Using strings as keys instead of tuples with lists
benchmark_cache = {}

# Cache for storing content based on hash
content_cache = {}

# ===== Benchmark Functions =====
@weave.op
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

# LLM-based evaluation function
@weave.op
async def evaluate_with_llm(extracted_code, ref_code, model=args.llm_evaluator_model):
    """Use an LLM to evaluate the quality of the optimized code"""
    # Ensure inputs are strings
    ref_code = ensure_string(ref_code)
    extracted_code = ensure_string(extracted_code)
    
    # Prepare the prompt
    prompt = f"""Analyse the following code and provide your evaluation in JSON format.

    ## Reference PyTorch Implementation:
    ```python
    {ref_code}
    ```

    ## Generated CUDA kernel:
    ```python
    {extracted_code}
    """

    try:
        # Call the LLM with structured output format
        response = await litellm.acompletion(
            model=model,
            messages=[
                {"role": "system", "content": LLM_EVAL_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
                ],
            response_format=LLMKernelEvaluation,
            temperature=0.0,
        )
        
        # Parse the response using Pydantic
        content = response.choices[0].message.content
        evaluation = LLMKernelEvaluation.model_validate_json(content)
        
        # Convert to result format
        result = {
            "analysis": evaluation.analysis,
            "correctness": evaluation.correctness,
            "code_quality": evaluation.code_quality,
        }
        
        return result
    except Exception as e:
        if accelerator.is_main_process and args.debug:
            print(f"Error in LLM evaluation: {e}")
        return {
            "analysis": f"Error in evaluation: {str(e)}",
            "correctness": None,
            "code_quality": None,
        }

# Use LRU cache with a high maxsize to store results during training
@lru_cache(maxsize=10000)
def call_benchmark_server_cached(ref_pytorch_code_hash, optimized_code_hash):
    """Cached version of call_benchmark_server to avoid duplicate calls"""
    # Look up the actual content from the content cache
    ref_pytorch_code = content_cache.get(ref_pytorch_code_hash, "")
    optimized_code = content_cache.get(optimized_code_hash, "")

    return call_benchmark_server(ref_pytorch_code, optimized_code)

@weave.op
def score_kernel(extracted_code, ref_code):
    """Score the generated kernel based on benchmark results"""
    # Ensure inputs are strings
    extracted_code = ensure_string(extracted_code)
    ref_code = ensure_string(ref_code)
    
    # Create hashes for the content
    ref_hash = hash_content(ref_code)
    code_hash = hash_content(extracted_code)
    
    # Store the actual content in the content cache
    content_cache[ref_hash] = ref_code
    content_cache[code_hash] = extracted_code
    
    # Decide whether to use hard scoring (benchmark) based on percentage
    use_hard_scoring = random.random() < args.hard_score_percentage
    
    # Initialize result with None values
    result = {
        "compiled": False,
        "correctness": False,
        "speedup_vs_compile": 0,
        "speedup_vs_eager": 0,
        "error": None,
        "is_hard_score": False
    }
    
    if use_hard_scoring:
        # Use the cached version of benchmark server call
        benchmark_result = call_benchmark_server_cached(ref_hash, code_hash)
        
        error = benchmark_result.get("error", None)
        if error is not None:
            result.update({
                "error": benchmark_result.get("content", str(error)),
                "is_hard_score": True
            })
        else:
            # Handle missing keys safely with .get() and provide defaults
            kernel_result = benchmark_result.get("kernel_result", {})
            result.update({
                "compiled": kernel_result.get("compiled", False),
                "correctness": kernel_result.get("correctness", False),
                "speedup_vs_compile": benchmark_result.get("speedup_vs_compile", 0),
                "speedup_vs_eager": benchmark_result.get("speedup_vs_eager", 0),
                "error": benchmark_result.get("error", None),
                "is_hard_score": True
            })
    
    return result

def get_cached_score(response, ref_code):
    """Get or compute kernel score, using a cache to avoid duplicate benchmark calls"""
    # Ensure inputs are strings for consistent hashing
    response = ensure_string(response)
    ref_code = ensure_string(ref_code)
    
    # Extract Python code from the response
    extracted_code = extract_python_code(response)
    
    # Create a cache key based on the extracted code and reference code
    cache_key = (hash_content(extracted_code), hash_content(ref_code))
    
    # Check if we've already computed this result
    if cache_key in benchmark_cache:
        return benchmark_cache[cache_key]
    
    # If not, compute the result and store it
    result = score_kernel(extracted_code, ref_code)
    benchmark_cache[cache_key] = result
    return result

def clear_caches():
    """Clear benchmark cache between batches to avoid memory leaks"""
    # Clear the benchmark results cache
    benchmark_cache.clear()
    
    # We don't clear content_cache here as it's more efficient to keep content 
    # cached across batches, but we monitor its size in the callback
    if accelerator.is_main_process and args.debug and len(content_cache) > 0:
        print(f"Content cache size: {len(content_cache)} entries")
        
# Custom callback to clear caches after each batch
class CacheCallback(TrainerCallback):
    def __init__(self, content_cache_max_size=5000):
        self.content_cache_max_size = content_cache_max_size
    
    def on_step_end(self, args, state, control, **kwargs):
        # Ensure we clear benchmark cache after each step
        clear_caches()
        
        # Only clear content cache if it grows too large
        if len(content_cache) > self.content_cache_max_size:
            if accelerator.is_main_process and args.debug:
                print(f"Clearing content cache (size: {len(content_cache)})")
            content_cache.clear()
            
            # Also clear the LRU cache for call_benchmark_server_cached
            call_benchmark_server_cached.cache_clear()

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
            # Use get_cached_score which handles the extraction internally
            result = get_cached_score(response, ref)
            
            # Only use rewards from hard scoring, otherwise return None
            if result.get("is_hard_score", False):
                rewards.append(1.0 if result["compiled"] else -1.0)
            else:
                rewards.append(None)  # Skip this reward for samples without hard scoring
        except Exception as e:
            if accelerator.is_main_process and args.debug:
                print(f"Error in reward_compilation: {e}")
            rewards.append(None)  # Return None on error instead of negative reward
    
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
            # Use get_cached_score which handles the extraction internally
            result = get_cached_score(response, ref)
            
            # Only use rewards from hard scoring, otherwise return None
            if result.get("is_hard_score", False):
                rewards.append(1.0 if result["correctness"] else -1.0)
            else:
                rewards.append(None)  # Skip this reward for samples without hard scoring
        except Exception as e:
            if accelerator.is_main_process and args.debug:
                print(f"Error in reward_correctness: {e}")
            rewards.append(None)  # Return None on error instead of negative reward
    
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
            # Use get_cached_score which handles the extraction internally
            result = get_cached_score(response, ref)
            
            # Only use rewards from hard scoring, otherwise return None
            if result.get("is_hard_score", False):
                speedup = result["speedup_vs_eager"]
                if speedup <= 0:
                    reward = -1.0
                else:
                    # Cap the reward at 10.0 for very high speedups
                    reward = min(10.0, speedup)
                rewards.append(reward)
            else:
                rewards.append(None)  # Skip this reward for samples without hard scoring
        except Exception as e:
            if accelerator.is_main_process and args.debug:
                print(f"Error in reward_speedup: {e}")
            rewards.append(None)  # Return None on error instead of negative reward
    
    return rewards

def reward_llm_scoring(completions, ref_code, **kwargs):
    """LLM-based reward function that evaluates code quality and correctness"""
    responses = [completion[0]['content'] for completion in completions]
    
    # Extract ref_code from kwargs if needed
    ref_codes = []
    if ref_code is None and 'ref_code' in kwargs:
        ref_codes = kwargs['ref_code']
    else:
        ref_codes = [ref_code] * len(responses)

    @weave.op
    async def get_llm_eval(response, ref):
        try:
            # Use get_cached_llm_eval which handles the extraction internally
            
            result = await evaluate_with_llm(response, ref)
            # Compute a combined score based on correctness and code quality
            correctness_score = result["correctness"]  
            quality_score = result["code_quality"]
            
            # Combined reward with more weight on correctness
            reward = correctness_score + quality_score
            
            return reward
        except Exception as e:
            if accelerator.is_main_process and args.debug:
                print(f"Error in reward_llm_scoring: {e}")
            return None  # Return None on error instead of negative reward

    # Get an event loop to run the coroutine
    loop = asyncio.get_event_loop()
    # Run the coroutine and get the results
    tasks = [get_llm_eval(response, ref) for response, ref in zip(responses, ref_codes)]
    rewards = loop.run_until_complete(asyncio.gather(*tasks))

    return rewards

# ===== Dataset Preparation =====

def get_dataset(dataset_name=args.dataset_name, split="train", max_samples=args.max_samples):
    # Load the dataset - this is expected to be preprocessed by prepare_dataset.py
    data = load_dataset(dataset_name)[split]
    
    # Take a subset of the dataset for faster training/testing
    if max_samples and len(data) > max_samples:
        data = data.select(range(max_samples))
    
    def format_example(example):
        # Format the prompt with the preprocessed code
        pytorch_code = example[args.code_column]
        
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Here is the code to optimize:\n```python\n{pytorch_code}\n```\n\nImplement an optimized version called \"ModelNew\" with custom CUDA operators."}
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
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

# Prepare the dataset
dataset = get_dataset()

# Configure training arguments
training_args = GRPOConfig(
    use_vllm=True,
    model_init_kwargs={
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
        # "device_map": "auto",
    },
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    beta=0,
    lr_scheduler_type="cosine",
    optim="adamw_torch",
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_generations=7,
    max_prompt_length=1024,
    max_completion_length=1024,
    # max_steps=100,
    num_train_epochs=1,
    save_steps=50,
    max_grad_norm=0.1,
    log_completions=True,
    report_to="wandb",
    output_dir="grpo_cuda_output",
    reward_weights=[0.2, 0.3, 0.5, args.llm_reward_weight],  # Weights for compilation, correctness, speedup, llm_scoring
)

if accelerator.is_main_process:
    try:
        # Use a custom serializer to handle non-serializable objects
        wandb_config = serialize_for_wandb(training_args)
        weave.init("grpo-cuda-optimization")
        wandb.init(project="grpo-cuda-optimization", name="qwen-coder", config=wandb_config)
    except Exception as e:
        print(f"Failed to initialize wandb: {e}")
        print("Continuing without wandb logging...")
        # If wandb initialization fails, disable it in the training args
        training_args.report_to = []

# Initialize the trainer
trainer = GRPOTrainer(
    model=args.model_name,
    processing_class=tokenizer,
    reward_funcs=[reward_compilation, reward_correctness, reward_speedup, reward_llm_scoring],
    args=training_args,
    train_dataset=dataset,
    callbacks=[CacheCallback()],
)

# Start training
trainer.train()