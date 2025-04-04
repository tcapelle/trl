"""
Loads the GPUMODE/Inductor_Created_Data_Permissive dataset, performs preprocessing, 
and prepares it for use in training CUDA optimization models.

Preprocessing steps include:
1. Renaming the main nn.Module class (identified by the 'entry_point' column) to 'Model'.
2. Standardizing the 'get_init_inputs' function to return only positional arguments.
3. Truncating code exceeding a specific length.

The processed dataset is then optionally pushed to the Hugging Face Hub.
"""
import re
from datasets import load_dataset
from dataclasses import dataclass
import simple_parsing as sp

@dataclass
class Args:
    dataset_name: str = "GPUMODE/Inductor_Created_Data_Permissive"
    split: str = "train"
    max_samples: int = None  # Set to None to use all samples
    output_dataset_name: str = "tcapelle/cuda-optimized-models"  # Change this to your HF username
    debug: bool = False
    push_to_hub: bool = True
    max_code_length: int = 4000 # Max length for code snippets before truncation

args = sp.parse(Args)

def preprocess_sample(code, entry_point_name):
    """Preprocess sample to rename the custom nn.Module to 'Model'.
    
    This is required because the benchmark server expects the module to be called 'Model'
    regardless of what the original name was.
    """
    original_class_name = entry_point_name
    
    # Check if the entry point is actually a class definition before proceeding
    # Simple check: look for "class OriginalClassName(" in the code
    if f"class {original_class_name}(" not in code:
        if args.debug:
            print(f"Entry point '{original_class_name}' not found as a class definition in code. Skipping renaming.")
        return code

    if original_class_name == "Model":
        # Already named Model, no need to rename
        if args.debug:
            print("Class is already named 'Model', no renaming needed")
        return code
    
    if args.debug:
        print(f"Renaming nn.Module class from '{original_class_name}' to 'Model'")
    
    # Replace the class name in the class definition
    # Use a more specific regex to avoid accidental replacements
    renamed_code = re.sub(
        f'class\s+{original_class_name}\s*\(', 
        'class Model(', 
        code,
        count=1 # Only replace the class definition itself
    )
    
    # Replace constructor calls: OriginalClass() -> Model()
    # Use word boundary to avoid replacing parts of other names
    renamed_code = re.sub(
        f'\b{original_class_name}\s*\(', 
        'Model(', 
        renamed_code
    )
    
    # Replace other references to the class (e.g., type hints, super calls)
    # Use word boundary to avoid replacing parts of other names
    renamed_code = re.sub(
        f'\b{original_class_name}\b', 
        'Model', 
        renamed_code
    )
    
    return renamed_code

def standardize_get_init_inputs(code):
    """Standardize get_init_inputs function to return only positional arguments.
    
    The dataset contains functions like:
    def get_init_inputs():
        return [[], {'key': value}]
    
    We want to convert them to:
    def get_init_inputs():
        return [value]
    
    This is required because the benchmark server expects only positional arguments.
    """
    # First check if the function exists
    if "def get_init_inputs" not in code:
        return code
    
    # Extract the function line by line
    lines = code.split('\n')
    func_start_idx = -1
    func_end_idx = -1
    return_line_idx = -1
    
    # Find the function boundaries
    for i, line in enumerate(lines):
        if "def get_init_inputs" in line:
            func_start_idx = i
        if func_start_idx != -1 and "return" in line:
            return_line_idx = i
            
            # Find the end of the function (next line with same or less indentation)
            indent = len(line) - len(line.lstrip())
            for j in range(return_line_idx + 1, len(lines)):
                if j >= len(lines) or (lines[j].strip() and len(lines[j]) - len(lines[j].lstrip()) <= indent):
                    func_end_idx = j - 1
                    break
            else:
                func_end_idx = len(lines) - 1
            
            break
    
    if func_start_idx == -1 or return_line_idx == -1:
        return code
    
    # Extract the return statement
    return_line = lines[return_line_idx]
    
    # Find what's inside the return statement
    return_match = re.search(r'return\s*(.*)', return_line)
    if not return_match:
        return code
    
    return_value = return_match.group(1).strip()
    
    # Try to parse the return value - looking for [[], {key: value}] pattern
    kwargs_match = re.search(r'\[\s*\[\s*\]\s*,\s*\{(.*?)\}\s*\]', return_value)
    
    if kwargs_match:
        # Extract the kwargs content
        kwargs_content = kwargs_match.group(1).strip()
        
        # If the dictionary is empty, return an empty list
        if not kwargs_content:
            new_return = "return []"
            lines[return_line_idx] = re.sub(r'return\s*.*', new_return, return_line)
            
            if args.debug:
                print(f"Empty dictionary found, returning empty list:\nOriginal: {return_line.strip()}\nNew: {lines[return_line_idx].strip()}")
            
            return '\n'.join(lines)
        
        # Find all the values from key-value pairs
        values = []
        kv_pattern = r'[\'\"](\w+)[\'\"]\s*:\s*(\d+(?:\.\d+)?|True|False)'
        
        for match in re.finditer(kv_pattern, kwargs_content):
            values.append(match.group(2))
        
        # If we found values, create a new return
        if values:
            new_return = f"return [{', '.join(values)}]"
        else:
            # Return empty list if no values found
            new_return = "return []"
        
        # Replace the return line
        lines[return_line_idx] = re.sub(r'return\s*.*', new_return, return_line)
        
        if args.debug:
            print(f"Standardized get_init_inputs:\nOriginal: {return_line.strip()}\nNew: {lines[return_line_idx].strip()}")
        
        # Join the lines back together
        return '\n'.join(lines)
    else:
        # Keep the original return statement if we don't match the expected pattern
        if args.debug:
            print(f"Using original return for get_init_inputs: {return_line.strip()}")
        
        return code
    
    return code

def prepare_dataset():
    """
    Load, preprocess and prepare the dataset for training.
    
    This function will:
    1. Load the dataset from Hugging Face
    2. Preprocess each example to ensure the model class is named 'Model'
    3. Standardize the get_init_inputs function
    4. Return a dataset with only the processed code
    
    NOTE: The prompt formatting is left to the training script, which will add:
    - System prompt with CUDA optimization instructions
    - User prompt with the task description
    
    Returns:
        Dataset: A HuggingFace dataset with a 'code' field containing preprocessed PyTorch code
    """
    # Load the dataset
    print(f"Loading dataset {args.dataset_name} ({args.split} split)")
    
    data = load_dataset(args.dataset_name)[args.split]
    
    # Take a subset of the dataset for faster training/testing
    if args.max_samples and len(data) > args.max_samples:
        data = data.select(range(args.max_samples))
        print(f"Selected {args.max_samples} samples from dataset")
    
    def process_example(example):
        # Extract the PyTorch model code
        pytorch_code = example["python_code"]
        entry_point = example["entry_point"]
        
        # Preprocess to ensure the model is called "Model" 
        pytorch_code = preprocess_sample(pytorch_code, entry_point)
        
        # Standardize get_init_inputs function
        pytorch_code = standardize_get_init_inputs(pytorch_code)
        
        # Truncate the code if it's too long to save memory
        if args.max_code_length is not None and len(pytorch_code) > args.max_code_length:
            pytorch_code = pytorch_code[:args.max_code_length] + f"\n# ... truncated (>{args.max_code_length} chars) for memory efficiency"
        
        # Return only the preprocessed code, without formatting the prompt
        # Prompt formatting will be handled by the training script
        return {
            "pytorch_code": pytorch_code  # This pytorch_code is already preprocessed
        }
    
    # Process each example in the dataset
    processed_data = data.map(process_example)
    
    print(f"Preprocessed {len(processed_data)} examples from {args.split} split")
        
    return processed_data

def main():
    """
    Main function to prepare the dataset and optionally push it to the Hugging Face Hub.
    
    The training script will need to modify its get_dataset function to:
    1. Load this processed dataset
    2. Format the prompts using the preprocessed code in the 'code' field
    """
    # Process the dataset
    processed_dataset = prepare_dataset()
    
    # Push the dataset to the Hugging Face Hub if requested
    if args.push_to_hub:
        print(f"Pushing dataset to Hugging Face Hub as {args.output_dataset_name}")
        processed_dataset.push_to_hub(args.output_dataset_name)
        print(f"Successfully pushed dataset to {args.output_dataset_name}")
    else:
        print("Skipping push to Hugging Face Hub")
    
    # Print a summary of what was done
    print("\n=== Dataset Processing Summary ===")
    print(f"Original dataset: {args.dataset_name}")
    print(f"Number of samples processed: {len(processed_dataset)}")
    print(f"Preprocessing steps applied:")
    print(f"  - Renamed nn.Module classes to 'Model'")
    print(f"  - Standardized get_init_inputs functions to return positional args")
    if args.max_code_length is not None:
        print(f"  - Truncated code longer than {args.max_code_length} characters")
    
    if args.push_to_hub:
        print(f"\nDataset is now available at: https://huggingface.co/datasets/{args.output_dataset_name}")
    else:
        print(f"\nTo use this dataset with the training script, run:")
        print(f"  python prepare_dataset.py --output_dataset_name your-username/cuda-optimized-models --push_to_hub True")
    
    return processed_dataset

if __name__ == "__main__":
    main()
