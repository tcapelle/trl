import re
from datasets import load_dataset
import sys

# Copy the function from prepare_dataset.py to avoid import conflicts
def standardize_get_init_inputs(code, debug=True):
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
        kwargs_content = kwargs_match.group(1)
        
        # Find all the values from key-value pairs
        values = []
        kv_pattern = r'[\'\"](\w+)[\'\"]\s*:\s*(\d+(?:\.\d+)?|True|False)'
        
        for match in re.finditer(kv_pattern, kwargs_content):
            values.append(match.group(2))
        
        # If we found values, create a new return
        if values:
            new_return = f"return [{', '.join(values)}]"
        else:
            # Default value if no values found
            new_return = "return [4]"
        
        # Replace the return line
        lines[return_line_idx] = re.sub(r'return\s*.*', new_return, return_line)
        
        if debug:
            print(f"Standardized get_init_inputs:\nOriginal: {return_line.strip()}\nNew: {lines[return_line_idx].strip()}")
        
        # Join the lines back together
        return '\n'.join(lines)
    else:
        # Use default value if we couldn't find the expected pattern
        new_return = "return [4]"
        lines[return_line_idx] = re.sub(r'return\s*.*', new_return, return_line)
        
        if debug:
            print(f"Using default value for get_init_inputs:\nOriginal: {return_line.strip()}\nNew: {lines[return_line_idx].strip()}")
        
        # Join the lines back together
        return '\n'.join(lines)
    
    return code

# Load samples from the dataset
num_samples = 5
if len(sys.argv) > 1:
    try:
        num_samples = int(sys.argv[1])
    except ValueError:
        print(f"Invalid number of samples: {sys.argv[1]}, using default (5)")

dataset = load_dataset("GPUMODE/Inductor_Created_Data_Permissive", split="train")
samples = dataset.select(range(num_samples))

print(f"Testing standardize_get_init_inputs on {num_samples} samples:\n")

for i, sample in enumerate(samples):
    code = sample["python_code"]
    print(f"=== EXAMPLE {i+1} ===")
    
    # Find the get_init_inputs function in the original code
    def extract_get_init_inputs(code):
        # Simple extraction based on indentation
        lines = code.split("\n")
        start_idx = -1
        end_idx = -1
        
        for idx, line in enumerate(lines):
            if "def get_init_inputs" in line:
                start_idx = idx
            if start_idx != -1 and idx > start_idx:
                if "return" in line:
                    end_idx = idx
                    break
        
        if start_idx != -1 and end_idx != -1:
            return "\n".join(lines[start_idx:end_idx+1])
        return None
    
    original_func = extract_get_init_inputs(code)
    if original_func:
        print("Original:")
        print(original_func)
    else:
        print("No get_init_inputs function found")
        continue
    
    # Apply the standardization
    standardized_code = standardize_get_init_inputs(code)
    
    # Extract the standardized function
    standardized_func = extract_get_init_inputs(standardized_code)
    if standardized_func:
        print("\nStandardized:")
        print(standardized_func)
    else:
        print("\nStandardization failed")
    
    print("\n" + "-"*50) 