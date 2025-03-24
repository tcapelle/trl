# train_grpo.py
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import wandb, weave


dataset = load_dataset("trl-lib/tldr", split="train")

# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

# MODEL = "Qwen/Qwen2-0.5B-Instruct"
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

training_args = GRPOConfig(
    output_dir=f"{MODEL}-GRPO", 
    num_train_epochs=1,
    num_generations=2,
    max_completion_length=40,
    log_completions=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=2,
    logging_steps=1)


wandb.init(project="grpo_example", config=training_args)

weave.init("grpo_example")


trainer = GRPOTrainer(
    model=MODEL,
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()