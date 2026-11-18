from datasets import Dataset
from peft import LoraConfig
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer

objects = ["a red circle", "a brown dog", "a frog"]
prompts = [
    [{"role": "user", "content": f"Draw {o}. Reply with only SVG code"}]
    for o in objects
] * 50
dataset = Dataset.from_dict({"prompt": prompts})

def svg_tag_reward(completions, **kwargs):
    scores = []
    for c in completions:
        text = c[0]["content"]
        score = 0.0
        if "<svg" in text:
            score += 0.5
        if "</svg>" in text:
            score += 0.5
        scores.append(score)
    return scores

args = GRPOConfig(
    output_dir="runs/test",
    per_device_train_batch_size=4,
    num_generations=4,      # 4 answers per prompt
    max_completion_length=512,
    max_steps=30,
    logging_steps=1,
    learning_rate=1e-4,
    save_strategy="no",
    report_to="none",
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=svg_tag_reward,
    args=args,
    train_dataset=dataset,
    peft_config=LoraConfig(r=16, lora_alpha=32, task_type="CAUSAL_LM"),
)
trainer.train()