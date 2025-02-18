import matplotlib.pyplot as plt
from datasets import load_dataset

dataset = load_dataset("yeates/PromptfixData", split="train", streaming=True)

num_points = 5
data_points = dataset.take(num_points)

fig, axes = plt.subplots(num_points, 2, figsize=(14, 6 * num_points))

for i, data in enumerate(data_points):
    input_img = data["input_img"]
    processed_img = data["processed_img"]
    instruction = data["instruction"]
    auxiliary_prompt = data["auxiliary_prompt"]
    task_id = data["task_id"]

    axes[i, 0].imshow(input_img)
    axes[i, 0].set_title(
        f"Original Image\n\nInstruction: {instruction}\nTask ID: {task_id}",
        fontsize=12,
        loc="left"
    )
    axes[i, 0].axis("off")

    axes[i, 1].imshow(processed_img)
    axes[i, 1].set_title("Processed Image (After Instruction)", fontsize=12)
    axes[i, 1].axis("off")

plt.tight_layout()
plt.show()
