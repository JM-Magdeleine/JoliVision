import json
import random
import os

DATASET_DIR = "/data3/jmarie/JoliVision/LLaVA-CC3M-Pretrain-595K"

with open(os.path.join(DATASET_DIR, "chat-copy.json")) as dataset_file:
    dataset = json.load(dataset_file)

train_idx = random.sample(range(len(dataset)), int(len(dataset)*0.08))
test_idx = random.sample(range(len(dataset)), int(len(dataset)*0.02))

overlaps = [idx for idx in train_idx if idx in test_idx]
print(overlaps, len(overlaps))

train_idx = [idx for idx in train_idx if idx not in overlaps] # only remove training data points, as one less data point will penalize the training diversity least

print(len(train_idx), len(test_idx))

with open(os.path.join(DATASET_DIR, "mini-llava-train.json"), "w") as mini_train_file:
    json.dump([dataset[idx] for idx in train_idx], mini_train_file)

with open(os.path.join(DATASET_DIR, "mini-llava-test.json"), "w") as mini_test_file:
    json.dump([dataset[idx] for idx in test_idx], mini_test_file)
