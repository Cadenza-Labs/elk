import random

from huggingface_hub import HfFolder

from datasets import ClassLabel, Dataset, Features, Value, load_dataset

# Load the dataset from Hugging Face
dataset = load_dataset("EleutherAI/truthful_qa_binary")

# Check what splits are available and adjust accordingly
if "train" in dataset:
    data_split = dataset["train"]
else:
    # Assuming there is a single dataset split, typically 'train'
    data_split = dataset["validation"]  # or dataset['test'] or simply dataset

# Define the features with ClassLabel
features = Features(
    {
        "question": Value("string"),
        "choice": Value("string"),
        "label": ClassLabel(names=["False", "True"]),
    }
)


# Function to transform the dataset
def transform_example(example):
    # Randomly select one of the two choices
    choice_index = random.randint(0, 1)
    choice = example["choices"][choice_index]

    label = False
    if choice_index == 0 and example["label"] == 0:
        label = True
    elif choice_index == 1 and example["label"] == 1:
        label = True
    elif choice_index == 0 and example["label"] == 1:
        label = False
    elif choice_index == 1 and example["label"] == 0:
        label = False

    return {"question": example["question"], "choice": choice, "label": label}


# Transform the dataset
transformed_examples = data_split.map(
    transform_example, remove_columns=["choices", "label"]
)

# Apply features schema
transformed_dataset = Dataset.from_dict(
    transformed_examples.to_dict(), features=features
)

# Split the dataset into training and testing
split_dataset = transformed_dataset.train_test_split(test_size=0.2)

# Authenticate with Hugging Face
user_token = HfFolder.get_token()
if user_token is None:
    raise ValueError(
        "Hugging Face token not found. Please login using `huggingface-cli login`."
    )

# Repository name on Hugging Face Hub
repo_name = "truthful_qa"

# Push to hub with split configuration
split_dataset.push_to_hub(repo_name)
