import pandas as pd
from huggingface_hub import HfFolder

from datasets import ClassLabel, Dataset, Features, Value

# Load your data
data_path = "./claims.csv"
claims_data = pd.read_csv(data_path)

# Define the features with ClassLabel
features = Features(
    {"statement": Value("string"), "label": ClassLabel(names=["False", "True"])}
)

# Convert pandas DataFrame into Hugging Face dataset
hf_dataset = Dataset.from_pandas(claims_data, features=features)

# Use datasets library to split the dataset
train_test_ratio = 0.8
train_dataset = hf_dataset.train_test_split(train_size=train_test_ratio, seed=42)[
    "train"
]
test_dataset = hf_dataset.train_test_split(train_size=train_test_ratio, seed=42)["test"]

# Authenticate with Hugging Face
user_token = HfFolder.get_token()
if user_token is None:
    raise ValueError(
        "Hugging Face token not found. Please login using `huggingface-cli login`."
    )

# Push to hub
repo_name = "claims"
train_dataset.push_to_hub(repo_name, split="train")
test_dataset.push_to_hub(repo_name, split="test")
