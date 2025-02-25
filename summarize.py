from datasets import load_dataset, DatasetDict, Dataset
from transformers import pipeline
from tqdm import tqdm

# Step 1: Load the MNLI dataset
mnli = load_dataset("multi_nli")

# Step 2: Load the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device='mps')

# Step 3: Filter the dataset for the "slate" genre with high premise-to-hypothesis ratio
def filter_slate_high_ratio(example, threshold=2.1):
    premise_length = len(example["premise"])
    hypothesis_length = len(example["hypothesis"])
    ratio = premise_length / hypothesis_length
    return example["genre"] == "slate" and ratio > threshold and hypothesis_length > 50
    # return example["genre"] == "slate" and hypothesis_length > 300

filtered_train = mnli["train"].filter(filter_slate_high_ratio)
filtered_val = mnli["validation_matched"].filter(filter_slate_high_ratio)

# Step 4: Define batch summarization function
def batch_summarize(dataset,
                    batch_size=64,
                    factor=4.7):
    """Summarize hypotheses in batches."""
    processed_examples = []

    for start_idx in tqdm(range(0, len(dataset), batch_size), desc="Processing batches", unit="batch"):
        # Prepare the batch
        batch = dataset[start_idx: start_idx + batch_size]
        hypotheses = batch['hypothesis']

        # Compute max_length and min_length for each hypothesis
        max_lengths = [int(len(hypothesis) / factor) for hypothesis in hypotheses]
        min_lengths = [max(10, int(max_len * 0.5)) for max_len in max_lengths]

        # make sure all max lengths are not longer than the input length
        max_lengths = [min(max_len, len(hypothesis)) for max_len, hypothesis in zip(max_lengths, hypotheses)]
        min_lengths = [min(min_len, max_len) for min_len, max_len in zip(min_lengths, max_lengths)]
        

        # Perform summarization for the batch
        summaries = summarizer(
            hypotheses,
            max_length=max_lengths[0],  # All elements in batch share the same lengths
            min_length=min_lengths[0],
            do_sample=False,
            batch_size=batch_size,
        )

        # Update the batch with summaries
        for idx, summary in enumerate(summaries):
            batch["hypothesis"][idx] = summary["summary_text"]

        # Add processed examples to the list
        processed_examples.extend([batch])

    # return dataset.from_list(processed_examples)
    return Dataset.from_dict({key: [item for batch in processed_examples for item in batch[key]] for key in batch})

# Step 5: Process filtered datasets using batch summarization
print("Processing the training dataset...")
processed_train = batch_summarize(filtered_train)

print("\nProcessing the validation_matched dataset...")
processed_val = batch_summarize(filtered_val)

# Step 6: Merge processed and unprocessed datasets
def merge_processed_with_original(original_dataset, processed_dataset, genre="slate"):
    """Merge processed data back with the original dataset."""

    processed_dataset = processed_dataset.add_column("processed", [True] * len(processed_dataset))
    original_dataset = original_dataset.add_column("processed", [False] * len(original_dataset))
    
    processed_dict = {
        (example["promptID"], example["pairID"]): example
        for example in processed_dataset
    }

    # Merge processed examples into the original dataset
    merged_dataset = [
        processed_dict[(example["promptID"], example["pairID"])] 
        if (example["promptID"], example["pairID"]) in processed_dict 
        else example
        for example in original_dataset
    ]

    merged_dataset = Dataset.from_list(merged_dataset)

    return merged_dataset


final_train = merge_processed_with_original(mnli["train"], processed_train)
final_val = merge_processed_with_original(mnli["validation_matched"], processed_val)

# Step 7: Combine into DatasetDict and save
processed_dataset = DatasetDict({
    "train": final_train,
    "validation_matched": final_val
})

save_path = "data/mnli_processed"
processed_dataset.save_to_disk(save_path)

print(f"Dataset saved successfully at {save_path}")
