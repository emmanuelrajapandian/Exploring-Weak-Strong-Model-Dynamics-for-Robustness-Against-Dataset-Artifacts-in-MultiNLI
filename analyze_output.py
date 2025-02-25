# read jsonl to pandas dataframe
import pandas as pd
import torch
import shap
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def prepare_dataset_nli(texts, tokenizer, max_seq_length=None):
    max_seq_length = tokenizer.model_max_length if max_seq_length is None else max_seq_length
    premises = []
    hypotheses = []
    for text in texts:
        if "[MASK]" in text and len(set(text.split())) == 1:
            # If the input is fully masked, insert a separator
            premise, hypothesis = "[MASK]", "[MASK]"
            text = f"{premise} [SEP] {hypothesis}"
        elif " [SEP] " in text:
            premise, hypothesis = text.split(" [SEP] ")
        else:
            premise = text
            hypothesis = ""
        
        premises.append(premise)
        hypotheses.append(hypothesis)

    tokenized_examples = tokenizer(
        premises,
        hypotheses,
        truncation=True,
        max_length=max_seq_length,
        padding='max_length',
        return_tensors="pt"
    )


    # print("input_ids shape:", tokenized_examples["input_ids"].shape)
    # print("attention_mask shape:", tokenized_examples["attention_mask"].shape)

    # # Check specific tensor values to confirm padding (0s should appear for padding tokens)
    # print("input_ids:", tokenized_examples["input_ids"])
    # print("attention_mask:", tokenized_examples["attention_mask"])

    

    return tokenized_examples


# Load the JSONL file
df = pd.read_json('eval_output/eval_predictions.jsonl', lines=True)

print(df.head())

# calculate global accuacy
df['correct'] = df['label'] == df['predicted_label']
print('Global accuracy:', df['correct'].mean())

# calculate accuracy for each class
print('Accuracy by class:')
print(df.groupby('label')['correct'].mean())

# Calculate accuracy per genre
print('Accuracy by genre:')
print(df.groupby('genre')['correct'].agg(['count', 'mean']))



# analyze shap values for slate genre
slate_df = df[df['genre']=='slate']
slate_texts = slate_df.apply(lambda row: f"{row['premise']} [SEP] {row['hypothesis']}", axis=1).tolist()


# Load your model and tokenizer
print("loading model and tokenizer...")
model = AutoModelForSequenceClassification.from_pretrained("./trained_model")
tokenizer = AutoTokenizer.from_pretrained("./trained_model")
model.eval()
print("done")

def predict(texts):
    inputs = prepare_dataset_nli(texts, tokenizer, max_seq_length=128 # TODO: max length is hardcoded for now
                                            )
    
    outputs= model(**inputs)
    return torch.softmax(outputs.logits, dim=-1).detach().numpy()

explainer = shap.Explainer(predict, tokenizer)
shap_values = explainer(slate_texts[:5])  # Adjust sample size as needed

# print("shap_values shape:", shap_values.shape)
# print("shap_values.values shape:", shap_values.values.shape)

# for i, sv in enumerate(shap_values):
#     print(f"Shape of shap_values[{i}]:", sv.values.shape)
#     print("shap_values entry:", sv.values)


# if len(shap_values.values.shape) == 1:
#     shap_values.values = shap_values.values.reshape(1, -1)


# shap.summary_plot(shap_values, slate_texts[:2])

print(type(shap_values))

tokenized_text = tokenizer.tokenize(slate_texts[0])
print(len(tokenized_text))
print(tokenized_text)

print("shap_values[0].values:", shap_values[0].values)
print(explainer.expected_value)

# shap.plots.force(explainer.expected_value[0], shap_values[0].values, tokenized_text)
shap.plots.text(shap_values)