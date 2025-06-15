####Imports
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch 
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import numpy as np


def preprocess_dataset() -> pd.DataFrame:
    # Dataset
    df = pd.read_csv("data/QueerReclaimLex.csv")

    # Remove missing values in relevant columns
    df = df[["template", "HARMFUL_IN_gold", "perspective", "detoxify"]].dropna()

    # Rename columns -- decided to take this column as the binary labels, since the use of it in ingroups is what we want to test, where reclamation is supposed to not be seen as harmful 
    df["label"] = df["HARMFUL_IN_gold"].astype(int)
    df["text"] = df["template"].astype(str)

    # Do the same for the perspective and detoxify scores
    df["pred_perspective"] = (df["perspective"] > 0.5).astype(int)
    df["pred_detoxify"] = (df["detoxify"] > 0.5).astype(int)

    return df

def load_model():
    # define device -- here it's a macbook, so CPU
    device = torch.device("cpu")

    model_name = "EZiisk/EZ_finetune_Vidgen_model_RHS_ensemble"
    base_tokenizer = "facebook/roberta-hate-speech-dynabench-r4-target"

    tokenizer = AutoTokenizer.from_pretrained(base_tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    return model, tokenizer, device

def run_model():
    # Get dataset
    df = preprocess_dataset()

    # Get model, tokenizer
    model, tokenizer, device = load_model()

    # Tokenize the input
    tokens = tokenizer(list(df['text']), padding=True, truncation=True, return_tensors="pt")
    labels = torch.tensor(df["label"].values)

    # Create evaluation dataset
    dataset = TensorDataset(tokens["input_ids"], tokens["attention_mask"], labels)
    loader = DataLoader(dataset, batch_size=32)

    # Model usage
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for input_ids, attention_mask, batch_labels in loader:
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_labels.numpy())

    return all_preds, all_labels, df


def eval_model():
    # Get predictions and labels
    all_preds, all_labels, df = run_model()

    report_model = classification_report(all_labels, all_preds, target_names=["non-hate", "hate"], output_dict=True)
    report_perspective = classification_report(df["label"], df["pred_perspective"], target_names=["non-hate", "hate"], output_dict=True)
    report_detoxify = classification_report(df["label"], df["pred_detoxify"], target_names=["non-hate", "hate"], output_dict=True)

    return report_model, report_perspective, report_detoxify

def extract_metrics(report, class_name):
    """Helper to extract and round precision, recall, f1 for a given class"""
    p = report[class_name]['precision']
    r = report[class_name]['recall']
    f1 = report[class_name]['f1-score']
    return f"{p:.2f}", f"{r:.2f}", f"{f1:.2f}"

def generate_latex_table(report_model, report_perspective, report_detoxify):
    # Extract scores
    m_non_p, m_non_r, m_non_f1 = extract_metrics(report_model, "non-hate")
    m_hate_p, m_hate_r, m_hate_f1 = extract_metrics(report_model, "hate")

    p_non_p, p_non_r, p_non_f1 = extract_metrics(report_perspective, "non-hate")
    p_hate_p, p_hate_r, p_hate_f1 = extract_metrics(report_perspective, "hate")

    d_non_p, d_non_r, d_non_f1 = extract_metrics(report_detoxify, "non-hate")
    d_hate_p, d_hate_r, d_hate_f1 = extract_metrics(report_detoxify, "hate")

    # Compose LaTeX string
    latex_table = f"""
        \\begin{{table}}[h!]
        \\centering
        \\begin{{tabular}}{{llccc}}
        \\toprule
        \\textbf{{Model}} & \\textbf{{Class}} & \\textbf{{Precision}} & \\textbf{{Recall}} & \\textbf{{F1 Score}} \\\\
        \\midrule
        \\multirow{{Detoxify}} 
            & non-hate & {d_non_p} & {d_non_r} & {d_non_f1} \\\\
            & hate     & {d_hate_p} & {d_hate_r} & {d_hate_f1} \\\\
        \\midrule
        \\multirow{{Perspective}} 
            & non-hate & {p_non_p} & {p_non_r} & {p_non_f1} \\\\
            & hate     & {p_hate_p} & {p_hate_r} & {p_hate_f1} \\\\
        \\midrule
        \\multirow{{RHS Model}} 
            & non-hate & {m_non_p} & {m_non_r} & {m_non_f1} \\\\
            & hate     & {m_hate_p} & {m_hate_r} & {m_hate_f1} \\\\
        \\bottomrule
        \\end{{tabular}}
        \\caption{{Precision, Recall, and F1 scores for hate and non-hate classes across Detoxify, Perspective, and RHS models on HARMFUL\\_IN classification.}}
        \\label{{tab:performance_comparison}}
        \\end{{table}}
        """
    return latex_table


if __name__ == '__main__':
    report_model, report_perspective, report_detoxify = eval_model()
    latex_tabel = generate_latex_table(report_model, report_perspective, report_detoxify)
    with open("score_table.txt", "w") as f:
        f.write(latex_tabel)
