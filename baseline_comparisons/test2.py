# FILE: generate_report_visuals.py (Now with more plots!)

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# --- Configuration ---
BASELINE_RESULTS_JSON = "sturm_split_baseline_results.json"
MAIN_MODEL_METRICS_JSON = "../metrics_std_splits_specaug.json"
ROBUSTNESS_JSON = "all_models_robustness_results.json"
OUTPUT_DIR = "report_visuals"
GTZAN_CLASSES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# --- Helper Functions ---
def load_json_data(filepath, file_description):
    """Loads data from a JSON file with error handling."""
    if not os.path.exists(filepath):
        print(f"Error: {file_description} file not found at '{filepath}'")
        return None
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def generate_performance_table(data):
    """Creates a formatted markdown table for clean data performance."""
    # This function remains the same as before.
    table_data = []
    model_order = [
        "Fine-Tuned DL Model", "LogisticRegression_SturmSplit", "RandomForest_SturmSplit",
        "SVM_RBF_SturmSplit", "KNN_SturmSplit"
    ]
    model_name_map = {
        "Fine-Tuned DL Model": "Advanced CNN (End-to-End)",
        "LogisticRegression_SturmSplit": "Logistic Regression (on Features)",
        "RandomForest_SturmSplit": "Random Forest (on Features)",
        "SVM_RBF_SturmSplit": "SVM (RBF) (on Features)",
        "KNN_SturmSplit": "KNN (on Features)"
    }
    for model_key in model_order:
        if model_key in data:
            clean_data_point = data[model_key]['additive_noise'][0]
            accuracy = clean_data_point['acc'] * 100
            f1_score = clean_data_point['f1']
            table_data.append({
                "Model": model_name_map.get(model_key, model_key),
                "Accuracy (%)": f"{accuracy:.1f}",
                "Macro F1-Score": f"{f1_score:.3f}"
            })
    df = pd.DataFrame(table_data)
    print("--- Table 1: Model Performance on Standardized Sturm Test Set ---\n")
    print(df.to_markdown(index=False))
    print("\n" + "="*60 + "\n")

def plot_genre_performance(main_model_data, baseline_data):
    """
    Plots a bar chart comparing the per-genre F1-scores of the best models.
    """
    plot_data = []
    
    # 1. Get per-genre F1 scores for the Advanced DL model (from test results)
    if 'gtz_test' in main_model_data and 'test_f1' in main_model_data['gtz_test']:
        # This part requires the classification report to be saved in the test results.
        # Let's assume the F1 scores are manually added or extracted from a saved report.
        # For now, let's pull from the baseline JSON as a placeholder. A more robust way
        # would be to ensure the test report is saved from main.py.
        # Let's use the clean data from the robustness JSON as the source of truth.
        pass # We will populate this from the robustness JSON below.

    # 2. Get per-genre F1 scores for the best baseline model (e.g., Logistic Regression)
    best_baseline_key = "LogisticRegression(Sturm_Split_Features)"
    if baseline_data and best_baseline_key in baseline_data:
        report = baseline_data[best_baseline_key].get('classification_report', {})
        for genre, metrics in report.items():
            if genre in GTZAN_CLASSES:
                plot_data.append({
                    "Model": "Logistic Regression",
                    "Genre": genre,
                    "F1-Score": metrics.get('f1-score', 0)
                })

    # 3. Use the clean data from the robustness JSON for the Advanced CNN
    robustness_data = load_json_data(ROBUSTNESS_JSON, "Robustness results")
    if robustness_data:
        # We need to calculate the per-genre F1 scores for the DL model on clean data.
        # This is a bit tricky without saving the full report. Let's make an assumption
        # and plot the baselines' genre performance for now, which is still very valuable.
        # A full implementation would require modifying main.py to save this report.
        # FOR NOW: We will plot the two best baselines: Logistic Regression and Random Forest.
        pass
    
    # Let's re-scope this to plot the top 3 baselines for simplicity and clarity
    baseline_keys_to_plot = {
        "LogisticRegression(Sturm_Split_Features)": "Logistic Regression",
        "RandomForest_SturmSplit(Sturm_Split_Features)": "Random Forest"
    }
    plot_data_baselines = []
    if baseline_data:
        for key, name in baseline_keys_to_plot.items():
             # Check for exact key, or key without the suffix
            actual_key = key if key in baseline_data else key.replace('(Sturm_Split_Features)','')
            if actual_key in baseline_data:
                report = baseline_data[actual_key].get('classification_report', {})
                for genre, metrics in report.items():
                    if genre in GTZAN_CLASSES:
                        plot_data_baselines.append({
                            "Model": name, "Genre": genre, "F1-Score": metrics.get('f1-score', 0)
                        })

    if not plot_data_baselines:
        print("Could not generate genre performance plot. Baseline data is missing.")
        return

    df_genre = pd.DataFrame(plot_data_baselines)

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    sns.barplot(
        data=df_genre,
        x="Genre",
        y="F1-Score",
        hue="Model",
        palette="viridis",
        ax=ax
    )

    ax.set_title("Per-Genre F1-Score Comparison of Top Baseline Models", fontsize=16, weight='bold')
    ax.set_xlabel("Genre", fontsize=12)
    ax.set_ylabel("F1-Score", fontsize=12)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.set_ylim(0, 1.0)
    
    plt.legend(title='Model', fontsize=10, title_fontsize='12')
    plt.tight_layout()

    # Save the figure
    save_path = os.path.join(OUTPUT_DIR, "genre_performance_comparison.png")
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")
    plt.close(fig)


def plot_training_history(data):
    """
    Plots the training and validation accuracy/loss curves from the main model's metrics file.
    """
    if 'gtz_finetune' not in data:
        print("No 'gtz_finetune' data found in metrics file.")
        return

    df_history = pd.DataFrame(data['gtz_finetune'])

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Accuracy Plot
    ax1.plot(df_history['epoch'], df_history['train_acc'], 'o-', label='Train Accuracy', color='b')
    ax1.plot(df_history['epoch'], df_history['val_acc'], 'o-', label='Validation Accuracy', color='r')
    ax1.set_title('Advanced CNN: Training & Validation Accuracy', fontsize=14, weight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    ax1.grid(True)

    # Loss Plot
    ax2.plot(df_history['epoch'], df_history['train_loss'], 'o-', label='Train Loss', color='b')
    ax2.plot(df_history['epoch'], df_history['val_loss'], 'o-', label='Validation Loss', color='r')
    ax2.set_title('Advanced CNN: Training & Validation Loss', fontsize=14, weight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(OUTPUT_DIR, "training_history.png")
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")
    plt.close(fig)

def plot_robustness(data, corruption_type, x_label, title):
    """Generates and saves a line plot comparing model robustness."""
    # This function remains the same as before.
    plot_data = []
    model_name_map = {
        "Fine-Tuned DL Model": "Advanced CNN (End-to-End)",
        "LogisticRegression_SturmSplit": "Logistic Regression",
        "RandomForest_SturmSplit": "Random Forest",
        "SVM_RBF_SturmSplit": "SVM (RBF)",
        "KNN_SturmSplit": "KNN"
    }
    for model_name, results in data.items():
        if corruption_type in results:
            # Handle potential key inconsistencies from saving
            clean_model_name = model_name.replace('(Sturm_Split_Features)', '')
            for point in results[corruption_type]:
                plot_data.append({
                    "Model": model_name_map.get(clean_model_name, clean_model_name),
                    "Level": point['level'],
                    "F1-Score": point['f1']
                })
    if not plot_data:
        print(f"No data found for corruption type: {corruption_type}")
        return
    df_plot = pd.DataFrame(plot_data)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=df_plot, x="Level", y="F1-Score", hue="Model", style="Model",
        markers=True, dashes=False, palette="colorblind", linewidth=2.5, ax=ax
    )
    ax.set_title(title, fontsize=16, weight='bold')
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Macro F1-Score", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_ylim(0, 1.0)
    plt.legend(title='Model', fontsize=10, title_fontsize='12')
    plt.tight_layout()
    filename = f"robustness_{corruption_type.replace(' ', '_').lower()}.png"
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")
    plt.close(fig)

# --- Main Execution ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Load all necessary data files
    robustness_data = load_json_data(ROBUSTNESS_JSON, "Robustness results")
    baseline_data = load_json_data(BASELINE_RESULTS_JSON, "Baseline results")
    main_model_metrics = load_json_data(MAIN_MODEL_METRICS_JSON, "Main model metrics")
    
    # 1. Performance Table
    if robustness_data:
        generate_performance_table(robustness_data)
    
    # 2. Genre-wise Performance Plot (using baseline report)
    if baseline_data:
        plot_genre_performance(main_model_metrics, baseline_data)
        
    # 3. Main Model Training History Plot
    if main_model_metrics:
        plot_training_history(main_model_metrics)

    # 4. Robustness Plots
    if robustness_data:
        print("\n--- Generating Robustness Plots ---\n")
        plot_robustness(
            robustness_data, 'freq_mask', 'Frequency Masking Width', 'Model Robustness to Frequency Masking'
        )
        plot_robustness(
            robustness_data, 'time_mask', 'Time Masking Width', 'Model Robustness to Time Masking'
        )
        plot_robustness(
            robustness_data, 'additive_noise', 'Additive Noise Intensity (Sigma)', 'Model Robustness to Additive Gaussian Noise'
        )