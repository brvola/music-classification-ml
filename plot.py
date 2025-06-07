# FILE: generate_final_plots.py
# PURPOSE: To generate all tables and plots for the final report by combining
#          results from the advanced and baseline model robustness tests.

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
ADVANCED_MODEL_ROBUSTNESS_JSON = "advanced_model_robustness_results.json"
BASELINE_CNN_ROBUSTNESS_JSON = "baseline_cnn_robustness_results.json"
MAIN_MODEL_METRICS_JSON = "metrics_std_splits_specaug.json" # For training history
OUTPUT_DIR = "report_visuals"

# --- Helper Functions ---
def load_json_data(filepath, file_description):
    """Loads data from a JSON file with robust error handling."""
    if not os.path.exists(filepath):
        print(f"Warning: {file_description} file not found at '{filepath}'. Skipping related visuals.")
        return None
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}. File might be empty or corrupted.")
        return None

def generate_performance_table(adv_data, base_data):
    """Creates a markdown table comparing clean data performance."""
    table_data = []

    # Get performance for Advanced CNN (level 0)
    if adv_data and "Advanced CNN" in adv_data:
        clean_adv = adv_data["Advanced CNN"]['additive_noise'][0]
        table_data.append({
            "Model": "Advanced CNN",
            "Accuracy (%)": f"{clean_adv.get('acc', 0) * 100:.1f}",
            "Macro F1-Score": f"{clean_adv.get('f1', 0):.3f}"
        })

    # Get performance for Baseline CNN (level 0)
    if base_data and "Baseline CNN" in base_data:
        clean_base = base_data["Baseline CNN"]['additive_noise'][0]
        table_data.append({
            "Model": "Baseline CNN",
            "Accuracy (%)": f"{clean_base.get('acc', 0) * 100:.1f}",
            "Macro F1-Score": f"{clean_base.get('f1', 0):.3f}"
        })
    
    if not table_data:
        print("Could not generate performance table. Required data is missing.")
        return

    df = pd.DataFrame(table_data)
    print("--- Table: Head-to-Head Performance on Standardized Sturm Test Set ---\n")
    print(df.to_markdown(index=False))
    print("\n" + "="*80 + "\n")
    return df

def plot_overall_accuracy_comparison(df_perf):
    """Generates a bar chart comparing overall accuracy."""
    if df_perf is None or df_perf.empty:
        print("Cannot plot overall accuracy; performance data frame is empty.")
        return
        
    df_plot = df_perf.copy()
    # Convert accuracy back to float for plotting
    df_plot['Accuracy'] = df_plot['Accuracy (%)'].astype(float) / 100.0
    df_plot = df_plot.sort_values(by='Accuracy', ascending=False)
    
    colors = ['#d62728' if 'Advanced' in model else '#1f77b4' for model in df_plot['Model']]
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 5))
    
    sns.barplot(x='Accuracy', y='Model', data=df_plot, palette=colors, ax=ax)

    ax.set_title('Head-to-Head Model Accuracy on Sturm Test Set', fontsize=16, weight='bold')
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_ylabel('')
    ax.set_xlim(0, 1.0)
    ax.bar_label(ax.containers[0], fmt='%.3f', fontsize=10, padding=3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "overall_accuracy_comparison.png")
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")
    plt.close(fig)


def plot_training_history(metrics_data):
    """Plots the training and validation curves for the Advanced CNN."""
    if not metrics_data or 'gtz_finetune' not in metrics_data or not metrics_data['gtz_finetune']:
        print("Could not generate training history plot. Data is missing.")
        return

    df_history = pd.DataFrame(metrics_data['gtz_finetune'])
    if 'epoch' not in df_history.columns:
        df_history['epoch'] = range(1, len(df_history) + 1)
        
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(df_history['epoch'], df_history['train_acc'], 'o-', label='Train Accuracy', color='b')
    ax1.plot(df_history['epoch'], df_history['val_acc'], 'o-', label='Validation Accuracy', color='r')
    ax1.set_title('Advanced CNN: Training & Validation Accuracy', fontsize=14, weight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_ylim(0, 1.05); ax1.legend(); ax1.grid(True)

    ax2.plot(df_history['epoch'], df_history['train_loss'], 'o-', label='Train Loss', color='b')
    ax2.plot(df_history['epoch'], df_history['val_loss'], 'o-', label='Validation Loss', color='r')
    ax2.set_title('Advanced CNN: Training & Validation Loss', fontsize=14, weight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "training_history.png")
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")
    plt.close(fig)

def plot_comparative_robustness(adv_data, base_data):
    """Generates line plots comparing the robustness of both models."""
    if not adv_data or not base_data:
        print("Could not generate robustness plots. Data for one or both models is missing.")
        return
        
    combined_data = {**adv_data, **base_data}
    corruption_tests = {
        'freq_mask': 'Frequency Masking Width',
        'time_mask': 'Time Masking Width',
        'additive_noise': 'Additive Noise Intensity (Sigma)'
    }
    
    print("\n--- Generating Comparative Robustness Plots ---\n")
    for corr_key, x_label in corruption_tests.items():
        plot_data = []
        for model_name, results in combined_data.items():
            if corr_key in results:
                for point in results[corr_key]:
                    plot_data.append({
                        "Model": model_name,
                        "Level": point['level'],
                        "F1-Score": point['f1']
                    })
        
        df_plot = pd.DataFrame(plot_data)

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.lineplot(
            data=df_plot, x="Level", y="F1-Score", hue="Model", style="Model",
            markers=True, dashes=False, palette="colorblind", linewidth=2.5, ax=ax
        )
        
        ax.set_title(f'Model Robustness to {corr_key.replace("_", " ").title()}', fontsize=16, weight='bold')
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel("Macro F1-Score", fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_ylim(0, 1.0)
        ax.legend(title='Model', fontsize=10, title_fontsize='12')
        
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, f"comparative_robustness_{corr_key}.png")
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to: {save_path}")
        plt.close(fig)


# --- Main Execution ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Load all necessary data files
    adv_robustness_data = load_json_data(ADVANCED_MODEL_ROBUSTNESS_JSON, "Advanced CNN robustness results")
    base_robustness_data = load_json_data(BASELINE_CNN_ROBUSTNESS_JSON, "Baseline CNN robustness results")
    main_model_metrics = load_json_data(MAIN_MODEL_METRICS_JSON, "Main model metrics")
    
    # 1. Performance Table and Bar Chart
    perf_df = None
    if adv_robustness_data and base_robustness_data:
        perf_df = generate_performance_table(adv_robustness_data, base_robustness_data)
        plot_overall_accuracy_comparison(perf_df)
        
    # 2. Main Model Training History Plot
    if main_model_metrics:
        plot_training_history(main_model_metrics)

    # 3. Comparative Robustness Plots
    if adv_robustness_data and base_robustness_data:
        plot_comparative_robustness(adv_robustness_data, base_robustness_data)