# FILE: generate_report_visuals.py

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
JSON_FILE_PATH = "all_models_robustness_results.json"
OUTPUT_DIR = "report_visuals"

# --- Helper Functions ---

def load_data(filepath):
    """Loads the robustness results from a JSON file."""
    if not os.path.exists(filepath):
        print(f"Error: JSON file not found at '{filepath}'")
        print("Please make sure the script is in the same directory as your results file.")
        return None
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def generate_performance_table(data):
    """Creates a formatted markdown table for clean data performance."""
    table_data = []
    
    # Define a consistent model order for the table
    model_order = [
        "Fine-Tuned DL Model",
        "LogisticRegression_SturmSplit",
        "RandomForest_SturmSplit",
        "SVM_RBF_SturmSplit",
        "KNN_SturmSplit"
    ]
    
    # Map script names to paper-friendly names
    model_name_map = {
        "Fine-Tuned DL Model": "Advanced CNN (End-to-End)",
        "LogisticRegression_SturmSplit": "Logistic Regression (on Features)",
        "RandomForest_SturmSplit": "Random Forest (on Features)",
        "SVM_RBF_SturmSplit": "SVM (RBF) (on Features)",
        "KNN_SturmSplit": "KNN (on Features)"
    }

    for model_key in model_order:
        if model_key in data:
            # Get the result for the clean data (corruption level 0)
            clean_data_point = data[model_key]['additive_noise'][0]
            accuracy = clean_data_point['acc'] * 100
            f1_score = clean_data_point['f1']
            
            table_data.append({
                "Model": model_name_map.get(model_key, model_key),
                "Accuracy (%)": f"{accuracy:.1f}", # One decimal place
                "Macro F1-Score": f"{f1_score:.3f}" # Three decimal places
            })

    df = pd.DataFrame(table_data)
    
    print("--- Table 1: Model Performance on Standardized Sturm Test Set ---\n")
    print(df.to_markdown(index=False))
    print("\n" + "="*60 + "\n")


def plot_robustness(data, corruption_type, x_label, title):
    """
    Generates and saves a line plot comparing model robustness for a specific corruption type.
    """
    plot_data = []
    
    # Map script names to paper-friendly names for the legend
    model_name_map = {
        "Fine-Tuned DL Model": "Advanced CNN (End-to-End)",
        "LogisticRegression_SturmSplit": "Logistic Regression",
        "RandomForest_SturmSplit": "Random Forest",
        "SVM_RBF_SturmSplit": "SVM (RBF)",
        "KNN_SturmSplit": "KNN"
    }

    for model_name, results in data.items():
        if corruption_type in results:
            for point in results[corruption_type]:
                plot_data.append({
                    "Model": model_name_map.get(model_name, model_name),
                    "Level": point['level'],
                    "F1-Score": point['f1']
                })

    if not plot_data:
        print(f"No data found for corruption type: {corruption_type}")
        return

    df_plot = pd.DataFrame(plot_data)

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid') # Use a clean, professional style
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(
        data=df_plot,
        x="Level",
        y="F1-Score",
        hue="Model",
        style="Model",
        markers=True,
        dashes=False,
        palette="colorblind", # Colorblind-friendly palette
        linewidth=2.5,
        ax=ax
    )

    ax.set_title(title, fontsize=16, weight='bold')
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Macro F1-Score", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Set y-axis to start from 0 for better context
    ax.set_ylim(0, 1.0) 

    # Improve legend
    plt.legend(title='Model', fontsize=10, title_fontsize='12')
    
    plt.tight_layout()

    # Save the figure
    filename = f"robustness_{corruption_type.replace(' ', '_').lower()}.png"
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300) # High DPI for quality
    
    print(f"Plot saved to: {save_path}")
    plt.close(fig)


# --- Main Execution ---
if __name__ == "__main__":
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    results_data = load_data(JSON_FILE_PATH)
    
    if results_data:
        # 1. Generate and print the main performance table
        generate_performance_table(results_data)

        # 2. Generate and save the robustness plots
        plot_robustness(
            results_data,
            corruption_type='freq_mask',
            x_label='Frequency Masking Width',
            title='Model Robustness to Frequency Masking'
        )

        plot_robustness(
            results_data,
            corruption_type='time_mask',
            x_label='Time Masking Width',
            title='Model Robustness to Time Masking'
        )
        
        # This plot will show the flat lines, which is a key finding
        plot_robustness(
            results_data,
            corruption_type='additive_noise',
            x_label='Additive Noise Intensity (Sigma)',
            title='Model Robustness to Additive Gaussian Noise'
        )