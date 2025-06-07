import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, class_names, title, filename="confusion_matrix.png"):
    """
    Generates and saves a confusion matrix heatmap.
    
    Args:
        cm (array-like): The confusion matrix.
        class_names (list): A list of class names for the labels.
        title (str): The title for the plot.
        filename (str): The name of the file to save the plot.
    """
    # Convert to a numpy array
    cm_array = np.array(cm)

    # Set the figure size
    plt.figure(figsize=(12, 10))

    # Create the heatmap
    sns.heatmap(cm_array, 
                annot=True,         # Annotate cells with their values
                fmt='d',            # Format annotations as integers
                cmap='Blues',       # Use a blue color map
                xticklabels=class_names,
                yticklabels=class_names,
                linewidths=.5)
    
    # Add titles and labels for clarity
    plt.title(title, fontsize=18, pad=20, weight='bold')
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)

    # Rotate tick labels for better visibility
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(rotation=0, fontsize=12)

    # Adjust layout to prevent labels from being cut off
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(filename, dpi=300)
    print(f"✅ Confusion matrix saved to {filename}")
    
    # Display the plot
    plt.show()

def main():
    # --- 1. Set up argument parser to read filename from command line ---
    parser = argparse.ArgumentParser(
        description="Generate a confusion matrix from a JSON results file."
    )
    parser.add_argument(
        "json_file", 
        help="Path to the JSON file containing the training/testing results."
    )
    args = parser.parse_args()

    # --- 2. Read and load the JSON data from the specified file ---
    try:
        print(f"🔄 Reading data from '{args.json_file}'...")
        with open(args.json_file, 'r') as f:
            # Use json.load() to read from a file object
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: The file '{args.json_file}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"❌ Error: The file '{args.json_file}' is not a valid JSON file.")
        return

    # --- 3. Extract the final test confusion matrix ---
    if 'gtz_test' not in data or 'confusion_matrix' not in data['gtz_test']:
        print("❌ Error: 'gtz_test' key or its 'confusion_matrix' not found in JSON.")
        return
        
    test_results = data['gtz_test']
    cm_test = test_results['confusion_matrix']
    test_acc = test_results.get('test_acc', 0) # Use .get for safety
    test_f1 = test_results.get('test_f1', 0)

    # Define the class names for the GTZAN dataset
    gtzan_genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
                    'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    # Check if matrix dimensions match class names
    if len(cm_test) != len(gtzan_genres):
        print(f"❌ Error: Mismatch between confusion matrix size ({len(cm_test)}) and number of classes ({len(gtzan_genres)}).")
        return

    # --- 4. Generate and save the plot ---
    plot_title = (f'Confusion Matrix on GTZAN Test Set\n'
                  f'Accuracy: {test_acc:.2%} | Macro F1-Score: {test_f1:.4f}')

    plot_confusion_matrix(cm_test, gtzan_genres, plot_title, filename="cm_gtzan_test.png")


if __name__ == "__main__":
    main()