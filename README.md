# Dynamic Medication Treatment Recommendation with SRL-RNN

This project replicates the "Supervised Reinforcement Learning with Recurrent Neural Network for Dynamic Medication Treatment" method. It aims to recommend optimal treatment strategies for patients based on their medical history using a combination of Supervised Learning (SL) and Reinforcement Learning (RL).

## Project Structure

- `data/`: Contains raw and processed data (CSV, SQL).
- `models/`: Stores trained model checkpoints (`.pth`).
- `scripts/`: Python scripts for data preprocessing.
- `main.ipynb`: The main Jupyter Notebook for training and evaluating the SRL-RNN and Basic LSTM models.
- `requirements.txt`: Python package requirements.

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

Install dependencies:
```bash
pip install -r requirements.txt
```

## How to Run

1.  Place the MIMIC-III dataset files in the `data/` directory (if not already present).
2.  Run the preprocessing scripts in `scripts/` if you need to regenerate processed data:
    ```bash
    python scripts/Remove_HADM_missing_variables_1.py
    python scripts/Impute_missing_var_2.py
    python scripts/Pre_process_matrix_3.py
    python scripts/GET_ATC_CODES_API.py
    ```
    *Note: You may need to adjust file paths in these scripts if running them directly.*
3.  Open `main.ipynb` in VS Code or Jupyter Notebook.
4.  Run all cells to train the models and visualize results.

## Model Training

The notebook trains two models:
1.  **SRL-RNN**: The proposed Supervised Reinforcement Learning model.
2.  **Basic LSTM**: A baseline LSTM model for comparison.

Results, including loss curves and Jaccard scores, are plotted at the end of the training sections.
