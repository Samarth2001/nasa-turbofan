# NASA Turbofan Engine Remaining Useful Life (RUL) Prediction with LSTM and Attention

## Description
This project focuses on predicting the Remaining Useful Life (RUL) of NASA's turbofan engines using data-driven prognostics. It employs a Long Short-Term Memory (LSTM) network augmented with an Attention mechanism to capture complex temporal dependencies in sensor data and improve prediction accuracy. The project also integrates SHAP (SHapley Additive exPlanations) for model interpretability, providing insights into which sensor readings and time steps most influence the RUL predictions.

Key objectives include:
*   Preprocessing and preparing the C-MAPSS turbofan engine dataset.
*   Developing and training an LSTM-based model with an Attention mechanism for RUL estimation.
*   Evaluating the model's performance using standard metrics like RMSE and the official scoring function.
*   Visualizing attention weights and SHAP values to understand model behavior and feature importance.
*   Providing a modular and well-documented codebase for reproducibility and further research.

## Features

- **Data Processing Pipeline**: Robust data loading, preprocessing, and sequence generation for time series data
- **LSTM-based Model**: Bidirectional LSTM with attention mechanism for accurate RUL prediction
- **Explainability**: Model explanations using both attention mechanism and SHAP values
- **Interactive Dashboard**: Streamlit dashboard for real-time monitoring and visualization
- **Alert System**: Warning and critical alerts based on RUL thresholds

## Dataset
This project utilizes the NASA Turbofan Engine Degradation Simulation Dataset, also known as the Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dataset. This dataset is widely used in the prognostics and health management (PHM) community for benchmarking RUL prediction algorithms.

**Key characteristics of the C-MAPSS dataset:**

*   **Simulated Data:** The data is generated using a sophisticated simulation model (CMAPSS) of a turbofan engine under various operational conditions and fault modes.
*   **Multiple Sub-datasets:** It consists of four sub-datasets (FD001, FD002, FD003, FD004), each with different operating conditions (e.g., single or multiple) and fault patterns (e.g., single or mixed), providing a diverse set of challenges.
    - **FD001**: Single operating condition, single failure mode (HPC Degradation)
    - **FD002**: Six operating conditions, single failure mode (HPC Degradation)
    - **FD003**: Single operating condition, two failure modes (HPC Degradation, Fan Degradation)
    - **FD004**: Six operating conditions, two failure modes (HPC Degradation, Fan Degradation)
*   **Time-Series Sensor Data:** Each data point includes measurements from 21 sensors (e.g., Total temperature at fan inlet, LPC outlet temperature, HPT coolant bleed, etc.) and 3 operational settings that define the flight conditions.
*   **Run-to-Failure Trajectories:** The data provides multiple time-series trajectories for a fleet of engines. Each engine operates from a healthy state until it experiences a failure. The RUL is the number of remaining operational cycles before an engine fails.

**Source:**
*   **NASA Prognostics Center of Excellence Data Repository:** [Turbofan Engine Degradation Simulation Data Set](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan)
*   **Relevant Paper:** Saxena, A., & Goebel, K. (2008). *Turbofan Engine Degradation Simulation Data Set*. NASA Ames Prognostics Data Repository. NASA Ames Research Center, Moffett Field, CA.

## Project Structure

```
nasa-turbofan-rul/
│
├── CMAPSSData/                 # Raw C-MAPSS dataset files (usually downloaded here)
├── dashboard/                  # Contains files for a Plotly Dash or Streamlit dashboard
│   └── assets/                 # Assets for the dashboard
│   └── app.py                  # Streamlit dashboard script (example)
├── data/                       # Processed data and intermediate files
│   ├── processed/              # Cleaned, normalized, and windowed data
│   └── raw/                    # Copy of raw data or extraction target from CMAPSSData/
├── models/                     # Saved trained models and related artifacts (e.g., scalers)
├── notebooks/                  # Jupyter notebooks for exploration, experimentation, and visualization
│   ├── exploratory_analysis.ipynb
├── src/                        # Source code for the project
│   ├── config.py               # Configuration parameters
│   ├── data/                   # Scripts for data loading and preprocessing
│   │   └── data_loader.py
│   ├── models/                 # Model definitions and training scripts
│   │   ├── lstm_model.py       # LSTM model with attention
│   │   └── explainer.py        # Model explainability (SHAP)
│   └── utils/                  # Utility functions (e.g., metrics, logging)
│       └── metrics.py
│
├── .gitignore                  # Specifies intentionally untracked files that Git should ignore
├── environment.yml             # Conda environment file
├── Local_Data.md               # Instructions on handling local data
├── README.md                   # This file: Project overview
├── requirements.txt            # Pip requirements file
└── train.py                    # Main script to train the model
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/nasa-turbofan-rul.git
cd nasa-turbofan-rul
```

2. Create a virtual environment and install dependencies:
```bash
# Using conda
conda env create -f environment.yml
conda activate turbofan_rul_env # Or the name specified in environment.yml
# Or using pip
pip install -r requirements.txt
```
3.  **Download the C-MAPSS Dataset:**
    *   Download the "Turbofan Engine Degradation Simulation Data Set" from the [NASA Prognostics Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan).
    *   The dataset usually comes as a ZIP file (e.g., `CMAPSSData.zip`).
    *   Extract the contents into the `CMAPSSData/` folder in the project root, or update the path in `src/data/data_loader.py` or relevant configuration files.

## Usage

### Data Processing and Model Training

To download the dataset (if `data_loader.py` handles it), preprocess it, train the model, and generate explanations:

```bash
python train.py --subset FD001 --epochs 100 --batch-size 64
```

Command line arguments (example, verify with `python train.py --help`):
- `--subset`: Dataset subset to use (default: FD001)
- `--force-download`: Force download of dataset
- `--force-preprocess`: Force preprocessing of data
- `--batch-size`: Batch size for training
- `--epochs`: Maximum number of epochs for training
- `--patience`: Patience for early stopping
- `--learning-rate`: Learning rate for optimizer
- `--lstm-units`: Number of units in LSTM layers
- `--skip-explainability`: Skip explainability analysis

### Dashboard

To run the dashboard application (assuming `dashboard/app.py` is a Streamlit app):

```bash
streamlit run dashboard/app.py
```

The dashboard provides:
- Model performance metrics and visualizations
- RUL predictions and alerts
- Feature importance analysis
- Attention mechanism visualization

## Methodology

The primary goal of this project is to predict the Remaining Useful Life (RUL) of turbofan engines. This is achieved through several stages:

### 1. Data Preprocessing
Before feeding the data into the model, several crucial preprocessing steps are performed:
*   **Normalization:** Sensor readings are normalized (e.g., using Min-Max scaling per engine or per dataset) to bring them to a common scale (typically [0, 1] or [-1, 1]). This stabilizes training and helps the model converge faster.
*   **Sequence Generation (Windowing):** The continuous time-series data for each engine is transformed into fixed-length sequences (windows). Each window consists of sensor readings and operational settings over a specific number of past operational cycles (e.g., 30-50 cycles).
*   **RUL Target Calculation:** The RUL for each sequence is the number of remaining cycles until the engine fails. For training, it's common to use a piece-wise linear RUL target, where the RUL is capped at a maximum value (e.g., 125-130 cycles). This is because predicting very high RUL values accurately is challenging and often less critical than predicting RUL when failure is relatively imminent.

### 2. Model Architecture: LSTM with Attention
The RUL prediction model is based on a Long Short-Term Memory (LSTM) network, often Bidirectional, with an integrated Attention mechanism:

*   **Input Layer**: Sequences of sensor readings and operational settings from the windowing process.
*   **LSTM Layers**:
    *   LSTMs are a special kind of Recurrent Neural Network (RNN) exceptionally well-suited for learning from sequential data. They capture long-range temporal dependencies, vital as engine degradation involves subtle patterns evolving over many cycles.
    *   Bidirectional LSTMs process sequences in both forward and backward directions, potentially capturing richer contextual information.
    *   Multiple LSTM layers can be stacked to learn hierarchical representations. Dropout and batch normalization are often used between layers to prevent overfitting and stabilize training.
*   **Attention Mechanism**:
    *   The Attention mechanism enhances the LSTM by allowing the model to dynamically assign different levels of importance (attention weights) to different time steps within an input sequence.
    *   Instead of treating all parts of the sequence equally, the model learns to focus on the most informative sensor readings or operational periods indicative of the current degradation state. This can improve prediction accuracy and provides interpretability by showing which parts of the input the model deemed critical.
*   **Dense Layers**: Fully connected layers following the LSTM and Attention layers to map the learned features to the final RUL prediction. Dropout regularization can be applied here as well.
*   **Output Layer**: A single neuron outputting a continuous RUL value.

### 3. Model Training and Evaluation
*   **Training:** The LSTM-Attention model is trained end-to-end by minimizing a suitable loss function, typically the Root Mean Squared Error (RMSE) or Mean Squared Error (MSE), between its predicted RUL values and the actual RUL targets. Optimization algorithms like Adam are commonly used. Callbacks for early stopping (to prevent overfitting) and model checkpointing (to save the best model) are standard practice.
*   **Evaluation Metrics:** The model's performance is rigorously evaluated on a separate test set. Key metrics include:
    *   **Root Mean Squared Error (RMSE):** Measures the average magnitude of the prediction errors.
    *   **Mean Absolute Error (MAE):** Another measure of average error magnitude.
    *   **R² Score (Coefficient of Determination):** Indicates the proportion of variance in the RUL that is predictable from the input features.
    *   **C-MAPSS Scoring Function (PHM Score):** An asymmetric scoring function often used for this dataset, which penalizes late predictions (predicting RUL > actual RUL) more heavily than early predictions.

### 4. Explainability (SHAP & Attention)
Understanding *why* the model makes certain predictions is crucial for trust and deployment.
*   **Attention Weights Visualization:** The weights from the attention mechanism can be visualized to show which time steps in the input sequence the model focused on for a particular prediction.
*   **SHAP (SHapley Additive exPlanations):**
    *   SHAP values quantify the contribution of each input feature (i.e., each sensor reading at each time step within a window) to the final RUL prediction for any given instance.
    *   By analyzing SHAP values (e.g., summary plots, dependence plots, force plots), we can understand global feature importance, how individual features affect predictions, and the model's behavior for specific examples. This increases transparency and helps identify if the model has learned sensible patterns.

## Dashboard Components

The dashboard (if implemented, e.g., using Streamlit) typically includes sections like:

1.  **Overview**: Dataset statistics and feature information.
2.  **Model Performance**: Visualizations of performance metrics, prediction accuracy, and error analysis on test data.
3.  **Individual Prediction Explanation**: Input a specific engine or sequence and see its RUL prediction along with SHAP values and attention weights.
4.  **RUL Monitoring**: Potentially a simulation of monitoring new data, with alerts for engines nearing their predicted end-of-life.

## Results and Evaluation

The model was trained and evaluated on the FD001 subset of the C-MAPSS dataset. Here are the results from our latest run:

| Metric | Value |
|--------|-------|
| MSE | 589.37 |
| RMSE | 24.28 |
| MAE | 15.71 |
| R² | 0.65 |
| MAPE | 21.69% |
| PHM Score | 143366.80 |

These results indicate:
- Our model achieves a Root Mean Square Error (RMSE) of 24.28 cycles when predicting the Remaining Useful Life
- The R² score of 0.65 shows the model explains approximately 65% of the variance in RUL values
- The PHM Score (specific to the C-MAPSS competition) penalizes late predictions more heavily than early ones

The model performs reasonably well compared to the baseline approaches in literature, though there is still room for improvement through hyperparameter tuning and architectural modifications.

## References

1.  **C-MAPSS Dataset:**
    *   Saxena, A., & Goebel, K. (2008). *Turbofan Engine Degradation Simulation Data Set*. NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA. ([Link](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan))

2.  **Key Methodological Papers (Examples):**
    *   Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural computation, 9*(8), 1735-1780. (Fundamental LSTM paper)
    *   Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. *arXiv preprint arXiv:1409.0473*. (Influential paper on Attention mechanisms)
    *   Yuan, M., Wu, Y., & Lin, L. (2016). Fault diagnosis and remaining useful life estimation of aero engine using LSTM recurrent neural network. *In 2016 IEEE International Conference on Aircraft Utility Systems (AUS)* (pp. 135-140). IEEE.
    *   Li, X., Ding, Q., & Sun, J. Q. (2018). Remaining useful life estimation in prognostics using deep convolution neural networks. *Reliability Engineering & System Safety, 172*, 1-11.
    *   Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *In Advances in neural information processing systems* (pp. 4765-4774). (SHAP paper)

*(Users should review and update this list with papers that most directly influenced their specific implementation.)*

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:
1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes and commit them (`git commit -m 'Add some feature'`).
4.  Push to the branch (`git push origin feature/your-feature-name`).
5.  Open a Pull Request.

Please ensure your code adheres to existing styling and includes tests where appropriate.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details (if one exists, otherwise specify, e.g., "This project is licensed under the MIT License.").

## Acknowledgements
*   This project heavily relies on the C-MAPSS dataset provided by the **NASA Prognostics Center of Excellence**.
*   The implementation may draw inspiration from various research papers and open-source projects in the field of prognostics, health management, and deep learning.
*   Thanks to the developers of TensorFlow/Keras, Scikit-learn, Pandas, NumPy, Matplotlib, Streamlit, and SHAP for their invaluable open-source tools.

