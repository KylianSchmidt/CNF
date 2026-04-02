# Adversarial Normalizing Flows for High Precision Classification
Solution to the FAIR UNIVERSE - Higgs Uncertainty Challenge
## Author(s): 
Ibrahim Elsharkawy ie4@illinois.edu

# How to run the solution 
This document describes the hardware setup, software dependencies, installation steps, and instructions for training, evaluating, and predicting with the HEP-Challenge models. Please read carefully to ensure the correct environment and file structure.

---

## Software Dependencies and Installation

### 1 Required Software
- **Python:** (Ensure you are using a compatible version, e.g., Python 3.8+). This fork was tested using python 3.12
- **pip:** For installing Python dependencies or **astral-uv** [installation](https://docs.astral.sh/uv/getting-started/installation/)
- **3rd-Party Libraries:** Listed in `requirements.txt`

### Installation Steps

1. **Clone the HEP-Challenge Repository**

   The HEP-Challenge repository must be cloned *outside* the current archive. Replace `/desired/path/HEP-Challenge/` with your preferred directory.

   ```bash
   git clone https://github.com/FAIR-Universe/HEP-Challenge.git /desired/path/HEP-Challenge/
   ```

2. **Download Public Data**

   Download the public dataset with the following command:

   ```bash
   wget -O public_data.zip https://www.codabench.org/datasets/download/b9e59d0a-4db3-4da4-b1f8-3f609d1835b2/
   ```

3. **Move and Extract the Data**

   Move the downloaded zip file into the HEP-Challenge repository, then extract it:

   ```bash
   mv public_data.zip /desired/path/HEP-Challenge/
   cd /desired/path/HEP-Challenge/
   unzip public_data.zip
   ```

4. **Install Python Dependencies**

   From the root of this project, install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

   Or if using `astral-uv`, simply do

   ```bash
   uv sync
   ```

---

## Model Training and Execution

### 1 Training the Normalizing Flow (NF) Model

Run `train_NF.py` with the following arguments:

```bash
python train_NF.py \
  --root-dir "../HEP-Challenge/" \
  --checkpoint-path "Test/" \
  --c 1 \
  --channel signal \
  --num_jets 1
```

**Argument Details:**
- `--root-dir`: Root directory containing `input_data`, `ingestion_program`, and `scoring_program` folders.
- `--checkpoint-path`: Directory to save and/or load the model checkpoint.
- `--c`: Hyperparameter (default value is 1).
- `--channel`: Train on either "signal" or "background"
- `--num_jets`: either 1 or 2 (although 0 is not forbidden)
---

### 2 Training the Classifier Model

Run `train_class.py` with these arguments:

```bash
python train_class.py \
  --root-dir "../HEP-Challenge/" \
  --models-dir "PreTrainedOwnTest/Models/" \
  --checkpoint-path "Test/"
```

**Argument Details:**
- `--root-dir`: Root directory for data files.
- `--models-dir`: Directory containing subdirectories `1_jet` and `2_jet` with checkpoint files.
- `--checkpoint-path`: Directory where model checkpoints will be saved during training.

---

### 3 Creating Histograms

Run `create_hist.py` with the following arguments:

```bash
python create_hist.py \
  --root-dir "../HEP-Challenge/" \
  --class_model_path "PreTrainedOwnTest/DNNclass.ckpt" \
  --json_save_path "Test/SavedStats/" \
  --models-dir "PreTrainedOwnTest/Models/"
```

**Argument Details:**
- `--root-dir`: Root directory path for data files.
- `--class_model_path`: Path to load the classifier model checkpoint.
- `--json_save_path`: Directory where the resulting JSON file will be saved.
- `--models-dir`: Directory containing NF model checkpoints.

---

### 4 Running Neyman Construction

Run `create_neyman.py` with these arguments:

```bash
python create_neyman.py \
  --hist_path "PreTrainedOwnTest/hist.json" \
  --json_save_path "Test/SavedStats/" \
  --root-dir "../HEP-Challenge/" \
  --class_model_path "PreTrainedOwnTest/DNNclass.ckpt" \
  --models-dir "PreTrained/Models/"
```

**Argument Details:**
- `--hist_path`: Path to the input histogram JSON file.
- `--json_save_path`: Directory to save the output JSON file.
- `--root-dir`: Root directory path for data files.
- `--class_model_path`: Path to the classifier model checkpoint.
- `--models-dir`: Directory containing `1_jet` and `2_jet` subdirectories with NF model checkpoint files.

---

### 5 Prediction

Run `predict.py` with these arguments:

```bash
python predict.py \
  --hist_path "PreTrainedOwnTest/hist.json" \
  --json_save_path "Test/SavedStats/" \
  --root_dir "../HEP-Challenge/" \
  --class_model_path "PreTrainedOwnTest/DNNclass.ckpt" \
  --models_dir "PreTrainedOwnTest/Models/" \
  --neyman_path "PreTrainedOwnTest/SavedStats/neyman.json" \
  --mu 1 \
  --predict_numevents \
  --nevent 10
```

**Argument Details:**
- `--hist_path`: Path to the input histogram JSON file.
- `--json_save_path`: Directory to save the resulting JSON file.
- `--root_dir`: Root directory for data files.
- `--class_model_path`: Path to load the classifier model checkpoint.
- `--models_dir`: Directory containing NF model checkpoints.
- `--neyman_path`: Path to the Neyman JSON file.
- `--mu`: Hyperparameter value (default is 1).
- `--predict_numevents`: Flag to indicate whether to predict mu on a test dataset.
- `--nevent`: Number of events to test if `predict_numevents` is not activated.

---

## Important Side Effects

- **Directory Structure:**  
  - Models are saved in `Test/lighting_logs`. The subsequent processing steps will fail if the directory structure is not changed or if the paths are incorrect.

---

## Key Assumptions

- The HEP-Challenge repository is cloned outside the current archive and its path is correctly provided via the `--root-dir` argument.
- The repository includes the necessary folder structure (`input_data`, `ingestion_program`, `scoring_program`).
- Public data is downloaded, moved, and extracted into the HEP-Challenge repository as specified.
- The output directories (e.g., `Test/SavedStats/`) are empty or properly configured before starting a new training or evaluation run.

---