import os
import sys
import json
import argparse
import numpy as np
import torch
from typing import Dict
from tqdm import tqdm

# Import functions from your custom libraries.
from lib.data_utils import return1j2j, load_nf_models, createJetData
from lib.stat_utils import compute_mu_nuan_2NP_class, fit_2D_splines_bin_by_bin_from_dict, string_to_tuple_str
from lib.class_model import CombinedClassifier


class Args:
    hist_path: str
    json_save_path: str
    root_dir: str
    class_model_path: str
    models_dir: str


def main(args: Args):
    # --------------------------
    # Step 1: Load histogram templates.
    # --------------------------
    if not os.path.exists(args.hist_path):
        raise FileExistsError(f"No file named {args.hist_path} found")

    with open(args.hist_path, "r") as f:
        serializable_dict: Dict = json.load(f)
    
    if not serializable_dict:
        raise ValueError("Histogram dict is empty")

    # Convert the loaded dictionary to one with numpy arrays.
    hist_dict = {key: (np.array(v["sig"]), np.array(v["bg"])) for key, v in serializable_dict.items()}

    # Create dictionaries mapping parameter tuples to signal and background arrays.
    S_templates_2d_2j = {string_to_tuple_str(i): hist_dict[i][0] for i in hist_dict.keys()}
    B_templates_2d_2j = {string_to_tuple_str(i): hist_dict[i][1] for i in hist_dict.keys()}

    for parameter_mapping in (S_templates_2d_2j, B_templates_2d_2j):
        if not any(parameter_mapping.keys()):
            raise ValueError(f"Parameter mapping dict is fully malformed {parameter_mapping.keys()=}")

    # Fit 2D splines bin-by-bin using the dictionaries.
    bin_splines_S_class = fit_2D_splines_bin_by_bin_from_dict(S_templates_2d_2j)
    bin_splines_BG_class = fit_2D_splines_bin_by_bin_from_dict(B_templates_2d_2j)

    # --------------------------
    # Step 2: Load models and classifier.
    # --------------------------
    # Load Normalizing Flow models from the provided directory.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = load_nf_models(args.models_dir, device)

    # Load the classifier model from its checkpoint.
    class_model_load = (
        CombinedClassifier.load_from_checkpoint(args.class_model_path).to(device).eval().to(torch.float32)
    )

    # --------------------------
    # Step 3: Loop over a range of "mu" values and compute MLE ratios.
    # --------------------------
    MLE_ratio_arr = {}
    frac_array = np.linspace(0.1, 3.2, 10)
    N_sample = 50

    for frac in tqdm(frac_array, "Mu", position=0):
        MLE_ratio_arr[frac] = []
        # Generate a set of random seeds.
        seed_array = np.random.randint(100_000, size=N_sample)

        for seed in tqdm(seed_array, "Seed", position=1, leave=False):
            # Create jet data. The 'createJetData' function is assumed to use the
            # provided data object to return the full set of jets.
            alljet_data, _ = createJetData(
                "all",
                True,
                set_mu=frac,
                seed=seed,
                n_param=[1, 1, 1, 1, 1, 0],
                useRand=True,
                root_dir=args.root_dir,
            )
            # Split the data into 2-jet and 1-jet subsets.
            data_2j, data_1j, _, _ = return1j2j(alljet_data, models=models, device=device)

            # Compute the MLE mu using the provided classifier and fitted splines.
            mu = compute_mu_nuan_2NP_class(
                data_2j,
                data_1j,
                class_model_load,
                bin_splines_S_class,
                bin_splines_BG_class,
            )

            MLE_ratio_arr[frac].append(mu)
            print(f"Estimated mu: {mu}")

    # --------------------------
    # Step 4: Save the results to a JSON file.
    # --------------------------
    output_filename = os.path.join(args.json_save_path, "neyman.json")

    if not os.path.exists(os.path.dirname(output_filename)):
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    try:
        with open(output_filename, "w") as f:
            json.dump(MLE_ratio_arr, f)
        print(f"Dictionary successfully written to {output_filename}")
    except IOError as e:
        print(f"An I/O error occurred while writing the file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute MLE ratios for jet data using NF models and a classifier.")
    # Path to load the input histogram JSON file.
    parser.add_argument(
        "--hist_path",
        type=str,
        default="PreTrained/SavedStats/histogram_2jet1jet_class_4nf_200bins.json",
        help="Path to the input histogram JSON file.",
    )
    # Path to save the output JSON file.
    parser.add_argument(
        "--json_save_path",
        type=str,
        default="Test/SavedStats/",
        help="Path to save the resulting JSON file.",
    )
    # Root directory containing input data and subdirectories for ingestion/scoring.
    parser.add_argument(
        "--root_dir",
        type=str,
        default="../HEP-Challenge/",
        help="Root directory path for data files.",
    )
    # Path to the classifier model checkpoint.
    parser.add_argument(
        "--class_model_path",
        type=str,
        default="PreTrained/Models/DNN/DNNclass4NF_2.ckpt",
        help="Path to load classifier model checkpoint using CombinedClassifier.load_from_checkpoint.",
    )
    # Directory where the Normalizing Flow models are stored.
    parser.add_argument(
        "--models_dir",
        type=str,
        default="PreTrained/Models/",
        help="Path to the directory containing the '1_jet' and '2_jet' subdirectories with checkpoint files.",
    )

    args = parser.parse_args()

    # Set up paths for additional program directories.
    input_dir = os.path.join(args.root_dir, "input_data")
    program_dir = os.path.join(args.root_dir, "ingestion_program")
    score_dir = os.path.join(args.root_dir, "scoring_program")

    # Append ingestion and scoring program directories to sys.path for module imports.
    sys.path.append(program_dir)
    sys.path.append(score_dir)

    main(args)
