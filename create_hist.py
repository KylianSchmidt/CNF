import json
import os
import numpy as np
import torch
import argparse

from tqdm import tqdm

from lib.data_utils import return1j2j, load_nf_models, createJetData
from lib.class_model import CombinedClassifier


class Args:
    root_dir: str
    class_model_path: str
    json_save_path: str
    models_dir: str


def main(args: Args):
    """
    Main function to process jet data, generate histograms using a classifier model,
    and save the results to a JSON file.
    """
    # Load the classifier model from checkpoint.
    # The loaded model is expected to be callable with (data, jet_type) arguments.
    device = "cpu"

    class_model_load = (
        CombinedClassifier.load_from_checkpoint(args.class_model_path)
        .to(device)
        .eval()
        .to(torch.float32)
    )
    models = load_nf_models(args.models_dir, device)

    # Define the parameter arrays for jet energy scale (jes_arr) and testing scale (tes_arr).
    jes_arr = np.linspace(0.9, 1.1, 10)
    tes_arr = np.linspace(0.9, 1.1, 10)
    # Define histogram parameters.
    nbins = 200
    bins = np.linspace(0, 1, num=nbins)
    
    hist_dict_class = {} 
    # Loop over combinations of test and jet energy scale parameters.
    for j in tqdm(tes_arr, "TES", position=0):
        for i in tqdm(jes_arr, "JES", position=1, leave=False):
            # Define parameter list for data generation.
            n_params = [1, 1, 1, j, i, 0]
            
            # Create jet data using the provided root directory.
            alljet_data, _ = createJetData(
                "all",
                True,
                set_mu=1000,
                seed=0,
                n_param=n_params,
                useRand=False,
                root_dir=args.root_dir,
            )
            # Split the data into 2-jet and 1-jet sets and obtain corresponding labels.
            data_2j, data_1j, label_2j, label_1j = return1j2j(
                alljet_data=alljet_data,
                models=models,
                device=device,
            )
            
            # Obtain classifier scores for each jet type without computing gradients.
            with torch.no_grad():
                scores_2j = torch.sigmoid(class_model_load(data_2j, 2)).cpu().numpy()
                scores_1j = torch.sigmoid(class_model_load(data_1j, 1)).cpu().numpy()
            
            # Concatenate scores and labels from both jet types.
            total_score = np.concatenate([scores_2j, scores_1j])
            total_label = np.concatenate([label_2j.numpy(), label_1j.numpy()])
            
            # Compute histograms for signal (label==1) and background (label==0) separately.
            S_hist_class, _ = np.histogram(total_score[total_label == 1], bins=bins, density=True)
            BG_hist_class, _ = np.histogram(total_score[total_label == 0], bins=bins, density=True)
            
            # Save the histograms in the dictionary keyed by the parameter tuple.
            hist_dict_class[(i, j)] = [S_hist_class, BG_hist_class]
    
    # Create a serializable dictionary to save as JSON.
    # Convert NumPy arrays to lists for JSON compatibility.
    serializable_dict = {
        str(key): {'sig': val[0].tolist(), 'bg': val[1].tolist()}
        for key, val in hist_dict_class.items()
    }
    
    # Save the dictionary to a JSON file using the provided file path.
    with open(args.json_save_path+"hist.json", 'w') as f:
        json.dump(serializable_dict, f)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description="Process jet data and generate histograms using a classifier model."
    )
    # Argument for the root directory containing data.
    parser.add_argument(
        "--root_dir", 
        type=str, 
        default="../HEP-Challenge/",
        help="Root directory path for data files."
    )
    # Argument for the checkpoint path to load the classifier model.
    parser.add_argument(
        "--class_model_path", 
        type=str, 
        default="PreTrained/Models/DNN/DNNclass4NF_2.ckpt",
        help="Path to load classifier model checkpoint using CombinedClassifier.load_from_checkpoint."
    )
    # Argument for the output JSON file path.
    parser.add_argument(
        "--json_save_path", 
        type=str, 
        default="Test/SavedStats/",
        help="Path to save the resulting JSON file."
    )

    parser.add_argument(
        "--models_dir", 
        type=str, 
        default="PreTrained/Models/",
        help="Path to the directory containing the '1_jet' and '2_jet' subdirectories with checkpoint files."
    )
    
    args = parser.parse_args()

    input_dir = os.path.join(args.root_dir, "input_data")
    program_dir = os.path.join(args.root_dir, "ingestion_program")
    score_dir = os.path.join(args.root_dir, "scoring_program")
    # Append directories so that modules from these paths can be imported
    import sys
    sys.path.append(program_dir)
    sys.path.append(score_dir)

    main(args)

