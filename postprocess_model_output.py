# Copyright 2024 xfold authors
# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Part 3: AlphaFold 3 Postprocessing Script."""

import csv
import dataclasses
import os
import pathlib
import time
import glob
from collections.abc import Sequence
from typing import overload # Keep for potential future use, though not strictly needed now

from absl import app
from absl import flags
from alphafold3.common import folding_input
from alphafold3.model import features, post_processing
from alphafold3.model.model import InferenceResult, ModelResult
import alphafold3.cpp # For OUTPUT_TERMS_OF_USE

import numpy as np
import torch # For loading intermediate files
import torch.utils._pytree as pytree


# --- Input Flags ---
_INTERMEDIATE_INPUT_DIR = flags.DEFINE_string(
    'intermediate_input_dir',
    None,
    'Path to the directory containing intermediate preprocessed results (output of preprocess.py). '
    'Needed for batch data context during structure extraction.',
)
_RAW_RESULTS_INPUT_DIR = flags.DEFINE_string(
    'raw_results_input_dir',
    None,
    'Path to the directory containing raw model outputs (output of inference.py).',
)

# --- Output Flags ---
_FINAL_OUTPUT_DIR = flags.DEFINE_string(
    'final_output_dir',
    None,
    'Path to a directory where the final results (PDBs, rankings) will be saved.',
)

@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ResultsForSeed:
    """Stores the inference results (diffusion samples) for a single seed.

    Attributes:
      seed: The seed used to generate the samples.
      inference_results: The inference results, one per sample.
      # Removed full_fold_input as we load necessary parts on demand
    """
    seed: int
    inference_results: Sequence[InferenceResult]


def extract_structures(
    batch: features.BatchDict,
    result: ModelResult,
    target_name: str,
) -> list[InferenceResult]:
    """Generates structures from model outputs. (Adapted from ModelRunner)"""
    # Note: This function relies on the static method `get_inference_result`
    # from the original `alphafold3.model.model.Model` class. Ensure this
    # class or an equivalent is available in the environment.
    try:
        # We need the original Model class definition for its static method
        from alphafold3.model import model as af3_model_definition
        return list(
            af3_model_definition.Model.get_inference_result(
                batch=batch, result=result, target_name=target_name
            )
        )
    except ImportError:
        print("Error: Could not import `alphafold3.model.model.Model` for structure extraction.")
        print("Ensure the alphafold3 library is correctly installed.")
        raise
    except AttributeError:
         print("Error: `get_inference_result` static method not found in `alphafold3.model.model.Model`.")
         print("The class definition might have changed or is incomplete.")
         raise


def write_outputs(
    all_inference_results: Sequence[ResultsForSeed],
    output_dir: os.PathLike[str] | str,
    job_name: str,
) -> None:
    """Writes final outputs (PDBs, ranking) to the specified directory."""
    ranking_scores = []
    max_ranking_score = None
    max_ranking_result = None

    # Load terms of use text
    try:
        terms_path = (
            pathlib.Path(alphafold3.cpp.__file__).parent / 'OUTPUT_TERMS_OF_USE.md'
        )
        output_terms = terms_path.read_text()
    except Exception as e:
        print(f"Warning: Could not read OUTPUT_TERMS_OF_USE.md: {e}")
        output_terms = "Output Terms of Use could not be loaded."

    os.makedirs(output_dir, exist_ok=True)

    # Create per-sample output directories and write PDBs
    for results_for_seed in all_inference_results:
        seed = results_for_seed.seed
        for sample_idx, result in enumerate(results_for_seed.inference_results):
            sample_dir = os.path.join(
                output_dir, f'seed-{seed}_sample-{sample_idx}')
            os.makedirs(sample_dir, exist_ok=True)

            print(f"Writing output for seed {seed}, sample {sample_idx} to {sample_dir}")
            post_processing.write_output(
                inference_result=result, output_dir=sample_dir
            )

            # Store ranking score
            ranking_score = float(result.metadata.get('ranking_score', -1.0)) # Safely get score
            if ranking_score == -1.0:
                 print(f"Warning: 'ranking_score' not found in metadata for seed {seed}, sample {sample_idx}.")
            ranking_scores.append((seed, sample_idx, ranking_score))

            # Track best result
            if max_ranking_score is None or ranking_score > max_ranking_score:
                max_ranking_score = ranking_score
                max_ranking_result = result

    # Write the best ranked structure to the main job directory
    if max_ranking_result is not None:  # True iff ranking_scores non-empty and valid scores found
        print(f"Writing best ranked output (score: {max_ranking_score:.2f}) to {output_dir}")
        post_processing.write_output(
            inference_result=max_ranking_result,
            output_dir=output_dir,
            terms_of_use=output_terms,
            name=job_name, # Use the sanitised job name for the main output file
        )
    else:
        print(f"Warning: No valid ranking scores found for job {job_name}. Cannot determine best structure.")


    # Save csv of ranking scores
    if ranking_scores:
        ranking_csv_path = os.path.join(output_dir, 'ranking_scores.csv')
        print(f"Writing ranking scores to {ranking_csv_path}")
        with open(ranking_csv_path, 'wt', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['seed', 'sample', 'ranking_score'])
            writer.writerows(ranking_scores)
    else:
        print(f"No ranking scores generated for job {job_name}.")



def load_processed_fold_input(
    json_path: pathlib.Path
) -> folding_input.Input:
    """Loads the processed fold input JSON."""
    print(f"Loading processed fold input from: {json_path}")
    return next(folding_input.load_fold_inputs_from_path(json_path))

def load_featurised_data(
    pt_path: pathlib.Path
) -> features.BatchDict:
    """Loads the featurised data saved by preprocess.py."""
    print(f"Loading featurised data from: {pt_path}")
    # Load directly to CPU, assuming structure extraction doesn't need GPU
    data = torch.load(pt_path, map_location='cpu', weights_only=False)
    # Convert tensors to numpy if needed by extract_structures (it usually expects numpy)
    data_np = pytree.tree_map_only(torch.Tensor, lambda x: x.numpy(), data)
    # return data_np
    # Keep as tensors if extract_structures handles them (less common)
    return data # Assuming extract_structures might handle tensors or conversion internally


def load_raw_result(
    pt_path: pathlib.Path
) -> ModelResult:
    """Loads the raw result dictionary saved by inference.py."""
    print(f"Loading raw results from: {pt_path}")
    # Load data saved by torch.save (could be numpy dict or tensor dict)
    result = torch.load(pt_path, weights_only=False, map_location='cpu')

    # If result was saved as numpy arrays via torch.save, it should load correctly.
    # If it was saved using np.savez, use: result = dict(np.load(pt_path))

    # Ensure __identifier__ is bytes, as expected by downstream functions
    if '__identifier__' in result and isinstance(result['__identifier__'], np.ndarray):
        result['__identifier__'] = result['__identifier__'].tobytes()
    elif '__identifier__' not in result:
         print(f"Warning: '__identifier__' not found in raw result file {pt_path}. Adding placeholder.")
         result['__identifier__'] = b'unknown_identifier_loaded' # Add placeholder if missing

    return result


def main(_):
    if _INTERMEDIATE_INPUT_DIR.value is None:
        raise ValueError('--intermediate_input_dir must be specified.')
    if _RAW_RESULTS_INPUT_DIR.value is None:
        raise ValueError('--raw_results_input_dir must be specified.')
    if _FINAL_OUTPUT_DIR.value is None:
        raise ValueError('--final_output_dir must be specified.')

    # Make sure we can create the base output directory
    try:
        os.makedirs(_FINAL_OUTPUT_DIR.value, exist_ok=True)
    except OSError as e:
        print(f'Failed to create final output directory {_FINAL_OUTPUT_DIR.value}: {e}')
        raise

    # --- Find Input Data Directories ---
    intermediate_dir = pathlib.Path(_INTERMEDIATE_INPUT_DIR.value)
    raw_results_dir = pathlib.Path(_RAW_RESULTS_INPUT_DIR.value)

    if not intermediate_dir.is_dir():
        raise FileNotFoundError(f"Intermediate input directory not found: {intermediate_dir}")
    if not raw_results_dir.is_dir():
        raise FileNotFoundError(f"Raw results input directory not found: {raw_results_dir}")

    # Find job subdirectories (assuming they exist in both input dirs with the same names)
    job_names = sorted([d.name for d in intermediate_dir.iterdir() if d.is_dir()])
    if not job_names:
        raise FileNotFoundError(f"No job subdirectories found in {intermediate_dir}")

    print(f"Found {len(job_names)} job(s) to postprocess.")

    # --- Process Each Job ---
    total_postprocessing_time = 0
    processed_jobs_count = 0

    for job_name in job_names:
        print(f"\nPostprocessing job: {job_name}")
        start_time = time.time()

        job_intermediate_dir = intermediate_dir / job_name
        job_raw_results_dir = raw_results_dir / job_name
        job_final_output_dir = pathlib.Path(_FINAL_OUTPUT_DIR.value) / job_name

        if not job_raw_results_dir.is_dir():
            print(f"Warning: Raw results directory not found for job {job_name} at {job_raw_results_dir}. Skipping.")
            continue

        # Load the processed fold input JSON (contains seeds, name etc.)
        processed_input_json_path = job_intermediate_dir / f'{job_name}_processed_input.json'
        if not processed_input_json_path.exists():
             print(f"Warning: Processed input JSON not found for job {job_name} at {processed_input_json_path}. Cannot get target name. Skipping.")
             continue
        # try:
        processed_fold_input = load_processed_fold_input(processed_input_json_path)
        target_name = processed_fold_input.name # Get original name
        # except Exception as e:
        #      print(f"Error loading processed input JSON {processed_input_json_path}: {e}. Skipping job {job_name}.")
        #      continue


        all_results_for_job: list[ResultsForSeed] = []

        # Iterate through seeds defined in the processed input
        if not processed_fold_input.rng_seeds:
            print(f"Warning: No RNG seeds found in processed input for job {job_name}. Skipping.")
            continue

        for seed in processed_fold_input.rng_seeds:
            print(f"--- Processing results for seed {seed} ---")

            # Load the corresponding featurised data (batch context)
            featurised_data_path = job_intermediate_dir / f'{job_name}_seed_{seed}_featurised.pt'
            if not featurised_data_path.exists():
                print(f"Warning: Featurised data file not found for seed {seed}: {featurised_data_path}. Skipping seed.")
                continue
            
            batch_data = load_featurised_data(featurised_data_path)
            # Ensure batch_data is compatible with extract_structures (e.g., numpy if needed)
            # Example conversion if needed:
            #  batch_data = torch.utils._pytree.tree_map_only(torch.Tensor, lambda x: x.numpy(), batch_data)



            # Load the raw model result
            raw_result_path = job_raw_results_dir / f'{job_name}_seed_{seed}_raw_results.pt'
            if not raw_result_path.exists():
                print(f"Warning: Raw result file not found for seed {seed}: {raw_result_path}. Skipping seed.")
                continue

            raw_result = load_raw_result(raw_result_path)

            # Extract structures (one per diffusion sample)
            print(f"Extracting structures for seed {seed}...")
            extract_start_time = time.time()
            # try:
            # Pass the numpy batch data and the loaded raw result dict
            inference_results_for_seed: list[InferenceResult] = extract_structures(
                batch=batch_data, result=raw_result, target_name=target_name
            )
            print(f"Structure extraction for seed {seed} took {time.time() - extract_start_time:.2f} seconds.")

            if not inference_results_for_seed:
                print(f"Warning: No structures extracted for seed {seed}.")
                continue # Skip adding empty results

            all_results_for_job.append(
                ResultsForSeed(
                    seed=seed,
                    inference_results=inference_results_for_seed,
                )
            )
            # finally: pass
            # except Exception as e:
            #     print(f"Error during structure extraction for seed {seed}: {e}. Skipping seed.")
            #     import traceback
            #     traceback.print_exc() # Optional: print full traceback for debugging
            #     continue


        # Write final outputs for this job if any results were successfully processed
        if all_results_for_job:
             print(f"Writing final outputs for job {job_name} to {job_final_output_dir}")
             write_outputs(
                 all_inference_results=all_results_for_job,
                 output_dir=job_final_output_dir,
                 job_name=job_name, # Use sanitised name for output files
             )
             job_time = time.time() - start_time
             total_postprocessing_time += job_time
             processed_jobs_count += 1
             print(f"Finished postprocessing job {job_name} in {job_time:.2f} seconds.")
        else:
             print(f"Skipping output writing for job {job_name} as no valid results were processed.")


    print(f"\nFinished postprocessing {processed_jobs_count} / {len(job_names)} jobs.")
    if processed_jobs_count > 0:
        print(f"Total postprocessing time: {total_postprocessing_time:.2f} seconds.")
        print(f"Average time per processed job: {total_postprocessing_time / processed_jobs_count:.2f} seconds.")
    print(f"Final outputs saved in subdirectories under: {_FINAL_OUTPUT_DIR.value}")


if __name__ == '__main__':
    flags.mark_flags_as_required([
        'intermediate_input_dir',
        'raw_results_input_dir',
        'final_output_dir',
    ])
    app.run(main)