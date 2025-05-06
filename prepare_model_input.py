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

"""Part 1: AlphaFold 3 Data Preprocessing Script."""

import multiprocessing
import os
import pathlib
import shutil
import string
import time
from collections.abc import Sequence

from absl import app
from absl import flags
from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.data import featurisation
from alphafold3.data import pipeline
from alphafold3.model.components import utils

import torch # For saving
import torch.utils._pytree as pytree

_HOME_DIR = pathlib.Path(os.environ.get('HOME'))
DEFAULT_DB_DIR = _HOME_DIR / 'public_databases'

# --- Input Flags ---
_JSON_PATH = flags.DEFINE_string(
    'json_path',
    None,
    'Path to the input JSON file.',
)
_INPUT_DIR = flags.DEFINE_string(
    'input_dir',
    None,
    'Path to the directory containing input JSON files.',
)

# --- Output Flags ---
_INTERMEDIATE_OUTPUT_DIR = flags.DEFINE_string(
    'intermediate_output_dir',
    None,
    'Path to a directory where the intermediate preprocessed results will be saved.',
)

# --- Data Pipeline Control Flags ---
_RUN_DATA_PIPELINE = flags.DEFINE_bool(
    'run_data_pipeline',
    False,
    'Whether to run the data pipeline on the fold inputs.',
)

_DB_DIR = flags.DEFINE_string(
    'db_dir',
    DEFAULT_DB_DIR.as_posix(),
    'Path to the directory containing the databases.',
)
_JACKHMMER_BINARY_PATH = flags.DEFINE_string(
    'jackhmmer_binary_path',
    shutil.which('jackhmmer'),
    'Path to the Jackhmmer binary.',
)
_NHMMER_BINARY_PATH = flags.DEFINE_string(
    'nhmmer_binary_path',
    shutil.which('nhmmer'),
    'Path to the Nhmmer binary.',
)
_HMMALIGN_BINARY_PATH = flags.DEFINE_string(
    'hmmalign_binary_path',
    shutil.which('hmmalign'),
    'Path to the Hmmalign binary.',
)
_HMMSEARCH_BINARY_PATH = flags.DEFINE_string(
    'hmmsearch_binary_path',
    shutil.which('hmmsearch'),
    'Path to the Hmmsearch binary.',
)
_HMMBUILD_BINARY_PATH = flags.DEFINE_string(
    'hmmbuild_binary_path',
    shutil.which('hmmbuild'),
    'Path to the Hmmbuild binary.',
)
_SMALL_BFD_DATABASE_PATH = flags.DEFINE_string(
    'small_bfd_database_path',
    '${DB_DIR}/bfd-first_non_consensus_sequences.fasta',
    'Small BFD database path, used for protein MSA search.',
)
_MGNIFY_DATABASE_PATH = flags.DEFINE_string(
    'mgnify_database_path',
    '${DB_DIR}/mgy_clusters_2022_05.fa',
    'Mgnify database path, used for protein MSA search.',
)
_UNIPROT_CLUSTER_ANNOT_DATABASE_PATH = flags.DEFINE_string(
    'uniprot_cluster_annot_database_path',
    '${DB_DIR}/uniprot_all_2021_04.fa',
    'UniProt database path, used for protein paired MSA search.',
)
_UNIREF90_DATABASE_PATH = flags.DEFINE_string(
    'uniref90_database_path',
    '${DB_DIR}/uniref90_2022_05.fa',
    'UniRef90 database path, used for MSA search. The MSA obtained by '
    'searching it is used to construct the profile for template search.',
)
_NTRNA_DATABASE_PATH = flags.DEFINE_string(
    'ntrna_database_path',
    '${DB_DIR}/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta',
    'NT-RNA database path, used for RNA MSA search.',
)
_RFAM_DATABASE_PATH = flags.DEFINE_string(
    'rfam_database_path',
    '${DB_DIR}/rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta',
    'Rfam database path, used for RNA MSA search.',
)
_RNA_CENTRAL_DATABASE_PATH = flags.DEFINE_string(
    'rna_central_database_path',
    '${DB_DIR}/rnacentral_active_seq_id_90_cov_80_linclust.fasta',
    'RNAcentral database path, used for RNA MSA search.',
)
_PDB_DATABASE_PATH = flags.DEFINE_string(
    'pdb_database_path',
    '${DB_DIR}/pdb_2022_09_28_mmcif_files.tar',
    'PDB database directory with mmCIF files path, used for template search.',
)
_SEQRES_DATABASE_PATH = flags.DEFINE_string(
    'seqres_database_path',
    '${DB_DIR}/pdb_seqres_2022_09_28.fasta',
    'PDB sequence database path, used for template search.',
)
_JACKHMMER_N_CPU = flags.DEFINE_integer(
    'jackhmmer_n_cpu',
    min(multiprocessing.cpu_count(), 8),
    'Number of CPUs to use for Jackhmmer. Default to min(cpu_count, 8). Going'
    ' beyond 8 CPUs provides very little additional speedup.',
)
_NHMMER_N_CPU = flags.DEFINE_integer(
    'nhmmer_n_cpu',
    min(multiprocessing.cpu_count(), 8),
    'Number of CPUs to use for Nhmmer. Default to min(cpu_count, 8). Going'
    ' beyond 8 CPUs provides very little additional speedup.',
)
# --- Buckets (can influence featurisation padding) ---
_BUCKETS = flags.DEFINE_list(
    'buckets',
    None,
    'Optional bucket sizes (integers) to pad the data to, comma-separated. '
    'If None, calculate appropriate size. Must be increasing. '
    'Example: --buckets=512,1024,2048'
)

def write_processed_fold_input_json(
    fold_input: folding_input.Input,
    output_dir: os.PathLike[str] | str,
) -> None:
    """Writes the processed input JSON (after data pipeline) to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f'{fold_input.sanitised_name()}_processed_input.json')
    print(f"Saving processed input JSON to {output_path}")
    with open(output_path, 'wt') as f:
        f.write(fold_input.to_json())

def save_featurised_example(
    featurised_example: featurisation.features.BatchDict,
    seed: int,
    output_dir: os.PathLike[str] | str,
    sanitised_name: str
) -> None:
    """Saves the featurised example using torch.save."""
    # featurised_example = pytree.tree_map(
    #     torch.from_numpy, utils.remove_invalidly_typed_feats(
    #         featurised_example)
    # )
    # featurised_example = pytree.tree_map_only(
    #     torch.Tensor,
    #     lambda x: x.to(device='cpu'),
    #     featurised_example,
    # )
    # featurised_example['deletion_mean'] = featurised_example['deletion_mean'].to(
    #     dtype=torch.float32)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f'{sanitised_name}_seed_{seed}_featurised.pt')
    print(f"Saving featurised data for seed {seed} to {output_path}")
    torch.save(featurised_example, output_path)


def run_preprocessing(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
):
    """Runs data pipeline and featurisation, saving results."""
    print(f'Processing fold input {fold_input.name}')
    if not fold_input.chains:
        raise ValueError('Fold input has no chains.')

    # --- 1. Run Data Pipeline ---
    if data_pipeline_config is None:
        print('Skipping data pipeline...')
        processed_fold_input = fold_input
    else:
        print('Running data pipeline...')
        pipeline_start_time = time.time()
        processed_fold_input = pipeline.DataPipeline(
            data_pipeline_config).process(fold_input)
        print(
            f'Data pipeline for {fold_input.name} took '
            f' {time.time() - pipeline_start_time:.2f} seconds.'
        )
    
    # Make output directory for this specific input
    job_output_dir = os.path.join(output_dir, fold_input.sanitised_name())
    os.makedirs(job_output_dir, exist_ok=True)
    print(f'Intermediate output directory for this job: {job_output_dir}')

    # Save the processed fold input (contains MSA, templates etc.)
    write_processed_fold_input_json(processed_fold_input, job_output_dir)

    # --- 2. Run Featurisation ---
    print(f'Featurising data for seeds {processed_fold_input.rng_seeds}...')
    featurisation_start_time = time.time()
    ccd = chemical_components.cached_ccd(user_ccd=processed_fold_input.user_ccd)

    # Parse buckets if provided
    bucket_sizes = None
    if _BUCKETS.value:
        try:
            bucket_sizes = sorted([int(b) for b in _BUCKETS.value])
            if not all(isinstance(b, int) and b > 0 for b in bucket_sizes):
                raise ValueError("Buckets must be positive integers.")
            print(f"Using bucket sizes: {bucket_sizes}")
        except ValueError as e:
            raise flags.ValidationError(f"Invalid format for --buckets: {e}")


    featurised_examples = featurisation.featurise_input(
        fold_input=processed_fold_input,
        buckets=bucket_sizes, # Use parsed buckets
        ccd=ccd,
        verbose=True
    )

    print(
        f'Featurising data for seeds {processed_fold_input.rng_seeds} took '
        f' {time.time() - featurisation_start_time:.2f} seconds.'
    )

    # --- 3. Save Featurised Data ---
    for seed, example in zip(processed_fold_input.rng_seeds, featurised_examples):
        save_featurised_example(
            featurised_example=example,
            seed=seed,
            output_dir=job_output_dir,
            sanitised_name=fold_input.sanitised_name()
        )

    print(f'Done preprocessing fold input {fold_input.name}.')


def main(_):
    if _JSON_PATH.value is None == _INPUT_DIR.value is None:
        raise ValueError(
            'Exactly one of --json_path or --input_dir must be specified.'
        )
    if _INTERMEDIATE_OUTPUT_DIR.value is None:
        raise ValueError('--intermediate_output_dir must be specified.')


    if _INPUT_DIR.value is not None:
        fold_inputs = folding_input.load_fold_inputs_from_dir(
            pathlib.Path(_INPUT_DIR.value)
        )
    elif _JSON_PATH.value is not None:
        fold_inputs = folding_input.load_fold_inputs_from_path(
            pathlib.Path(_JSON_PATH.value)
        )
    else:
        # Should be caught by the check above, but adding for safety
        raise AssertionError("Input source not specified correctly.")

    # Make sure we can create the base output directory
    try:
        os.makedirs(_INTERMEDIATE_OUTPUT_DIR.value, exist_ok=True)
    except OSError as e:
        print(f'Failed to create intermediate output directory {_INTERMEDIATE_OUTPUT_DIR.value}: {e}')
        raise

    # Configure Data Pipeline
    if _RUN_DATA_PIPELINE.value:
        def replace_db_dir(x): return string.Template(x).substitute(
            DB_DIR=_DB_DIR.value
        )
        data_pipeline_config = pipeline.DataPipelineConfig(
            jackhmmer_binary_path=_JACKHMMER_BINARY_PATH.value,
            nhmmer_binary_path=_NHMMER_BINARY_PATH.value,
            hmmalign_binary_path=_HMMALIGN_BINARY_PATH.value,
            hmmsearch_binary_path=_HMMSEARCH_BINARY_PATH.value,
            hmmbuild_binary_path=_HMMBUILD_BINARY_PATH.value,
            small_bfd_database_path=replace_db_dir(
                _SMALL_BFD_DATABASE_PATH.value),
            mgnify_database_path=replace_db_dir(_MGNIFY_DATABASE_PATH.value),
            uniprot_cluster_annot_database_path=replace_db_dir(
                _UNIPROT_CLUSTER_ANNOT_DATABASE_PATH.value
            ),
            uniref90_database_path=replace_db_dir(
                _UNIREF90_DATABASE_PATH.value),
            ntrna_database_path=replace_db_dir(_NTRNA_DATABASE_PATH.value),
            rfam_database_path=replace_db_dir(_RFAM_DATABASE_PATH.value),
            rna_central_database_path=replace_db_dir(
                _RNA_CENTRAL_DATABASE_PATH.value
            ),
            pdb_database_path=replace_db_dir(_PDB_DATABASE_PATH.value),
            seqres_database_path=replace_db_dir(_SEQRES_DATABASE_PATH.value),
            jackhmmer_n_cpu=_JACKHMMER_N_CPU.value,
            nhmmer_n_cpu=_NHMMER_N_CPU.value,
        )
    else:
        print('Skipping running the data pipeline.')
        data_pipeline_config = None

    input_count = 0
    print(fold_inputs)
    # print(f'Processing {len(list(fold_inputs))} fold inputs.')
    for fold_input in fold_inputs:
        input_count += 1
        run_preprocessing(
            fold_input=fold_input,
            data_pipeline_config=data_pipeline_config,
            output_dir=_INTERMEDIATE_OUTPUT_DIR.value,
            buckets=_BUCKETS.value # Pass buckets flag value
        )

    print(f'Finished preprocessing {input_count} fold inputs.')


if __name__ == '__main__':
    flags.mark_flags_as_required([
        'intermediate_output_dir',
        # Require either json_path or input_dir (handled in main)
    ])
    app.run(main)