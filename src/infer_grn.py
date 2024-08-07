# Load modules
from inferelator import inferelator_workflow, inferelator_verbose_level, MPControl, CrossValidationManager
import os
import fnmatch
from multiprocessing import Pool, cpu_count


# Set verbosity level to "Talky"
inferelator_verbose_level(1)


def set_up_workflow(wkf, DATA_DIR, OUTPUT_DIR, TF_LIST_FILE_NAME, PRIORS_FILE_NAME, GOLD_STANDARD_FILE_NAME, EXPRESSION_FILE_NAME):
    wkf.set_file_paths(input_dir=DATA_DIR,
                       output_dir=OUTPUT_DIR,
                       tf_names_file=TF_LIST_FILE_NAME,
                       priors_file=PRIORS_FILE_NAME,
                       gold_standard_file=GOLD_STANDARD_FILE_NAME)
    wkf.set_expression_file(tsv=EXPRESSION_FILE_NAME)
    wkf.set_file_properties(expression_matrix_columns_are_genes=False)
    wkf.set_run_parameters(num_bootstraps=5)
    wkf.set_crossvalidation_parameters(split_gold_standard_for_crossvalidation=True, cv_split_ratio=0.2)
    return wkf

def process_file(EXPRESSION_FILE_NAME, DATA_DIR, PRIORS_FILE_NAME, GOLD_STANDARD_FILE_NAME, TF_LIST_FILE_NAME, n_cores):
    outfile = EXPRESSION_FILE_NAME.split('.')[0]
    OUTPUT_DIR = os.path.join('/home/surabhi/Documents/PatStrat-Personalized-Health-Technologies-Conference-2024/results/transcriptomics', outfile)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    MPControl.set_multiprocess_engine("multiprocessing")
    MPControl.client.processes = n_cores
    MPControl.connect()

    CV_SEEDS = list(range(10, 15))

    worker = inferelator_workflow(regression="bbsr", workflow="tfa")
    worker = set_up_workflow(worker, DATA_DIR, OUTPUT_DIR, TF_LIST_FILE_NAME, PRIORS_FILE_NAME, GOLD_STANDARD_FILE_NAME, EXPRESSION_FILE_NAME)
    worker.append_to_path("output_dir", "bbsr")

    cv_wrap = CrossValidationManager(worker)
    cv_wrap.add_gridsearch_parameter('random_seed', CV_SEEDS)
    cv_wrap.run()

    worker = inferelator_workflow(regression="bbsr", workflow="tfa")
    worker = set_up_workflow(worker, DATA_DIR, OUTPUT_DIR, TF_LIST_FILE_NAME, PRIORS_FILE_NAME, GOLD_STANDARD_FILE_NAME, EXPRESSION_FILE_NAME)
    worker.append_to_path('output_dir', 'final')
    worker.set_crossvalidation_parameters(split_gold_standard_for_crossvalidation=False, cv_split_ratio=None)
    worker.set_run_parameters(num_bootstraps=20, random_seed=100)

    final_network_results = worker.run()
    return final_network_results

def InferGRN(DATA_DIR, PRIORS_FILE_NAME, GOLD_STANDARD_FILE_NAME, TF_LIST_FILE_NAME):
    pattern = "*_gene_expression.tsv.gz"
    all_files = os.listdir(DATA_DIR)
    matched_files = fnmatch.filter(all_files, pattern)

    n_cores = cpu_count()
    with Pool(processes=n_cores) as pool:
        results = pool.starmap(process_file, [(f, DATA_DIR, PRIORS_FILE_NAME, GOLD_STANDARD_FILE_NAME, TF_LIST_FILE_NAME, n_cores // len(matched_files)) for f in matched_files])
    
    return results