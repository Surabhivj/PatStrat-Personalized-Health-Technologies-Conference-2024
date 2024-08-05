import argparse
from src.infer_grn import InferGRN
from src.infer_ppi import Infer_PPI
from src.infer_mutation_net import Infer_MUT
from src.infer_psn import Infer_PSN

# Set the location of the input data and the desired location of the output files
DATA_DIR = 'data/transcriptomics'
OUTPUT_DIR = '~/results/transcriptomics'

PRIORS_FILE_NAME = 'refnet.tsv.gz'
GOLD_STANDARD_FILE_NAME = 'refnet.tsv.gz'
TF_LIST_FILE_NAME = 'tf_names.tsv'

drug_response_file = "data/drug_response.csv"
prot_expression_file = "data/protein_abundance.csv"
mutation_file = "data/mutations_all_20230202.csv"
gene_expression_file = "data/gene_expression.csv"

# Define a function to run the InferGRN
def run_infer_grn():
    InferGRN(DATA_DIR, PRIORS_FILE_NAME, GOLD_STANDARD_FILE_NAME, TF_LIST_FILE_NAME)

# Define a function to run the Infer_PPI
def run_infer_ppi():
    Infer_PPI(prot_expression_file, drug_response_file)

# Define a function to run the Infer_MUT
def run_infer_mut():
    Infer_MUT(mutation_file, drug_response_file)

# Define a function to run the Infer_PSN
def run_infer_psn():
    Infer_PSN(mutation_file, prot_expression_file, gene_expression_file, drug_response_file, n_neighbors=20)

# Set up argument parsing
def main():
    parser = argparse.ArgumentParser(description="Run inference methods")
    parser.add_argument(
        '--method', 
        choices=['grn', 'ppi', 'mut', 'psn'], 
        required=True, 
        help="Inference method to run: 'grn', 'ppi', 'mut', 'psn'"
    )
    
    args = parser.parse_args()
    
    if args.method == 'grn':
        run_infer_grn()
    elif args.method == 'ppi':
        run_infer_ppi()
    elif args.method == 'mut':
        run_infer_mut()
    elif args.method == 'psn':
        run_infer_psn()

if __name__ == "__main__":
    main()