# %%
import numpy as np
from Bio import SeqIO
import json

from hdam import HDAM
from reads import Reads

# %%
alphabet  = ['A', 'C', 'G', 'T', 'N']

# %%
# generate random patterns of size k over the alphabet {A,C,G,T}
def generate_patterns(num_of_patterns, pattern_length):
    patterns = []
    for _ in range(num_of_patterns):
        pattern = "".join(np.random.choice(["A", "C", "G", "T"], size=pattern_length))
        patterns.append(pattern)
    return patterns

# %%
dirpath = "data/genomes/"
database_viruses = [
    "chickenpox",
    "dengue",
    "ebola",
    "herpes",
    "kyasanur",
    "marburg",
    "measles",
    "sars-cov-2"    
]

other_viruses = [
    "crimea-congo",
    "hantavirus",
    "influenza",
    "junin",
    "lassa",
    "machupo",
    "papiloma",
    "rotavirus",
]

kmer_length = 21
number_of_reads = 1000
read_length = 150

# %%
references_filepath = []
for idx, reference in enumerate(database_viruses):
    references_filepath.append(dirpath + reference + ".fna")

other_filepath = []
for idx, reference in enumerate(other_viruses):
    other_filepath.append(dirpath + reference + ".fna")

# %%
################ Sanity Functions ################

# extract genome from fasta, accounting for all contigs, output a list of contings
def extract_genome_from_fasta(fasta_file):
    genome = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        genome.append(str(record.seq).upper())
    return genome


# return the length of the longest matching string between a reference and a read
def longest_match(reference, read):
    max_match = 0
    for i in range(len(read)):
        for j in range(len(reference)):
            match = 0
            while i + match < len(read) and j + match < len(reference) and read[i + match] == reference[j + match]:
                match += 1
            max_match = max(max_match, match)
    return max_match

# %%
def extract_kmers_from_fasta(fasta_file, kmer_length, overall_size):
    kmers = []

    # Parse the FASTA file
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(record.seq).upper()
        
        # Generate kmers for each contig
        for i in range(len(sequence) - kmer_length + 1):
            kmer = sequence[i:i + kmer_length]
            kmers.append(kmer)

    stride = len(kmers) // overall_size
    if stride == 0:
        return kmers
    else:
        return [kmer for idx, kmer in enumerate(kmers) if idx % stride == 0]

# %%
reads = Reads("data", read_length=read_length)
hdam = HDAM(pattern_length=kmer_length, alphabet=alphabet)


database_size = 2048
for virus, reference in zip(database_viruses, references_filepath):
    hdam.save(virus, extract_kmers_from_fasta(reference, kmer_length, database_size))

# %%
def generate_reverse_complements(sequences):
    reverse_complements = []
    for sequence in sequences:
        complement = {"A": "T", "C": "G", "G": "C", "T": "A"}
        reverse_complements.append("".join([complement.get(base, "N") for base in sequence[::-1]]))
    return reverse_complements

# %%
'''
The structure of the raw results should be as follows:
raw_results[platform][threshold][virus]["results_type, i.e. rc or ord"] = List[viruses] || None
'''
def add_raw_result(raw_results, platform, threshold, virus, results_type, results):
    curr_res_ptr = raw_results
    if not platform in raw_results:
        curr_res_ptr[platform] = {}
    curr_res_ptr = raw_results[platform]
    if not threshold in curr_res_ptr:
        curr_res_ptr[threshold] = {}
    curr_res_ptr = curr_res_ptr[threshold]
    if not virus in curr_res_ptr:
        curr_res_ptr[virus] = {}
    curr_res_ptr = curr_res_ptr[virus]
    curr_res_ptr[results_type] = results
    return raw_results

def write_raw_results_to_file(raw_results, filename):
    with open(filename, "w") as f:
        json.dump(raw_results, f)

def read_raw_results_from_file(filename):
    with open(filename, "r") as f:
        return json.load(f)

# %%
def preprocess_reads(reads, read_length):
    preprocessed = []
    for read in reads:
        if len(read) < read_length:
            continue
        else:
            preprocessed.append(read[:read_length])
    return preprocessed

# %%
platforms = ["pacbio0", "pacbio5", "pacbio10", "pacbio15"]
threshold = 0
raw_results = {}

for platform in platforms:
    for threshold in range(0, 22):
        hdam.set_threshold(threshold)
        for virus, reference in zip(database_viruses + other_viruses, references_filepath + other_filepath):
            search_patterns = reads.getReads(platform, reference, reads_num=number_of_reads)
            search_patterns = preprocess_reads(search_patterns, read_length)
            rc_search_patterns = generate_reverse_complements(search_patterns)
            results = hdam.search(search_patterns, read_length)
            rc_results = hdam.search(rc_search_patterns, read_length)
            add_raw_result(raw_results, platform, threshold, virus, "reads", results)
            add_raw_result(raw_results, platform, threshold, virus, "rc_reads", rc_results)
            write_raw_results_to_file(raw_results, "raw_results.json")
            print(f"Finished searching for {virus} with platform {platform} and threshold {threshold}", flush=True)



