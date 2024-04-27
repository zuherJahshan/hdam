import sys

def getReadsFromFastq(
        fastq_name
):
    with open(fastq_name, 'r') as f:
        return [line for idx, line in enumerate(f.read().split("\n")) if idx % 4 == 1]

def getKmersFromRead(
        s,
        kmer_size
    ):
        if kmer_size > len(s):
            return []

        subsequences = [s[i:i+kmer_size] for i in range(len(s) - kmer_size + 1)]
        return subsequences

def print_progress_bar(opening_str = "progress is", progress = 0):
    if progress < 0:
        progress = 0
    elif progress > 1:
        progress = 1

    filled_length = int(progress * 50)
    filled_chars = '█' * filled_length
    empty_chars = '-' * (50 - filled_length)
    
    progress_percentage = round(progress * 100, 2)
    progress_bar = f"{opening_str}: |{filled_chars}{empty_chars}| {progress_percentage}%"
    
    sys.stdout.write('\r' + progress_bar)
    sys.stdout.flush()

    if progress == 1:
        print()  # Move to the next line when the progress is complete
