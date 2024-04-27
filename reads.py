import glob
import os
import random
from typing import List
from utils import getKmersFromRead, getReadsFromFastq

class Reads(object):
    def __init__(self,
                 datadir,
                 read_length):
        self.read_length = read_length
        self.datadir = os.path.abspath(datadir)
        self.read_dir = os.path.abspath(datadir + "/reads/")
        self.platforms = {
            "illumina": {
                "sequencer_command": lambda genome_filepath, read_len, reads_filepath, accuracy, coverage:\
                    f"art_illumina -sam -i {genome_filepath} -l {read_len} -ss HS25 -f {coverage} -o {reads_filepath} > /dev/null",
                "read_file_suffix": ".fq"
            },
            "pacbio0": {
                "sequencer_command": lambda genome_filepath, read_len, reads_filepath, accuracy, coverage:\
                    f"pbsim {genome_filepath} --hmm_model pbsim_models/P6C4.model --prefix {reads_filepath} \
                    --depth {coverage} --length-mean {read_len} --length-sd 0 --accuracy-mean 1  > /dev/null",
                "read_file_suffix": ".fastq"
            },
            "pacbio5": {
                "sequencer_command": lambda genome_filepath, read_len, reads_filepath, accuracy, coverage:\
                    f"pbsim {genome_filepath} --hmm_model pbsim_models/P6C4.model --prefix {reads_filepath} \
                    --depth {coverage} --length-mean {read_len} --length-sd 0 --accuracy-mean 0.95 > /dev/null",
                "read_file_suffix": ".fastq"
            },
            "pacbio10": {
                "sequencer_command": lambda genome_filepath, read_len, reads_filepath, accuracy, coverage:\
                    f"pbsim {genome_filepath} --hmm_model pbsim_models/P6C4.model --prefix {reads_filepath} \
                    --depth {coverage} --length-mean {read_len} --length-sd 0 --accuracy-mean 0.9 > /dev/null",
                "read_file_suffix": ".fastq"
            },
            "pacbio15": {
                "sequencer_command": lambda genome_filepath, read_len, reads_filepath, accuracy, coverage:\
                    f"pbsim {genome_filepath} --hmm_model pbsim_models/P6C4.model --prefix {reads_filepath} \
                    --depth {coverage} --length-mean {read_len} --length-sd 0 --accuracy-mean 0.85 > /dev/null",
                "read_file_suffix": ".fastq"
            },
            "roche": {
                "sequencer_command": lambda genome_filepath, read_len, reads_filepath, accuracy, coverage:\
                    f"art_454 -t -B {genome_filepath} {reads_filepath} {coverage} > /dev/null",
                "read_file_suffix": ".fq"
            },
        }
        os.makedirs(self.read_dir, exist_ok=True)

    
    def getPlatforms(self):
        return list(self.platforms.keys())

    def getReadFiles(self,
                     platform,
                     genome_filepath) -> List[str]:
        if not self._isPlatformValid(platform):
            return []
        reads_filepath = f"{self.read_dir}/{platform}-{genome_filepath.split('/')[-1].split('.')[0]}"
        if not glob.glob(f"{reads_filepath}*{self.platforms[platform]['read_file_suffix']}"):
            os.system(self.platforms[platform]['sequencer_command'](genome_filepath, self.read_length, reads_filepath, 0.9, 1000))
        return glob.glob(f"{reads_filepath}*{self.platforms[platform]['read_file_suffix']}")

    def getReads(self,
                 platform,
                 genome_filepath,
                 reads_num = 100) -> List[str]:
        if not self._isPlatformValid(platform):
            return []
        else:
            read_filepaths = self.getReadFiles(platform, genome_filepath)
            reads = []
            for file in read_filepaths:
                reads += getReadsFromFastq(file)
            return random.sample(reads, min(reads_num, len(reads)))
        
    def getKmers(
        self,
        platform: str,
        genome_filepath,
        kmer_size,
        kmers_num = 10000) -> List[str]:
            kmers = []
            for read in self.getReads(platform=platform, genome_filepath=genome_filepath):
                kmers += getKmersFromRead(read, kmer_size=kmer_size)
            return random.sample(kmers, min(kmers_num, len(kmers)))

    def _isPlatformValid(self, platform):
        if not platform in self.platforms:
            print(f"There is no existing platform named {platform}. Please run Reads.getPlatforms() to get available platforms.")
            return False
        return True