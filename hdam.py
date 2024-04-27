from typing import List, Dict
import jax.numpy as jnp
from jax import jit, vmap
import jax
import numpy as np
from functools import partial

Pattern = str
DatabaseName = str
DatabaseIdx = int
Results = List[DatabaseName]


def generate_patterns(num_of_patterns, pattern_length, alphabet):
    patterns = []
    for _ in range(num_of_patterns):
        pattern = "".join(np.random.choice(alphabet, size=pattern_length))
        patterns.append(pattern)
    return patterns

'''
@ args:
- database: a jnp.array of shape [database_size, self.pattern_length]
- patterns: a list of patterns, of length self.pattern_length.

@ return:
- int: the number of matching patterns in the database.

@ description:
This function will return the number of matching patterns in the database.
'''
@jit
def compare_patterns_to_database(
    database,
    patterns,
    minimal_matches
) -> int:
    '''
    This function will return the number of matching patterns in the database
    
    Dimensions of database [database_size, self.pattern_length]
    Dimensions of patterns [len(patterns), self.pattern_length]

    The output will be of shape 1
    '''
    # 1. Compare the patterns with the database
    database = jnp.expand_dims(database, axis=1)
    compare_res = jnp.sum(jnp.equal(database, patterns), axis=-1) >= minimal_matches
    # compare_res shape is [dataset_size, len(patterns)]

    # 2. For each pattern, check if it matches in the database or not, can't have more than 1 match
    patterns_match_database = jnp.sum(compare_res, axis=0, dtype=jnp.int32) > 0
    return jnp.sum(patterns_match_database, dtype=jnp.int32)


class HDAM(object):
    # The class of Hamming Distnace tolerant Associative Memory (HDAM)
    def __init__(
        self,
        pattern_length: int,
        alphabet: List[str],
    ) -> None:
        self.pattern_length = pattern_length
        self.alphabet = {
            char: i for i, char in enumerate(alphabet)
        }
        self.threshold = 0
        # self.database_name_to_idx: Dict[DatabaseName, DatabaseIdx] = {}
        self.database_idx_to_name: Dict[DatabaseIdx, DatabaseName] = {}
        self.pattern_match_threshold = 0
        self.databases: List[jnp.array] = []
    
        self.batch = 256

        self._batched_compare_patterns_to_database = vmap(
            compare_patterns_to_database,
            in_axes=[None, 0, None],
            out_axes=0
        )


    '''
    @ args:
    - database_name: the database name to save
    - list_of_patterns: is a list of patterns, of the same length (self.pattern_length) and over the same alphabet.
    These patterns will constitute the database identified by the database_name.
    
    @ return:
    - None
    
    @ description:
    This function will change the state of the Hamming Distnace tolerant Associative Memory (HDAM) object,
    by adding to the list of databases another database.
    '''
    def save(
        self,
        database_name: DatabaseName,
        list_of_patterns: List[Pattern]
    ) -> None:
        # First construct a numpy array using a for loop, of shape (len(list_of_patterns), self.pattern_length)
        # Then store the numpy array in the databases dictionary
        database = []
        for pattern in list_of_patterns:
            if len(pattern) != self.pattern_length:
                raise ValueError(f"Pattern length is not equal to the pattern length of the HDAM object")
            
            char_list = [self.alphabet.get(char, 4) for char in pattern]
            database.append(char_list)
        self.databases.append(jnp.array(database, dtype=jnp.int8))
        self.database_idx_to_name[len(self.databases) - 1] = database_name
        
    '''
    @ args:
    - patterns: a list of search_patterns of the same length (which  must be greater than the pattern length of the HDAM object) and over the same alphabet.
    - query_pattern_length: the length of the search patterns.

    @ return:
    - Results: a List of DatabaseNames, where each  element describe the most likely database that contains the search pattern.
    The order of the elements in the list should be the same as the order of the search patterns in the input list.

    @ description:
    This function will search for the patterns in the databases. It will return a list of database names, where each element
    in the list corresponds to the most likely database that contains the search pattern.
    '''
    def search(
        self,
        search_patterns: List[Pattern],
        search_pattern_length: int,
    ) -> Results:
        '''
        Transform the pattern to numpy arrays, search for it in the databases.
        Every pattern is transformed to an array of size
        (query_pattern_length - self.pattern_length + 1) x self.pattern_length
        we batch the search for all patterns in the patterns dictionary
        to create a batched search of size
        len(patterns) x (query_pattern_length - self.pattern_length + 1) x self.pattern_length
        '''
        # 1. Transform the patterns to numpy arrays
        shatered_search_patterns: jnp.array = self._shater_search_patterns(
            search_patterns, 
            search_pattern_length
        )
        # should be of size:
        # len(patterns) x (query_pattern_length - self.pattern_length + 1) x self.pattern_length

        # 2. Search for the patterns in the databases
        index_results = self._search_databases(shatered_search_patterns)
        # Will return a jnp.array of size len(patterns) x len(datasets) specifying
        # the number of matches found for each pattern in each database

        # 3. Transform the indexes to database names
        return self.transform_indexes_to_database_names(index_results)

    '''
    Changes the threshold of the HDAM object, should cause a re-compilation of all jit functions that make use of its value.
    '''
    def set_threshold(
        self,
        new_thresh: int
    ) -> bool: # returns True in success and False in failure
        if new_thresh < 0 or new_thresh > self.pattern_length:
            return False
        self.threshold = new_thresh
        return True
    

    def set_pattern_match_threshold(
        self,
        new_thresh: int
    ) -> bool:
        if new_thresh < 0:
            return False
        self.pattern_match_threshold = new_thresh
        return True
    
    ###################################
    ###### Private functions ##########
    ###################################
    '''
    @ args:
    - patterns: a dictionary where the key is a pattern and the value is the database name.
    - query_pattern_length: the length of the search patterns.

    @ return:
    - sahtered_search_patterns: a jnp.array where we return for each search_pattern, all of its substrings of length self.pattern_length.
    The shape of the array should be (len(patterns), query_pattern_length - self.pattern_length + 1, self.pattern_length)
    '''
    def _shater_search_patterns(
        self,
        search_patterns: List[Pattern],
        search_pattern_length: int
    ) -> jnp.array:
        shatered_search_patterns = []
        for search_pattern in search_patterns:
            if len(search_pattern) != search_pattern_length:
                raise ValueError(f"Pattern length is not equal to the pattern length of the HDAM object")
            shatered_search_pattern = []
            for i in range(search_pattern_length - self.pattern_length + 1):
                shatered_search_pattern.append([self.alphabet.get(char, 4) for char in search_pattern[i:i+self.pattern_length]])
            shatered_search_patterns.append(shatered_search_pattern)
        return jnp.array(shatered_search_patterns, dtype=jnp.int8)            
    

    '''
    @ args:
    - shatered_search_patterns: a jnp.array that contains the shatered patterns for each search pattern.
    The shape of the array should be (len(patterns), query_pattern_length - self.pattern_length + 1, self.pattern_length)

    @ return:
    - results: a List of indexes, where each element describe the most likely database index that contains the search pattern.
    # If there is no match, the index should be -1.
    '''
    def _search_databases(
        self,
        shatered_search_patterns: jnp.array
    ) ->  List[DatabaseName]:
        '''
        This function batch search_patterns and will search against all of the databases. It will keep record of the maximal database
        '''
        # batch the shatered_search_patterns and process them in batches
        # with batched_compare_patterns_to_database
        results = []

        for batch_idx in range(0, shatered_search_patterns.shape[0], self.batch):
            batched_search_patterns = shatered_search_patterns[batch_idx:batch_idx+self.batch]
            # for each batch, compare the patterns to all databases
            databases_num_of_matches = []
            for idx, database in enumerate(self.databases):
                # compare the patterns to the database
                batch_cpm_res = self._batched_compare_patterns_to_database(
                    database,
                    batched_search_patterns,
                    self.pattern_length - self.threshold
                )
                databases_num_of_matches.append(batch_cpm_res)
            databases_num_of_matches = jnp.stack(databases_num_of_matches, axis=0)
            batch_maximal_num_of_matches = jnp.max(databases_num_of_matches, axis=0)
            batch_maximal_num_of_matches = jnp.where(
                batch_maximal_num_of_matches <= self.pattern_match_threshold,
                -1,
                batch_maximal_num_of_matches
            )
            num_of_maximal_appearances = jnp.sum(
                jnp.equal(
                    databases_num_of_matches,
                    batch_maximal_num_of_matches
                ),
                axis=0
            )
            match_index = jnp.argmax(databases_num_of_matches, axis=0)
            match_index = jnp.where(
                batch_maximal_num_of_matches == -1,
                -1,
                match_index
            )
            batch_maximal_database_idx = jnp.where(
                num_of_maximal_appearances > 1,
                len(self.databases),
                match_index
            )

            results += batch_maximal_database_idx.tolist()

        return results
    

    def transform_indexes_to_database_names(
        self,
        results: List[DatabaseIdx]
    ) -> Results:
        name_results = []
        for idx_result in results:
            if idx_result == -1:
                name_results.append("None")
            elif idx_result == len(self.databases):
                name_results.append("Multiple")
            else:
                name_results.append(self.database_idx_to_name[idx_result])
        return name_results