from pathlib import Path
from torch.utils.data import IterableDataset
from typing import Optional, List, Union, Tuple
import numpy as np
import uproot
import pandas as pd 
import fnmatch
from tqdm import tqdm
import torch

from mach3pythonutils.utils.utils import (
    get_styled_logger, 
    log_section_header, 
    log_success, 
    log_debug,
    log_warning,
    log_info
)

TORCH_RETURN_TYPE=Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[torch.Tensor, torch.Tensor]]

class ROOTDatasetError(Exception):
    """Base exception class for ROOTDataset errors."""
    pass


class ROOTFileNotFoundError(ROOTDatasetError):
    """Raised when the specified ROOT file cannot be found or opened."""
    pass


class TreeNotFoundError(ROOTDatasetError):
    """Raised when the specified tree is not found in the ROOT file."""
    pass


class BranchNotFoundError(ROOTDatasetError):
    """Raised when one or more specified branches are not found in the tree."""
    pass


class EmptyDatasetError(ROOTDatasetError):
    """Raised when the dataset has no entries after applying cuts or filters."""
    pass


class InvalidIndexError(ROOTDatasetError):
    """Raised when an invalid index is used to access dataset entries."""
    pass


class InvalidCutError(ROOTDatasetError):
    """Raised when invalid cuts are applied to the tree."""
    pass

class ROOTDataset(IterableDataset):
    '''
    Basic PyTorch Dataset for reading ROOT files using uproot.
    Supports wildcard patterns in branch names (e.g., 'param_*', '*_weight').
    
    Efficient Entry Access Methods:
    1. Single entry: dataset[5]
    2. Slice: dataset[10:20]
    3. Multiple specific entries (FAST): dataset.get_specific_entries([1, 4, 6, 7])
    
    The get_specific_entries() method uses uproot's fancy indexing which is much more
    efficient than looping through individual entries. This is equivalent to 
    tree[1,4,6,7] in ROOT terminology but without the performance penalty of loops.
    
    Example:
        # Instead of slow loops:
        data_list = []
        for i in [1, 4, 6, 7]:
            data_list.append(dataset[i])
        
        # Use fast fancy indexing:
        data, labels = dataset.get_specific_entries([1, 4, 6, 7])
    '''

    
    def __init__(self, root_file_path: str | Path, tree_name: str, branches: List[str], labels: List[str], cuts: Optional[List]=None, only_unique_entries: bool = True, nchunks: int = 10, tensor_mode=True):
        """
        ROOTDataset constructor.
        :param root_file_path: Path to the ROOT file.   
        :type root_file_path: str | Path
        :param tree_name: Name of the TTree in the ROOT file.
        :type tree_name: str
        :param branches: List of branch names or wildcard patterns to read.
        :type branches: List[str]
        :param labels: List of label branch names or wildcard patterns.
        :type labels: List[str]
        :param cuts: Optional cuts to apply when reading the tree.
        :type cuts: Optional[List]
        :param only_unique_entries: If True, branches with unique values are kept.
        :type only_unique_entries: bool
        :param nchunks: Number of chunks to use for caching unique entries, defaults to 10.
        :type nchunks: int
        """
        self.file_path = root_file_path
        self.tree_name = tree_name
        self.cuts = cuts
        
        self._logger = get_styled_logger(self.__class__.__name__)
        log_section_header(self._logger, "Initializing ROOTDataset")
        # Get metadata in main process and resolve wildcards
        log_info(self._logger, f"Opening ROOT file: {self.file_path}")
        
        try:
            file = uproot.open(self.file_path)
        except FileNotFoundError:
            raise ROOTFileNotFoundError(f"ROOT file not found: {self.file_path}")
        except Exception as e:
            raise ROOTFileNotFoundError(f"Failed to open ROOT file {self.file_path}: {e}")
        
        try:
            tree = file[self.tree_name]
        except KeyError:
            raise TreeNotFoundError(f"Tree '{self.tree_name}' not found in file {self.file_path}")
        
        self._total_entries = tree.num_entries
        log_info(self._logger, f"Tree '{self.tree_name}' has {self._total_entries} entries")
        
        # Resolve wildcard patterns in branches and labels
        self.branches = self._expand_wildcards(tree, branches)
        self.labels = self._expand_wildcards(tree, labels)
        
        # Check if any branches were found
        if not self.branches:
            raise BranchNotFoundError(f"No branches found matching patterns: {branches}")
        if not self.labels:
            raise BranchNotFoundError(f"No label branches found matching patterns: {labels}")
        
        self._entries = np.arange(self._total_entries)
    
        self._tensor_mode = tensor_mode
    
        if only_unique_entries:

            # Need to briefly disable tensor mode for caching unique entries
            tensor_mode_tmp = tensor_mode
            self._tensor_mode = False

            self._entries = self._cache_unique_entries(nchunks=nchunks)
            log_info(self._logger, f"Found {len(self._entries)} unique entries after applying filters ({len(self._entries)/self._total_entries:.2%}% of total)")
            self._total_entries = len(self._entries)
            
            # Can re-enable tensor mode after caching
            self._tensor_mode = tensor_mode_tmp
            
        # Check if dataset is empty after processing
        if self._total_entries == 0:
            raise EmptyDatasetError("Dataset has no entries after applying filters")
        
        log_debug(self._logger, f"Total entries in tree '{self.tree_name}': {self._total_entries}")
        log_debug(self._logger, f"Resolved branches: {self.branches}")
        log_debug(self._logger, f"Resolved labels: {self.labels}")
        
        log_success(self._logger, f"Initialized ROOTDataset with {self._total_entries} entries, branches: {self.branches}, labels: {self.labels}")

    def _cache_unique_entries(self, nchunks: int = 10) -> List[int]:
        """
        Cache unique entries based on their values. NOTE this will only look at consecutive entries in a given chunk. 
        Since the assumed input is MCMC this is probably fine.
        
        :param tree: The uproot TTree object.
        :type tree: uproot.TTree
        :param nchunks: Number of chunks to use for caching unique entries, defaults to 10.
        :type nchunks: int
        
        :return: List of unique entry indices.
        :rtype: List[int]
        
        """
        log_info(self._logger, "Caching unique entries, this will be slow for larger trees...")

        # Loop over tree        
        unique_ids: List[int] = []
        
        chunk_size = max(1, self._total_entries // nchunks)
        
        for start in tqdm(range(0, self._total_entries, chunk_size), desc="Caching unique entries", unit="chunk"):
            unique_ids.extend(self._get_unique_ids_chunk(start, min(start + chunk_size, self._total_entries)))
        
        # We have unique ids per chunk, now reduce to unique ids overall
        log_info(self._logger, f"Found {len(unique_ids)} unique entries across {nchunks} chunks, reducing to unique indices...")

        return unique_ids
                
    def _get_unique_ids_chunk(self, start, end)->List[int]:
        data, _ = self.get_items(start, end)
        data.drop_duplicates(inplace=True)
        # reduce data to unique entries
        unique_indices = data.index.to_numpy()

        return unique_indices.tolist()

    def _expand_wildcards(self, tree: uproot.TTree, patterns: List[str]) -> List[str]:
        '''
        Expand wildcard patterns to actual branch names.
        
        Args:
            tree: The uproot TTree object
            patterns: List of branch patterns (may include wildcards)
            
        Returns:
            List of actual branch names matching the patterns
        '''
        available_branches = tree.keys()
        expanded_branches = []
        
        for pattern in patterns:
            if '*' in pattern or '?' in pattern:
                # This is a wildcard pattern
                matches = [branch for branch in available_branches if fnmatch.fnmatch(branch, pattern)]
                if not matches:
                    log_warning(self._logger, f"No branches found matching pattern: {pattern}")
                else:
                    log_debug(self._logger, f"Pattern '{pattern}' matched branches: {matches}")
                    expanded_branches.extend(matches)
            else:
                # This is a literal branch name
                if pattern in available_branches:
                    expanded_branches.append(pattern)
                else:
                    log_warning(self._logger, f"Branch '{pattern}' not found in tree")
        
        # Remove duplicates while preserving order
        seen = set()
        result = []
        for branch in expanded_branches:
            if branch not in seen:
                seen.add(branch)
                result.append(branch)
                
        return result

    def __len__(self):
        return self._total_entries
        
    @property
    def total_entries(self) -> int:
        return self._total_entries
    
    def __get_list_from_tree(self, tree, branches: List[str], entries: Union[int, slice, List[int], np.ndarray], library: str = "pd"):
        '''
        Helper function to get a list of arrays from a tree.
        Supports fancy indexing with specific entry indices.
        '''
        if isinstance(entries, (int, slice)):
            # Single entry or slice - use entry_start/entry_stop
            if isinstance(entries, int):
                entry_start, entry_stop = entries, entries + 1
            else:
                entry_start, entry_stop = entries.start, entries.stop
            return tree.arrays(
                branches,
                cut=self.cuts,
                entry_start=entry_start,
                entry_stop=entry_stop,
                library=library
            )
        elif isinstance(entries, (list, np.ndarray)):
            # Fancy indexing with specific entry indices
            return tree.arrays(
                branches,
                cut=self.cuts,
                entry_start=entries,  # uproot accepts lists/arrays for fancy indexing
                library=library
            )
        else:
            raise ValueError(f"Unsupported entries type: {type(entries)}")
    
    def get_items(self, start: int, end: Optional[int]= None)-> TORCH_RETURN_TYPE:
        '''
        Get items from start to end (exclusive). Each worker process will open its own file handle.
        '''
        log_debug(self._logger, f"Getting items from {start} to {end}")
        
        if start < 0 or start >= self._total_entries:
            raise InvalidIndexError(f"Start index {start} out of range. Must be between 0 and {self._total_entries -1}.")
        
        start_entry = self._entries[start]
        
        if end is not None:
            if start>=end:
                raise InvalidIndexError("Start index must be less than end index.")
            if end > self._total_entries or end < 0:
                raise InvalidIndexError(f"End index {end} exceeds total entries {self._total_entries} in the dataset.")
            end_entry = self._entries[end-1] + 1  # Make it exclusive for range
            entry_slice = slice(start_entry, end_entry)
        else:
            entry_slice = start_entry
        
        try:
            file = uproot.open(self.file_path)
            tree = file[self.tree_name]
            data = self.__get_list_from_tree(tree, self.branches, entry_slice)            
            labels = self.__get_list_from_tree(tree, self.labels, entry_slice, library="pd")
        except Exception as e:
            if "cut" in str(e).lower() or "filter" in str(e).lower():
                raise InvalidCutError(f"Error applying cuts to tree: {e}")
            else:
                raise ROOTDatasetError(f"Error reading data from tree: {e}")
        
        if self._tensor_mode:
            data = torch.from_numpy(data.to_numpy()).to(torch.float32)
            labels = torch.from_numpy(labels.to_numpy()).to(torch.float32)
            
        return data, labels
    
    def __getitem__(self, idx: int | slice)-> TORCH_RETURN_TYPE:
        '''
        Get item at index idx. Each worker process will open its own file handle.
        '''        
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self._total_entries)
            if step != 1:
                raise InvalidIndexError("Step size other than 1 is not supported.")
            if stop < 0:
                stop += self._total_entries
            if start < 0:
                start += self._total_entries
            if start < 0 or stop > self._total_entries:
                raise InvalidIndexError("Slice indices out of range.")
         
            data, labels = self.get_items(start, stop)
        elif isinstance(idx, int):
            if idx < 0:
                idx += self._total_entries
            if idx < 0 or idx >= self._total_entries:
                raise InvalidIndexError(f"Index {idx} out of range [0, {self._total_entries}].")
            data, labels = self.get_items(idx, idx + 1)
        else:
            raise InvalidIndexError("Invalid argument type.")
        return data, labels
    
    def __iter__(self):
        '''
        Iterate over the dataset. Each worker process will open its own file handle.
        '''
        for idx in range(self._total_entries):
            yield self[idx]
    
    def get_n_labels(self):
        return len(self.labels)
    
    def get_n_features(self):
        return len(self.branches)