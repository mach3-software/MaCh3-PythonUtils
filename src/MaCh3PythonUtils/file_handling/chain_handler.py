'''
Python tool to load in some generic TTree objects and export to numpy array/pandas dataframe
'''
import uproot as ur
import pandas as pd
from typing import List, Union, Any, Protocol
import warnings
from concurrent.futures import ThreadPoolExecutor
import gc
import numpy as np
from numpy.typing import NDArray

class ChainProtocol(Protocol):
    """Protocol defining the interface for chain handlers."""
    
    @property
    def plot_branches(self) -> List[str]: ...
    
    @plot_branches.setter
    def plot_branches(self, useful_branches: List[str]) -> None: ...
    
    def add_additional_plots(self, additional_branches: List[str] | str, exact: bool = False) -> None: ...
    
    def ignore_plots(self, ignored_branches: List[str] | str) -> None: ...
    
    def add_new_cuts(self, new_cuts: Union[str, List[str]]) -> None: ...
    
    def convert_ttree_to_array(self, close_file: bool = True) -> None: ...
    
    def close_file(self) -> None: ...
    
    @property
    def ttree_array(self) -> pd.DataFrame: ...
    
    @property
    def ndim(self) -> int: ...
    
    @property
    def lower_bounds(self) -> NDArray: ...
    
    @property
    def upper_bounds(self) -> NDArray: ...

class ChainHandler:
    """
    Class to load in ROOT files containing a single TTree

    :param file_name: Name of ROOT file containing useful TTree
    :type file_name: str
    :param ttree_name: Name of TTree contained in ROOT file
    :type ttree_name: str, optional
    """
    def __init__(self, file_name: str, ttree_name: str="posteriors", verbose=False)->None:
        """_summary_

        :param file_name: Input file name
        :type file_name: str
        :param ttree_name: Input TTree name, defaults to "posteriors"
        :type ttree_name: str, optional
        :param verbose: Verbose or not, defaults to False
        :type verbose: bool, optional
        :raises IOError: No file found
        """
        print(f"Attempting to open {file_name}")
        try:
            self._posterior_ttree =  ur.open(f"{file_name}:{ttree_name}")

        except FileNotFoundError:
            raise IOError(f"The file '{file_name}' does not exist or does not contain '{ttree_name}")
        
        print(f"Succesfully opened {file_name}:{ttree_name}")
        warnings.filterwarnings("ignore", category=DeprecationWarning) #Some imports are a little older
        warnings.filterwarnings("ignore", category=UserWarning) #Some imports are a little older
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning) # Not a fan of being yelled at by pandas

        self._plotting_branches = [] # Filled with branches we want to plot
        self._cuts = [] # If we want to apply cuts (can be done later but fastest at load time)

        self._ttree_array = None #For storing the eventual TTree
        
        self._is_file_open = True
        
        self._verbose = verbose
        self._ignored_branches = []

        self._arviz_tree = None

    def close_file(self)->None:
        '''
        Closes ROOT file, should be called to avoid memory issues!
        '''

        if not self._is_file_open:
            self._posterior_ttree.close()
            self._is_file_open = False

    @property
    def plot_branches(self)->List[str]:
        '''
        Getter for list of useful branches
        :return: List of branches used in file
        :rtype: list
        '''
        return self._plotting_branches

    @plot_branches.setter
    def plot_branches(self, useful_branches: List[str])->None:
        '''
        Setter for list of useful branches
        :param useful_branches: List of branches we want to plot with
        :type useful_branches: list
        '''
        if not self._is_file_open:
            raise Warning("Adding branches after shutting the file has no effect")

        self._plotting_branches = useful_branches

    def add_additional_plots(self, additional_branches: List[str] | str, exact=False)->None:
        '''
        To add more branches to the plotting branch list
        :param additional_branches: List of branches to add to the plotting list
        :type additional_branches: list
        '''
        if not self._is_file_open:
            raise Warning("Adding branches after shutting the file has no effect")

        if isinstance(additional_branches, str):
            additional_branches = [additional_branches]

        branch_list = []
        for key in self._posterior_ttree.keys():
            
            if key in self._ignored_branches: continue
            
            if any(var in key for var in additional_branches) and not exact: # Not the most efficient but adds variables to our list of variables
                branch_list.append(key)
            elif exact and key in additional_branches:
                branch_list.append(key)

        self._plotting_branches.extend(branch_list)
    
    def ignore_plots(self, ignored_branches: List[str]| str)->None:
        """List of plots to ignore

        :param ignored_branches: _description_
        :type ignored_branches: List[str] | str
        """        
        if isinstance(ignored_branches, str):
            ignored_branches = list(ignored_branches)
            
        for branch in ignored_branches:
            if branch in self._ignored_branches: continue

            self._ignored_branches.append(branch)
    
            if branch in self._plotting_branches:
                self._plotting_branches.remove(branch)
        
    
    def add_new_cuts(self, new_cuts: Union[str, List[str]])->None:
        '''
        Specifies list of cuts to apply to the TTree (something like ['step>80000', 'dm23>0'])
        :param new_cuts: List of/single cut to apply
        :type new_cuts: list, str
        '''

        # Hacky but lets us be a little bit polymorphic
        if not self._is_file_open:
            raise Warning("Applying cuts after shutting the file has no effect")

        if type(new_cuts)==str:
            new_cuts = [new_cuts]

        self._cuts.extend(new_cuts)

    def convert_ttree_to_array(self, close_file=True)->None:
        '''
        Converts the TTree table to array
        :param close_file: Do you want to close the ROOT file after calling this method?
        :type close_file: bool, optional
        '''
        if not self._is_file_open:
            raise IOError("Cannot convert TTree to array after input ROOT file is shut")

        cuts = ""
        if len(self._cuts)>0:
            cuts = f"*".join(f"({cut})" for cut in self._cuts)
        
        
        with ThreadPoolExecutor() as executor:
            # Make sure we have loads of memory available!
            # Ensures we don't run into funny behaviour when uncompressing
            total_memory_needed = self._posterior_ttree.uncompressed_bytes #in bytes

            if self._verbose:
                print(f"Using {executor._max_workers} threads and requiring {np.round(self._posterior_ttree.uncompressed_bytes*1e-9,3)} Gb memory")
                print("Using the following branches: ")
                for i in self._plotting_branches:
                    print(f"  -> {i}")
            
            # To make sure we don't run into any unpleasantness
            # total_available_memory = int(psutil.virtual_memory().available)
            # if total_memory_needed < total_available_memory: # For some reason I can't just do needed<available or it breaks... [not sure why, regardless this check works!]
            #     print(total_memory_needed - total_available_memory, type(total_memory_needed - total_available_memory))
            #     raise MemoryError(f"Posterior tree requires {np.round(self._posterior_ttree.uncompressed_bytes*1e-9,3)} Gb memory, system only has {np.round(total_available_memory*1e-9,3)} Gb available")

            # We're going to surpress some pandas warnings here since ROOT isn't totally efficient when converting to Pandas (but it'll do!)
            # warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
            # Now we generate an array object
            if len(self._plotting_branches)==0:
                self._ttree_array = self._posterior_ttree.arrays(self._posterior_ttree.keys(), cut=cuts, library='pd', decompression_executor=executor, interpretation_executor=executor) # Load in ROOT TTree
            else:
                self._ttree_array = self._posterior_ttree.arrays(self._plotting_branches, cut=cuts, library='pd', array_cache=f"{total_memory_needed} b", decompression_executor=executor, interpretation_executor=executor) # Load in ROOT TTree

            if self._verbose:
                print(f"Converted TTree to pandas dataframe with {len(self._ttree_array)} elements")

        if close_file:
            self.close_file()

        # Just to really make sure we have no memory overuse we'll call the Garbage collector
        gc.collect()    


    @property
    def ttree_array(self)->pd.DataFrame:
        '''
        Getter for the converted TTree array
        :return: Table containing TTree in non-ROOT format
        :rtype: Union[np.array, pd.DataFrame, ak.Array]
        '''
        return self._ttree_array
    
    @ttree_array.setter
    def ttree_array(self, new_array: Any=None)->None:
        '''
        Setter for TTree array object ::: NOTE THIS WILL JUST RAISE AN ERROR
        :param new_array: Object to set our ttree_array_to
        :type new_array: Any
        '''
        # Implemented in case someone tries to do something daft!
        raise NotImplementedError("Cannot set converted TTree array to new type")        
 
    @property
    def ndim(self)->int:
        if self._ttree_array is None:    
            return 0

        return self._ttree_array.shape[1]
 
    @property
    def lower_bounds(self)->NDArray:
        # Lower bounds for all params
        if self._ttree_array is None:    
            return np.empty(self.ndim)
        return self._ttree_array.min(axis=0).to_numpy()

    @property
    def upper_bounds(self)->NDArray:
        # Upper bounds for all params
        if self._ttree_array is None:    
            return np.empty(self.ndim)
        
        return self._ttree_array.max(axis=0).to_numpy()
    

class MultiChainHandler:
    """
    Wrapper class to handle multiple chains, applies all ChainHandler methods to all chains.
    Implements ChainProtocol for consistent interface.
    """
    def __init__(self, file_names: List[str], ttree_name: str="posteriors", verbose=False, add_id_col: bool = True)->None:
        self._add_id_col = add_id_col
        self._n_files = len(file_names)
        self._chain = [ChainHandler(file_name, ttree_name, verbose) for file_name in file_names]

    @property
    def n_files(self)->int:
        return self._n_files

    def close_file(self) -> None:
        """Close all chain files."""
        for chain in self._chain:
            chain.close_file()
    
    def add_additional_plots(self, additional_branches: List[str] | str, exact=False) -> None:
        """Add additional plotting branches to all chains."""
        for chain in self._chain:
            chain.add_additional_plots(additional_branches, exact)
    
    def ignore_plots(self, ignored_branches: List[str] | str) -> None:
        """Ignore specified branches in all chains."""
        for chain in self._chain:
            chain.ignore_plots(ignored_branches)
    
    def add_new_cuts(self, new_cuts: Union[str, List[str]]) -> None:
        """Add cuts to all chains."""
        for chain in self._chain:
            chain.add_new_cuts(new_cuts)
    
    @property
    def ndim(self) -> int:
        """Get number of dimensions from the combined array."""
        if hasattr(self, '_ttree_array') and self._ttree_array is not None:
            return self._ttree_array.shape[1]
        elif len(self._chain) > 0:
            return self._chain[0].ndim
        return 0

    def convert_ttree_to_array(self, close_file=True)->None:
        # Special handling to concatenate arrays and add chain_id
        for i, chain in enumerate(self._chain):
            chain.convert_ttree_to_array(close_file=False)
            if self._add_id_col:
                chain.ttree_array['chain_id'] = i
        self._ttree_array = pd.concat([chain.ttree_array for chain in self._chain])        
        if close_file:
            self.close_file()
        gc.collect()

    @property
    def ttree_array(self)->pd.DataFrame:
        '''
        Getter for the converted TTree array
        :return: Table containing TTree in non-ROOT format
        :rtype: Union[np.array, pd.DataFrame, ak.Array]
        '''
        return self._ttree_array

 
    @property
    def lower_bounds(self)->NDArray:
        # Lower bounds for all params
        if self._ttree_array is None:    
            return np.empty(self.ndim)
        return self._ttree_array.min(axis=0).to_numpy()

    @property
    def upper_bounds(self)->NDArray:
        # Upper bounds for all params
        if self._ttree_array is None:    
            return np.empty(self.ndim)
        
        return self._ttree_array.max(axis=0).to_numpy()


    @property
    def plot_branches(self)->List[str]:
        '''
        Getter for list of useful branches
        :return: List of branches used in file
        :rtype: list
        '''
        return self._chain[0]._plotting_branches
    
    @plot_branches.setter
    def plot_branches(self, useful_branches: List[str]) -> None:
        """Set plotting branches for all chains."""
        for chain in self._chain:
            chain.plot_branches = useful_branches
    
    def get_individual_results(self, method_name: str, *args, **kwargs) -> List[Any]:
        """
        Call a method on all individual chains and return list of results.
        Useful when you need results from each chain separately.
        """
        results = []
        for chain in self._chain:
            method = getattr(chain, method_name)
            results.append(method(*args, **kwargs))
        return results
    
    def get_individual_properties(self, property_name: str) -> List[Any]:
        """
        Get a property value from all individual chains.
        Useful when you need property values from each chain separately.
        """
        return [getattr(chain, property_name) for chain in self._chain]
