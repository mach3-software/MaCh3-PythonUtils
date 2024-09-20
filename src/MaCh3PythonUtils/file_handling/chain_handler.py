'''
Python tool to load in some generic TTree objects and export to numpy array/pandas dataframe
'''
import uproot as ur
import pandas as pd
from typing import List, Union, Any
import warnings
from concurrent.futures import ThreadPoolExecutor
import gc
import numpy as np
import arviz as az

class ChainHandler:
    """
    Class to load in ROOT files containing a single TTree

    :param file_name: Name of ROOT file containing useful TTree
    :type file_name: str
    :param ttree_name: Name of TTree contained in ROOT file
    :type ttree_name: str, optional
    """
    def __init__(self, file_name: str, ttree_name: str="posteriors", verbose=False)->None:
        '''
        Constructor method
        '''
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
            total_memory_needed = 8*self._posterior_ttree.uncompressed_bytes*(executor._max_workers) #in bytes

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
    
    def make_arviz_tree(self):
        if self._ttree_array is None:
            raise RuntimeError("Error have not converted ROOT TTree to pandas data frame yet!")

        print("Generating Arviz data struct [this may take some time!]")
        self._arviz_tree = az.dict_to_dataset(self._ttree_array.to_dict(orient='list'))
        
 
    @property
    def arviz_tree(self)->az.InferenceData:
        """Gets arviz version of dataframe

        :return: Arviz version of tree
        :rtype: arviz.InferenceData
        """        
        return self._arviz_tree