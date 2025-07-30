from sklearn.metrics import classification_report
from typing import TypedDict, List, Optional
from rich import print as rprint
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp 
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from MaCh3PythonUtils.machine_learning.file_ml_interface import FileMLInterface
from MaCh3PythonUtils.machine_learning.ml_factory import MLFactory
from MaCh3PythonUtils.machine_learning.diagnostics import MLDiagnostics
from MaCh3PythonUtils.file_handling.chain_handler import MultiChainHandler

'''
Application class for R* diagnostics.
'''
class RStarOutput(TypedDict):
    train_rstar: List[float]
    train_rstar_mean:  np.float32
    train_rstar_std: np.float32
    test_rstar: List[float]
    test_rstar_mean: np.float32
    test_rstar_std: np.float32


class RStar:
    def __init__(self, chain_handler: MultiChainHandler, n_iterations: int=1, algorithm: str="histboostclassifier", 
                 test_size: float= 0.8, **kwargs):
        """Initialises R* diagnostics with chain files.

        :param chain_handler: Handler for multiple chain files
        :type chain_handler: MultiChainHandler
        :param n_iterations: Number of model iterations with different train/test splits
        :type n_iterations: int
        :param algorithm: ML algorithm to use
        :type algorithm: str
        :param test_size: Fraction of data to use for testing
        :type test_size: float
        :param random_seed: Base random seed for reproducibility. Each model gets seed + iteration
        :type random_seed: Optional[int]
        """
        self.chain_handler = chain_handler
        
        self._n_files = self.chain_handler.n_files
        self._test_size = test_size
        
        factory = MLFactory(self.chain_handler, prediction_variable="chain_id", plot_name="rstar")
        
        rprint(f"[bold green]Creating {n_iterations} models using {algorithm} algorithm[/bold green]")

        # Create models but don't set training data yet
        self.models = [
            factory.make_interface("scikit", algorithm, **kwargs) for _ in tqdm_notebook(range(n_iterations), desc="Creating Models")
        ]
                
        # Create different train/test splits for each model using only indices
        self._train_indices = []
        self._test_indices = []
        
        rprint(f"[green]Creating {n_iterations} different train/test splits...[/green]")
                
        # Now assign data to each model using the indices (working with internal attributes)
        def assign_data_to_model(model: FileMLInterface):
            """Helper function to assign data to a single model."""

            train_idx, test_idx = train_test_split(
                np.arange(len(self.chain_handler.ttree_array)),
                test_size=self._test_size,
            )

            # Set the internal attributes directly
            model._training_data = self.chain_handler.ttree_array.iloc[train_idx]
            model._training_labels = self.chain_handler.ttree_array.iloc[train_idx]
            model._test_data = self.chain_handler.ttree_array.iloc[test_idx]
            model._test_labels = self.chain_handler.ttree_array.iloc[train_idx]
    
        
        rprint(f"[green]Assigning data to {len(self.models)} models in parallel...[/green]")
        with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
            # Submit all data assignment tasks
            futures = [
                executor.submit(assign_data_to_model, model) 
                for model in self.models
            ]
            
            
            # Use tqdm to track progress
            for future in tqdm_notebook(as_completed(futures), total=len(futures), desc="Assigning Data"):
                future.result()  # Get the result to ensure any exceptions are raised
        
        self._trained = False
        rprint(f"[bold green]Initialised RStar with {len(self.models)} models using {algorithm} algorithm[/bold green]")
        rprint(f"[green]Each model has a unique train/test split (seed base: {self._random_seed})[/green]")
        
        
    def train_models(self):
        """Trains all models."""
        if self._trained:
            rprint("[bold red]Models already trained![/bold red]")
            return
        
        rprint(f"[green]Training models of type[/]\n[bold red]{self.models[0].model}")
                
        for model in tqdm_notebook(self.models, desc="Training Models"):
            model.train_model()

        print("LogLoss for each model:")

        rprint(f"[bold green]Trained {len(self.models)} models[/bold green]")
    
    def make_rstar_hist(self, rstars: List[float], outfile: str):
        """Creates a histogram of R* values.

        :param outfile: Output file path for the histogram
        :type outfile: str
        :param rstars: List of R* values
        :type rstars: List[float]
        """
        
        plt.figure(figsize=(10, 6))
        plt.hist(rstars, bins=30, color='blue', alpha=0.7)
        plt.title('Histogram of R* Values')
        plt.xlabel('R* Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        print(outfile)
        plt.savefig(outfile)
        
        MLDiagnostics.show_plot()
        plt.close()
        
        rprint(f"[bold green]Saved histogram to {outfile}[/bold green]")
    
    def get_rstar_single_iter(self, prediction, true_values, full_report: bool=False, verbose: bool=False)->float:
        report = classification_report(true_values, prediction, output_dict=True)
       
        # Just in case we get get a dictionary
        if isinstance(report, str):
            report = eval(report)
        
        pred_acc = report.get('accuracy', 0)
        
        
        if verbose:
            if full_report:
                rprint("[bold blue]Full Classification Report:[/bold blue]")
                rprint(report)

            rprint(f"[bold blue]Predictive Accuracy: {pred_acc:.4f}[/bold blue]")
            rprint(f"[bold spring_green1]R*: {pred_acc*self._n_files:.4f}[/bold spring_green1]")
        
        return pred_acc * self._n_files
    
    
    def get_rstar(self, print_all_confusion: bool = False)->RStarOutput:
        """Runs the R* diagnostics.

        :param true_values: True values for comparison
        :type true_values: List[str]
        :param full_report: Whether to print full classification report, defaults to False
        :type full_report: bool, optional
        :param verbose: Whether to print additional information, defaults to False
        :type verbose: bool, optional
        """

        if not self._trained:
            rprint("[bold red]Models not trained! Training models first.[/bold red]")
            self.train_models()

        loss = []
        
        rprint("Getting final Log Loss for each model:")
        for model in tqdm_notebook(self.models, desc="Calculating Log Loss"):
            predictions = model.model_predict(model.test_data)
            loss.append(log_loss(model.scaled_test_labels, predictions))

        rprint(f"[bold green]Average Log Loss: {np.mean(loss):.4f}±{np.std(loss):.4f}[/bold green]")
        rprint("[bold green]Running R* diagnostics...[/bold green]")


        def run_phase(data_attr, label_attr, phase, print_all_confusion: bool = False)->List[float]:
            rprint(f"[bold green]Running R* diagnostics for {phase} data...[/bold green]")
            rstars = []
            for i, model in tqdm_notebook(enumerate(self.models), desc=f"Getting R* for {phase}", total=len(self.models)):
                predictions = model.model_predict(getattr(model, data_attr))
                
                rstar = self.get_rstar_single_iter(predictions, getattr(model, label_attr), full_report=False, verbose=False)                
                
                if print_all_confusion or i==0:
                    MLDiagnostics.confusion_matrix(predictions, getattr(model, label_attr),
                                                outfile=f"{phase}_{i}_confusion_matrix.pdf", normalise=True)

                rstars.append(rstar)

            rprint(f"[bold green]Average R* for {phase} data: [bold cyan]{np.mean(rstars):.4f}±{np.std(rstars):.4f}[/bold cyan][/bold green]")
            rprint(f"[dim green] For [cyan]{self._n_files}[/cyan] files, an R* of ~[cyan]1.0[cyan] indicates a model that cannot distinguish between chains (i.e. they're similarly mixed).\n\
                [dim green]As R*->{self._n_files} the model becomes better at telling the difference between chains![/dim green]")

            return rstars

        train_rstar = run_phase("training_data", "scaled_training_labels", "train", print_all_confusion)
        test_rstar = run_phase("test_data", "scaled_test_labels", "test", print_all_confusion)

        return {
            "train_rstar": train_rstar,
            "train_rstar_mean": np.mean(train_rstar, dtype=np.float32),
            "train_rstar_std": np.std(train_rstar, dtype=np.float32),
            "test_rstar": test_rstar,
            "test_rstar_mean": np.mean(test_rstar, dtype=np.float32),
            "test_rstar_std": np.std(test_rstar, dtype=np.float32)
        }