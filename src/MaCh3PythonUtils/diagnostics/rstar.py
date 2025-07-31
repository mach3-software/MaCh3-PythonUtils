
from sklearn.metrics import classification_report
from typing import TypedDict, List, Tuple
import numpy as np
import multiprocessing as mp 
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

from rich import print as rprint
from sklearn.metrics import log_loss

from MaCh3PythonUtils.machine_learning.ml_factory import MLFactory
from MaCh3PythonUtils.machine_learning.diagnostics import MLDiagnostics
from MaCh3PythonUtils.file_handling.chain_handler import MultiChainHandler

class RStarOutput(TypedDict):
    '''
    Application class for R* diagnostics.
    '''
    train_rstar: List[float]
    train_rstar_mean:  np.float32
    train_rstar_std: np.float32
    test_rstar: List[float]
    test_rstar_mean: np.float32
    test_rstar_std: np.float32


class RStar:
    def __init__(self, chain_handler: MultiChainHandler, n_fitters: int=1, interface: str="scikit", algorithm: str="histboostclassifier",
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
        """
        self.chain_handler = chain_handler
        
        self._n_files = self.chain_handler.n_files
        self._test_size = test_size
        
        factory = MLFactory(self.chain_handler, prediction_variable="chain_id", plot_name="rstar")
        
        rprint(f"[bold green]Creating {n_fitters} models using {algorithm} algorithm[/bold green]")

        # Create models but don't set training data yet
        self.models = [
            factory.make_interface(interface, algorithm, **kwargs) for _ in tqdm_notebook(range(n_fitters), desc="Creating Models")
        ]
                
        
        rprint(f"[green]Creating {n_fitters} different train/test splits...[/green]")
                
        # Now assign data to each model using the indices (working with internal attributes)
        
        rprint(f"[green]Assigning data to {len(self.models)} models in parallel...[/green]")
        with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
            futures = []
            for model in tqdm_notebook(self.models, desc="Assigning Data"):
                futures.append(executor.submit(model.set_training_test_set, self._test_size))

        # Wait for all futures to complete
        for future in futures:
            future.result()


        rprint(f"[cyan]Using {self._n_files} files for R* diagnostics, with a training set containing [bold green]{len(self.models[0].train_labels)}[/bold green] entries and a testing set containing [bold green]{len(self.models[0].test_labels)}[/bold green] entries[/cyan]")

        self._trained = False
        rprint(f"[bold green]Initialised RStar with {len(self.models)} models using {algorithm} algorithm[/bold green]")
        
        
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
    
    def make_rstar_hist(self, rstars: List[float], outfile: str, zoom: bool = True):
        """Creates a histogram of R* values.

        :param outfile: Output file path for the histogram
        :type outfile: str
        :param rstars: List of R* values
        :type rstars: List[float]
        """
        
        plt.figure(figsize=(10, 6))
        
        if zoom:
            min_bin = min(1.0, min(rstars))
        else:
            min_bin = min(rstars)
            
        
        plt.hist(rstars, bins=np.linspace(min_bin, self._n_files, 100).tolist(),
                color='blue', alpha=0.7)
        # Add a vertical line at the mean
        mean_rstar = float(np.mean(rstars))
        plt.axvline(mean_rstar, color='red', linestyle='dashed', linewidth=1, label=f'Mean R*: {mean_rstar:.2f}')
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
            import json
            report = json.loads(report)
        
        pred_acc = report.get('accuracy', 0)
        
        
        if verbose:
            if full_report:
                rprint("[bold blue]Full Classification Report:[/bold blue]")
                rprint(report)

            rprint(f"[bold blue]Predictive Accuracy: {pred_acc:.4f}[/bold blue]")
            rprint(f"[bold spring_green1]R*: {pred_acc*self._n_files:.4f}[/bold spring_green1]")
        
        return pred_acc * self._n_files


    def get_rstar(self, plot_log_loss: bool = False)->RStarOutput:
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
        
        if plot_log_loss:
            rprint("Getting final Log Loss for each model:")
            for model in tqdm_notebook(self.models, desc="Calculating Log Loss"):
                predictions = model.model_predict(model.test_data)
                loss.append(log_loss(model.scaled_test_labels, predictions))

            rprint(f"[bold green]Average Log Loss: {np.mean(loss):.4f}±{np.std(loss):.4f}[/bold green]")
            rprint("[bold green]Running R* diagnostics...[/bold green]")


        def run_phase(phase)->List[float]:
            rprint(f"[bold green]Running R* diagnostics for {phase} data...[/bold green]")
            rstars = []
            all_predictions = []
            all_true_values = []
            
            for model in tqdm_notebook(self.models, desc=f"Getting R* for {phase}", total=len(self.models)):
                predictions = model.model_predict(getattr(model, f"scaled_{phase}_data"))
                all_predictions.extend(predictions)
                all_true_values.extend(getattr(model, f"scaled_{phase}_labels"))
                
                rstar = self.get_rstar_single_iter(predictions, getattr(model, f"scaled_{phase}_labels"), full_report=False, verbose=False)                                
                rstars.append(rstar)
            
            
            # Get confusion matrix for all models
            MLDiagnostics.confusion_matrix(all_predictions, all_true_values, outfile=f"confusion_matrix_{phase}.pdf", normalise=True)

            rprint(f"[bold green]Average R* for {phase} data: [bold cyan]{np.mean(rstars):.4f}±{np.std(rstars):.4f}[/bold cyan][/bold green]")
            rprint(f"[dim green] For [cyan]{self._n_files}[/cyan] files, an R* of ~[cyan]1.0[cyan] indicates a model that cannot distinguish between chains (i.e. they're similarly mixed).\n\
                [dim green]As R*->{self._n_files} the model becomes better at telling the difference between chains![/dim green]")

            return rstars

        train_rstar = run_phase("train")
        test_rstar = run_phase("test")


        return {
            "train_rstar": train_rstar,
            "train_rstar_mean": np.mean(train_rstar, dtype=np.float32),
            "train_rstar_std": np.std(train_rstar, dtype=np.float32),
            "test_rstar": test_rstar,
            "test_rstar_mean": np.mean(test_rstar, dtype=np.float32),
            "test_rstar_std": np.std(test_rstar, dtype=np.float32)
        }