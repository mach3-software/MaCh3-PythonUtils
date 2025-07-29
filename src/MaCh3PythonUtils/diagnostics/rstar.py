from sklearn.metrics import classification_report
from typing import List
from rich import print as rprint
import numpy as np
from tqdm import tqdm_notebook

from MaCh3PythonUtils.machine_learning.ml_factory import MLFactory
from MaCh3PythonUtils.machine_learning.diagnostics import MLDiagnostics
from MaCh3PythonUtils.file_handling.chain_handler import MultiChainHandler

'''
Application class for R* diagnostics.
'''

class RStar:
    def __init__(self, chain_files: List[str], n_iterations: int=1, **kwargs):
        """Initialises R* diagnostics with chain files.

        :param chain_files: List of chain files to use
        :type chain_files: List[str]
        """
        chain_handler = MultiChainHandler(chain_files)
        self._n_files = chain_handler.n_files
        
        factory = MLFactory(chain_handler, prediction_variable="chain_id", plot_name="rstar")
        
        # Now we create a bunch of instances of the ML models
        self.models = [
            factory.make_interface("scikit", "histboostclassifier", **kwargs) for _ in range(n_iterations)
        ]
        
    def train_models(self):
        """Trains all models."""
        for model in tqdm_notebook(self.models, desc="Training Models"):
            model.train_model()

        rprint([f"[bold green]Trained {len(self.models)} models[/bold green]"])
    
    def get_rstar(self, prediction, true_values, full_report: bool=False, verbose: bool=False):        
        report = classification_report(true_values, prediction, output_dict=True)
       
        # Just in case we get get a dictionary
        if isinstance(report, str):
            report = eval(report)
        
        pred_acc = report.get('accuracy', 0)
        
        
        if verbose:
            if full_report:
                rprint("[bold blue]Full Classification Report:[/bold blue]")
                rprint(report)

            print(f"[bold blue]Predictive Accuracy: {pred_acc:.4f}[/bold blue]")
        
        # Get R*
        print(f"[bold spring_green1]R*: {pred_acc*self._n_files:.4f}[/bold spring_green1]")
        
        return pred_acc * self._n_files
    
    def __call__(self):
        """Runs the R* diagnostics.

        :param true_values: True values for comparison
        :type true_values: List[str]
        :param full_report: Whether to print full classification report, defaults to False
        :type full_report: bool, optional
        :param verbose: Whether to print additional information, defaults to False
        :type verbose: bool, optional
        """
        rprint("[bold green]Running R* diagnostics...[/bold green]")

        def run_phase(data_attr, label_attr, phase):
            predictions = [model.test_model(getattr(model, data_attr)) for model in self.models]
            rstars = [self.get_rstar(predictions, getattr(model, label_attr)) for model in self.models]
            for i, (pred, model) in enumerate(zip(predictions, self.models)):
               MLDiagnostics.confusion_matrix(pred, getattr(model, label_attr),
                               outfile=f"{phase}_confusion_matrix_iteration_{i}.pdf", normalise=True)

            rprint(f"[bold green]Average R* for {phase} data: [bold cyan]{np.mean(rstars):.4f}±{np.std(rstars):.4f}[/bold cyan][/green]")
            rprint(f"[dim green] For [cyan]{self._n_files}[/cyan] files, an R* of ~[cyan]1.0[cyan] indicates a model that cannot distinguish between chains (i.e. they're similarly mixed).\n\
                [dim green]As R*->{self._n_files} the model becomes better at telling the difference between chains![/dim green]")

            return rstars

        train_rstar = run_phase("training_data", "scaled_training_labels", "train")
        test_rstar = run_phase("test_data", "scaled_test_labels", "test")

        return {
            "train_rstar": train_rstar,
            "train_rstar_mean": np.mean(train_rstar),
            "train_rstar_std": np.std(train_rstar),
            "test_rstar": test_rstar,
            "test_rstar_mean": np.mean(test_rstar),
            "test_rstar_std": np.std(test_rstar)
        }
