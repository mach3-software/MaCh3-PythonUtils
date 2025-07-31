import mpl_scatter_density
from matplotlib.colors import LinearSegmentedColormap
import warnings
from matplotlib import pyplot as plt
import numpy as np
import warnings
from sklearn import metrics
import matplotlib.pyplot as plt
from rich import print
import seaborn as sns

'''
Collection of diagnostics for machine learning models.
'''

class MLDiagnostics:
    """Class for handling diagnostics in machine learning models."""
    
    WHITE_VIRIDIS = LinearSegmentedColormap.from_list('white_viridis', [
        (0, '#ffffff'),
        (1e-20, '#440053'),
        (0.2, '#404388'),
        (0.3, '#2a788e'),
        (0.4, '#21a784'),
        (0.7, '#78d151'),
        (1, '#fde624'),
    ], N=256)

    @classmethod
    def plot_pred_true_scatter(cls, predicted_values, true_values, outfile: str="pred_true_scatter.pdf"):
        '''
        Plots scatter plot of predicted/ture
        '''
        fig = plt.figure()
        
        ax = fig.add_subplot(1,1,1, projection='scatter_density')
        
        # Bit hacky put plotting code is... bad so we're going to ignore the error it raises! 
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        density = ax.scatter_density(predicted_values, true_values, cmap=self.WHITE_VIRIDIS)
        # warnings.resetwarnings()
        lobf = np.poly1d(np.polyfit(predicted_values, true_values, 1))

        fig.colorbar(density, label="number of points per pixel")
        
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]

        ax.plot(lims, lobf(lims), "m", label=f"Best fit: true={lobf.c[0]}pred + {lobf.c[1]}", linestyle="dashed", linewidth=0.3)

        ax.plot(lims, lims, 'r', alpha=0.75, zorder=0, label="true=predicted", linestyle="dashed", linewidth=0.3)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        
        ax.set_xlabel("Predicted Log likelihood")
        ax.set_ylabel("True Log Likelihood")
        
        fig.legend()
        
        if outfile=="": outfile = f"evaluated_model_qq_tf.pdf"
        
        print(f"[bold spring_green1]Saving QQ to[/bold spring_green1][dodger_blue1] {outfile}")
            
        fig.savefig(outfile)
        
        try:
            cls.show_plot()
        except Exception:
            ...
            
        
        plt.close()

    @classmethod
    def plot_pred_true_projection(cls, predicted_values, true_values, outfile: str="pred_true_projection.pdf"):
        print(f"[bold red3]Mean Absolute Error :[/bold red3] [yellow3]{metrics.mean_absolute_error(predicted_values,true_values)}")
        
        outfile_name = outfile.split(".")[0]
        outfile = f"{outfile_name}.pdf"
        warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")
        lobf = np.poly1d(np.polyfit(predicted_values, true_values, 1))
        
        print(f"[bold purple]Line of best fit :[/bold purple] [dodger_blue1]y={lobf.c[0]}x + {lobf.c[1]}")
                
        # Gonna draw a hist
        difs = true_values-predicted_values
        print(f"mean: {np.mean(difs)}, std dev: {np.std(difs)}")
        plt.hist(difs, bins=100, density=True, range=(np.std(difs)*-5, np.std(difs)*5))
        plt.xlabel("True - Pred")
        plt.savefig(f"diffs_5sigma_range_{outfile}")
        
        try:
            cls.show_plot()
        except Exception:
            ...


        plt.close()

    
    @classmethod
    def confusion_matrix(cls, predicted_values, true_values, outfile: str="confusion_matrix.pdf", normalise: bool = False)-> "np.ndarray":
        """
        Plots a confusion matrix for the predicted and true values.
        """
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    
        cm = metrics.confusion_matrix(true_values, predicted_values)
        
        # Set so sum(cm) = 1
        if normalise:
            # Divide each row by its sum
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Generate heat map
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar_kws={
            'orientation': 'vertical',
            'ticks': np.linspace(0, cm.sum(), 5),
            'shrink': 0.8
            },
            vmin = 0,
            vmax = cm.sum(),
            linewidths=0.5,
            linecolor='black',

            ax = ax,
        )
        
        ax.set_ylabel('Actual Chain ID', fontsize=13)
        ax.set_title('Confusion Matrix', fontsize=17, pad=12)
        ax.xaxis.set_label_position('top') 
        ax.set_xlabel('Predicted Chain ID', fontsize=13)
        ax.xaxis.tick_top()
        plt.xticks(rotation=45)
        plt.show()

        plt.savefig(f"conf_matrix_{outfile}", bbox_inches='tight')
        
        try:
            cls.show_plot()
        except Exception:
            ...
        
        plt.close()
        return cm
    

    @classmethod
    def show_plot(cls) -> None:
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                plt.show()   # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return  # Terminal running IPython
            else:
                return  # Other type (?)
        except NameError:
            return      # Probably standard Python interpreter