from MaCh3PythonUtils.file_handling.chain_handler import ChainHandler
from sklearn.ensemble import HistGradientBoostingClassifier


from abc import ABC, abstractmethod
from typing import Any, Tuple, Iterable
from sklearn.model_selection import train_test_split
import pandas as pd
import mpl_scatter_density
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pickle
from typing import List, Dict
# import tensorflow as tf
import warnings
from tqdm import tqdm
from scipy.optimize import minimize, OptimizeResult
import sklearn.ensemble as ske
from sklearn.metrics import log_loss
from sklearn.metrics import roc_curve, auc

from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import LabelBinarizer

from rich import print

class FileMLInterface(ABC):
    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
        (0, '#ffffff'),
        (1e-20, '#440053'),
        (0.2, '#404388'),
        (0.3, '#2a788e'),
        (0.4, '#21a784'),
        (0.7, '#78d151'),
        (1, '#fde624'),
    ], N=256)
    
    def __init__(self, chain: ChainHandler, prediction_variable: str, fit_name: str) -> None:
        """General Interface for all ML models

        :param chain: ChainHandler instance
        :type chain: ChainHandler
        :param prediction_variable: "Label" used for prediction
        :type prediction_variable: str
        :raises ValueError: Checks to see if label exists in tree
        """        
        self._chain = chain
        
        self._fit_name = fit_name
        self._prediction_variable = prediction_variable
        
        if prediction_variable not in self._chain.ttree_array.columns:
            raise ValueError(f"Cannot find {prediction_variable} in input tree")
        
        self._model = None
        
        self._training_data=None
        self._training_labels=None
        self._test_data=None
        self._test_labels=None

        # Scaling components
        self._scaler = StandardScaler()
        # self._pca_matrix = PCA(n_components=0.95)
        
        self._label_scaler = StandardScaler()
        #self.datapoints=None
        
        
            
    def __separate_dataframe(self)->Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data frame into feature + label objects

        :return: features, labels
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]
        """        
        # Separates dataframe into features + labels
        features = self._chain.ttree_array.copy()
        labels   = pd.DataFrame(features.pop(self._prediction_variable) )
        
        return features, labels
    
    def set_training_test_set(self, test_size: float):
        """Splits data/labels into training and testing tests

        :param test_size: Proportion of data used for testing
        :type test_size: float
        """        
        # Splits in traing + test_spit
        features, labels = self.__separate_dataframe()
        self._training_data, self._test_data, self._training_labels, self._test_labels =  train_test_split(features, labels, test_size=test_size)

        # Fit scaling pre-processors. These get applied properly when scale_data is called
        self._scaler.fit(self._training_data)
        self._label_scaler.fit(self._training_labels)
        
        # self._pca_matrix.fit(scaled_training)

    def scale_data(self, input_data):
        # Applies transformations to data set
        scale_data = self._scaler.transform(input_data)
        return scale_data
    
    def scale_labels(self, labels):
        # return self._label_scaler.transform(labels)
        return labels
        # return labels.values.reshape(-1, 1)

    def invert_scaling(self, input_data):
        # Inverts transform
        # unscaled_data = self._pca_matrix.inverse_transform(input_data)
        unscaled_data = self._scaler.inverse_transform(input_data)
        return unscaled_data

    @property
    def model(self)->Any:
        """Model used

        :return: Returns ML model being used
        :rtype: Any
        """        
        # Returns model being used
        return self._model    
    
    @property
    def chain(self)->ChainHandler:
        return self._chain
    
    @property
    def training_data(self)->pd.DataFrame:
        """Gets training data

        :return: Training data set
        :rtype: pd.DataFrame
        """        
        if self._training_data is None:
            return self._chain.ttree_array.iloc[:,:-1]

        return self._training_data

    @property
    def scaled_training_data(self):
        return self.scale_data(self.training_data)
    
    @property
    def scaled_training_labels(self):
        return self.scale_labels(self._training_labels).T

    @property
    def test_data(self)->pd.DataFrame:
        """Gets training data

        :return: Training data set
        :rtype: pd.DataFrame
        """ 
        if self._test_data is None:
            return self._chain.ttree_array.iloc[:,:-1]
        
        return self._test_data

    
    def add_model(self, ml_model: Any)->None:
        """Add model to data set

        :param ml_model: Sets model to be ml_model
        :type ml_model: Any
        """        
        # Add ML model into your interface
        self._model = ml_model
    
    @abstractmethod
    def train_model(self):
        """Abstract method, should be overwritten with model training
        """        
        # Train Model method
        pass
    
    @abstractmethod
    def model_predict(self, testing_data: pd.DataFrame)->Iterable:
        """Abstract method, should return model prediction

        :param testing_data: Data to test model on 
        :type testing_data: pd.DataFrame
        """
        pass
        
    def save_model(self, output_file: str):
        """Save model to pickle

        :param output_file: Pickle file to save to
        :type output_file: str
        """        
        print(f"Saving to {output_file}")
        with open(output_file, 'wb') as f:
            pickle.dump(self._model, f)

    def save_scaler(self, output_file: str):
        pickle.dump(self._scaler, open(output_file, 'wb'))
        
    def load_scaler(self, input_scaler: str):
        self._scaler = pickle.load(open(input_scaler, 'rb'))

        
    def load_model(self, input_model: str):
        """Unpickle model

        :param input_file: Pickled Model
        :type input_file: str
        """        
        print(f"[spring_green1]Attempting to load file from[/spring_green1][bold red3] {input_file}")
        with open(input_model, 'r') as f:
            self._model = pickle.load(f)
            
        
    def test_model(self):
        """Test model

        :raises ValueError: No model set
        :raises ValueError: No test data set 
        """        
        if self._model is None:
            raise ValueError("No Model has been set!")

        if self._test_data is None or self._test_labels is None:
            raise ValueError("No test data set")


        print("Training Results!")
        train_prediction = self.model_predict(self.scale_data(self._training_data))
        train_as_numpy = self.scale_labels(self._training_labels)#.T[0]
        self.evaluate_model(train_prediction, train_as_numpy, "train_qq_plot.pdf")

        print("=====")
        print("Testing Results!")

        test_prediction = self.model_predict(self.scale_data(self._test_data))
        test_as_numpy = self.scale_labels(self._test_labels)#.T[0]
        
        self.evaluate_model(test_prediction, test_as_numpy, outfile=f"{self._fit_name}")
        print("=====")
        
        


    def print_model_summary(self):
        print("Model Summary")
       
    def model_predict_single_sample(self, sample):
        sample_shaped = sample.reshape(1,-1)
        return self.model_predict(sample_shaped)[0]
       
    def get_maxlikelihood(self)->OptimizeResult:
        init_vals = self.training_data.iloc[[1]].to_numpy()[0]
    
        print("[bold purple]Calculating max LLH")
        maximal_likelihood = minimize(self.model_predict_single_sample, init_vals, bounds=zip(self._chain.lower_bounds[:-1], self._chain.upper_bounds[:-1]), method="L-BFGS-B", options={"disp": True})
        return maximal_likelihood

       
    def run_likelihood_scan(self, n_divisions: int = 500):
        # Get nominals
        print("Running LLH Scan")
        maximal_likelihood = self.get_maxlikelihood()
        maximal_nominal=maximal_likelihood.x        
    
        errors = np.sqrt(np.diag(maximal_likelihood.hess_inv(np.identity(self.chain.ndim-1))))

        print("[bold red3]Maximal Pars :")
        for i in range(self.chain.ndim-1):
            print(f"[bold red3]Param :[/bold red3] [yellow3]{self.chain.plot_branches[i]} : {maximal_likelihood.x[i]}±{errors[i]}")


        with PdfPages("llh_scan.pdf") as pdf:
            for i in tqdm(range(self.chain.ndim-1), total=self.chain.ndim-1):
                # Make copy since we'll be modifying!
                
                lower_bound = self.chain.lower_bounds[i]
                upper_bound = self.chain.upper_bounds[i]
                
                param_range = np.linspace(lower_bound, upper_bound, n_divisions)
                modified_values = [maximal_nominal.copy() for _ in range(n_divisions)]
                
                
                for j, div in enumerate(param_range):
                    modified_values[j][i]=div
                                        
                prediction = self.model_predict(modified_values)
                # Save as histogram
                
                plt.plot(param_range, prediction)
                plt.xlabel(self.chain.plot_branches[i])
                plt.ylabel("-2*loglikelihood")
                pdf.savefig()
                plt.close()


    # create test model for classifiers:
    def test_model_class(self, path_l, adapt, exp, norm, model):
        """Test model

        :raises ValueError: No model set
        :raises ValueError: No test data set 
        """        
        if self._model is None:
            raise ValueError("No Model has been set!")

        if self._test_data is None or self._test_labels is None:
            raise ValueError("No test data set")

        chain_N = len(path_l)
        print(f'Number of independent MCMCs: {chain_N}')
        print("Training Results!")

        train_prediction = model.model_predict(self._training_data)
        #train_prediction = model.model_predict(self.scale_data(self._training_data))
        train_as_numpy = model.scale_labels(self._training_labels)#.T[0]
        
        
        self.evaluate_model_class(train_prediction, train_as_numpy, chain_N, path_l, adapt, exp, norm)

        print(f'unscaled data: {self._training_data}')
        print(f'scaled data: {self.scale_data(self._training_data)}')
     
       
       
        print("=====")
        print("Testing Results!")

        print(f'train prediction:{train_prediction}')
        print(f'train true:{train_as_numpy}')
        #test_prediction = model.model_predict(self.scale_data(self._test_data))

        test_prediction = model.model_predict(self._test_data)
        test_as_numpy = model.scale_labels(self._test_labels)#.T[0]

        
        self.evaluate_model_class(test_prediction, test_as_numpy, chain_N, path_l, adapt, exp, norm)
        print("=====")  

        self.plot_loss(model.model,  self.scale_labels(self._test_labels), self._test_data)
        #self.plot_loss(model.model,  self.scale_labels(self._test_labels), self.scale_data(self._test_data))
        #print(f'predicted proba:{model.predict_proba(self.scale_data(self.test_data))}')
                
        print(f'test prediction:{test_prediction}')
        print(f'test true:{test_as_numpy}')
    
        

        
    def evaluate_model_class(self, pred_vals, true_vals, num, mylist, adapt, exp, norm ):
        #assigns an index to each .root file (in ascending, starting from 0) and then puts it into a list
        #this list is then used as input for the labelling for the Confusion Matrixt_fix_no_only.root', '../models/demo_files 2/NoAdapt/T2K/mcmc_NoAdapt_T2K_all_fixed.root', '../models/demo_files 2/NoAdapt/T2K/mcmc_NoAdapt_T2K_all_on.root']:
        x_val=[mylist[0], mylist[1]]
        x_val.append(mylist) 

        if adapt == 'NoAdapt' or 'noadapt' and exp == 'T2K' or 't2k':
            for m in mylist:
                bef, sep, after = m.partition('/NoAdapt/T2K/mcmc_')
                x_val.append(after)      

        elif adapt == 'NoAdapt' or 'noadapt' and exp == 'NOVA' or 'nova' or 'NOvA':
            for m in mylist:
                bef, sep, after = m.partition('/NoAdapt/NOvA/mcmc_')
                x_val.append(after)

        elif adapt == 'Adapt' or 'adapt' and exp == 'NOVA' or 'nova' or 'NOvA':
            for m in mylist:
                bef, sep, after = m.partition('/Adapt/NOvA/mcmc_')
                x_val.append(after)    

        #elif adapt == 'Adapt' or 'adapt' and exp == 'T2K' or 't2k':
            #for m in mylist:
                #bef, sep, after = m.partition('/Adapt/T2K/mcmc_')

                           
        
        conf_matrix = confusion_matrix(true_vals, pred_vals)    
        if norm == True:
            cm_normalized = self.normalise_confusion_matrix(conf_matrix, 'all')

            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues')
            plt.ylabel('Actual', fontsize=13)
            plt.title('Confusion Matrix', fontsize=17, pad=12)
            plt.gca().xaxis.set_label_position('top') 
            plt.xlabel('Prediction', fontsize=13)
            plt.gca().xaxis.tick_top()
            plt.xticks(rotation=45)
            plt.show()

        elif norm == False:
            sns.heatmap(conf_matrix, 
                annot=True,
                fmt='g')
                    
            plt.ylabel('Actual', fontsize=13)
            plt.title('Confusion Matrix', fontsize=17, pad=12)
            plt.gca().xaxis.set_label_position('top') 
            plt.xlabel('Prediction', fontsize=13)
            plt.gca().xaxis.tick_top()
            plt.xticks(rotation=45)
            plt.show()

        elif norm == 'None' or 'none':
            pass


        #calc of R*
        pred_acc = classification_report(true_vals, pred_vals, target_names=None, output_dict=True)['accuracy']
        print(f'predictive accuracy: {pred_acc}')
        R_Star=  pred_acc *num
        print(f'R*: {R_Star}')
        return R_Star
    


    def normalise_confusion_matrix(self, cm, normalise):

        if normalise == 'true':
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        elif normalise == 'pred':
            cm_normalized = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
        elif normalise == 'all':
            cm_normalized = cm.astype('float') / cm.sum()
        else:
            raise ValueError("Unknown normalization type. Use 'true', 'pred', or 'all'.")
        
        return cm_normalized

    def getRStar_vals(self, path_l, model):
 
        """Test model

        :raises ValueError: No model set
        :raises ValueError: No test data set 
        """        
        if self._model is None:
            raise ValueError("No Model has been set!")

        if self._test_data is None or self._test_labels is None:
            raise ValueError("No test data set")

        chain_N = len(path_l)
        print(f'Number of independent MCMCs: {chain_N}')
        print("Training Results!")

        train_prediction = model.model_predict(self._training_data)
        #train_prediction = model.model_predict(self.scale_data(self._training_data))
        train_as_numpy = model.scale_labels(self._training_labels)#.T[0]
        
        
        
        print("=====")
        print("Testing Results!")

        print(f'train prediction:{train_prediction}')
        print(f'train true:{train_as_numpy}')
        #test_prediction = model.model_predict(self.scale_data(self._test_data))

        test_prediction = model.model_predict(self._test_data)
        test_as_numpy = model.scale_labels(self._test_labels)#.T[0]
        print("=====")  
                    
        print(f'test prediction:{test_prediction}')
        print(f'test true:{test_as_numpy}')
        #calc of R*
        pred_acc = classification_report(test_as_numpy, test_prediction, target_names=None, output_dict=True)['accuracy']
        print(f'predictive accuracy: {pred_acc}')
        R_Star=  pred_acc * len(path_l)
        print(f'R*: {R_Star}')
        return R_Star


    '''
    def getRStar_vals(self, path_l, datapoints, model):
        chain_N = len(path_l) 
        y_value2= []
        for f in range(datapoints):
            """Trains model

            :raises ValueError: Model not initialised
            :raises ValueError: Data set not initialised
            """        
            
            print(f"Training Model")
                
            if self._model is None:
                raise ValueError("No Model has been set!")
                
            if self._training_data is None or self._training_labels is None:
                raise ValueError("No test data set")
                
            self._model.fit(self._training_data, self._training_labels)


            train_prediction = model.model_predict(self._training_data)
            #train_prediction = model.model_predict(self.scale_data(self._training_data))
            train_as_numpy = model.scale_labels(self._training_labels)#.T[0]        
        
            print("=====")
            print("Testing Results!")

            print(f'train prediction:{train_prediction}')
            print(f'train true:{train_as_numpy}')
            #test_prediction = model.model_predict(self.scale_data(self._test_data))

            test_prediction = model.model_predict(self._test_data)
            test_as_numpy = model.scale_labels(self._test_labels)#.T[0]

            #self.R_star = classification_report(test_as_numpy, test_prediction, target_names=None, output_dict=True)['accuracy']*chain_N
            R_star = classification_report(test_as_numpy, test_prediction, target_names=None, output_dict=True)['accuracy']*chain_N
            print(R_star)
            if len(path_l) == 2:
                y_value2.append(R_star)            
            elif len(path_l) == 4:
                y_value2.append(R_star)
            elif len(path_l) == 8:           
                y_value2.append(R_star)      
            elif len(path_l) == 16:
                y_value2.append(R_star) 
        return y_value2
    '''



    def plot_loss(self, model, y_test, X_test):
        train_loss = model.train_score_
        train_loss_list = train_loss.tolist()

        train_loss_list = [abs(x) for x in train_loss_list]# abs val

        x_vals= list(range(1,len(train_loss_list)+1 ))
        test_losses = [
        log_loss(y_test, proba)        
        for proba in model.staged_predict_proba(X_test)]

        test_losses = [abs(x) for x in test_losses]# abs val

        x_vals2= list(range(1,len(test_losses)+1 ))


        w, h = 12, 8             
        margin = 2

        # create a 1×2 grid of axes
        fig, axes = plt.subplots(nrows=1, ncols=2,
                                figsize=(w, h),
                                facecolor='lightblue')

        fig.subplots_adjust(
            left=margin/w,         # space on left
            right=1 - margin/w,    # space on right
            bottom=margin/h,       # space on bottom
            top=1 - margin/h,      # space on top
            wspace=margin/h        # horizontal gap between plots
        )

        # now you can plot into each axis:
        axes[0].plot(x_vals, train_loss_list, label= 'Training Loss')
        axes[0].plot(x_vals2, test_losses, label='Testing loss')
        axes[0].set_title('Loss Function')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].legend()


        axes[1].loglog(x_vals, train_loss_list, label= 'Training loss')
        axes[1].loglog(x_vals2, test_losses, label='Testing loss')
        axes[1].set_title('Loss Function - Log')
        axes[1].set_xlabel('Epochs - Log')
        axes[1].set_ylabel('Loss - Log')
        axes[1].legend()
        plt.legend()
        plt.show()



    def evaluate_model(self, predicted_values: Iterable, true_values: Iterable, outfile: str=""):
        """Evalulates model

        :param predicted_values: Label values predicted by model
        :type predicted_values: Iterable
        :param true_values: Actual label values
        :type true_values: Iterable
        :param outfile: File to output plots to, defaults to ""
        :type outfile: str, optional
        """                
        
        print(f"[bold red3]Mean Absolute Error :[/bold red3] [yellow3]{metrics.mean_absolute_error(predicted_values,true_values)}")
        
        outfile_name = outfile.split(".")[0]
        outfile = f"{outfile_name}.pdf"
        warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")
        lobf = np.poly1d(np.polyfit(predicted_values, true_values, 1))
        
        print(f"[bold purple]Line of best fit :[/bold purple] [dodger_blue1]y={lobf.c[0]}x + {lobf.c[1]}")
        
        fig = plt.figure()
        
        
        ax = fig.add_subplot(1,1,1, projection='scatter_density')
        
        # Bit hacky put plotting code is... bad so we're going to ignore the error it raises! 
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        density = ax.scatter_density(predicted_values, true_values, cmap=self.white_viridis)
        # warnings.resetwarnings()
        
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
            is_notebook = self.is_notebook()
            if is_notebook:
                plt.show()
        except Exception:
            ...
            
        
        plt.close()
        
        # Gonna draw a hist
        difs = true_values-predicted_values
        print(f"mean: {np.mean(difs)}, std dev: {np.std(difs)}")
        plt.hist(difs, bins=100, density=True, range=(np.std(difs)*-5, np.std(difs)*5))
        plt.xlabel("True - Pred")
        plt.savefig(f"diffs_5sigma_range_{outfile}")
        
        plt.close()    



    @classmethod
    def is_notebook(cls) -> bool:
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True   # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return False  # Terminal running IPython
            else:
                return False  # Other type (?)
        except NameError:
            return False      # Probably standard Python interpreter

