#ALL IMPORTS
# MaCh3Python Deps
from MaCh3PythonUtils.file_handling.chain_handler import ChainHandler
from MaCh3PythonUtils.machine_learning.ml_factory import MLFactory

# Other imports
from matplotlib import pyplot as plt
from pathlib import Path
import gdown
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import *
from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np
from pathlib import Path
import pandas as pd
import tqdm

#ALL DEFINED FUNCTIONS
#new version to try and work with combo masses
def get_mean_for_col(data_frame: pd.DataFrame, col_we_care_about: str, greater_than: bool, split_col="delm2_23"):
    if greater_than:
        return np.mean(data_frame[data_frame[split_col]>0][col_we_care_about])
    else:
        return np.mean(data_frame[data_frame[split_col]<0][col_we_care_about])

def do_grid_search(ml_model):
    #OPTIONAL Do the grid search over the training data using the params below
    param_grid = {
        "learning_rate":(0.001,0.01,0.1,1),
        "max_leaf_nodes":(50,100,200)
    }
    grid_search = RandomizedSearchCV(estimator=HistGradientBoostingRegressor(), param_distributions=param_grid, n_iter=10)
    training_data = ml_model.training_data #all x_sec columns etc
    target_data = ml_model._training_labels #log-likelihood 

    grid_search.fit(training_data, target_data)
    fit_params = grid_search.best_params_
    print("Best params found by grid search:")
    print(fit_params)

    max_leaves = fit_params['max_leaf_nodes']
    lr = fit_params['learning_rate']

    return max_leaves, lr

def combo_ml_test_function(): 
    chain_handler = 0
    # Set up file properties
    chain_name = 'posteriors'
    verbose=False

    chain_handler = ChainHandler(str(input_file), chain_name, verbose=verbose)
    # Firstly we set things we want to ignore, these are parameters we really don't care about/want to learn!
    chain_handler.ignore_plots(["LogL_systematic_xsec_cov", "Log", "LogL_systematic_osc_cov", "xsec_20", "xsec_19"])
    fitting_label = "LogL" #what target we want
    # Now we add parameters we care about. Note this only need to be a substring of the full name
    chain_handler.add_additional_plots(["sin2th", "delm2", "delta", "xsec"])
    # We need to make sure the chain handler knows this exists, passing true means it is looking for some with that exact name
    chain_handler.add_additional_plots(fitting_label, True)

    #specify the cut for the mass being tested
    chain_handler.add_new_cuts(["LogL_systematic_xsec_cov<1234", "step>10000"])
    chain_handler.convert_ttree_to_array()

    #setup BCs SEE IF WE WANT TO GO PM 10% AGAIN OR NOT
    no_table = chain_handler.ttree_array[chain_handler.ttree_array["delm2_23"]>0]
    max_no = no_table.max().to_numpy(dtype=np.float64)[:-1].copy()
    min_no = no_table.min().to_numpy(dtype=np.float64)[:-1].copy()

    io_table = chain_handler.ttree_array[chain_handler.ttree_array["delm2_23"]<0]
    max_io = io_table.max().to_numpy(dtype=np.float64)[:-1].copy()
    min_io = io_table.min().to_numpy(dtype=np.float64)[:-1].copy()

    model_output=Path("../models/my_model")
    # Make sure the model output directory exists
    model_output.parent.mkdir(parents=True, exist_ok=True)
    # The factory produces ML models, we need to pass it the chain handler and the fitting label to get started
    ml_factory = MLFactory(chain_handler, fitting_label, f"{model_output}.pdf")
    
    #Initialise the model here
    ml_model = ml_factory.make_interface('SciKit', 'histboost', loss='absolute_error', max_iter=10000, verbose=1, max_leaf_nodes=40) 
    # Let's use 20% of the data for testing 
    ml_model.set_training_test_set(0.2)
        
    training_data = ml_model.training_data #all x_sec columns etc
    #target_data = ml_model._training_labels #log-likelihood    
    
    #OPTIONAL perform grid search to get the optimal parameters  
    #max_leaves, lr = do_grid_search(ml_model)

    max_leaves = 200
    lr = 0.01
    ml_model = ml_factory.make_interface('SciKit', 'histboost', loss='absolute_error', 
                                        max_iter=10000, verbose=1, max_leaf_nodes=max_leaves, learning_rate=lr) 
    
    #train the ML model
    ml_model.set_training_test_set(0.2)
    ml_model.train_model()
    
    #make train and test plots for the ML model for completeness
    ml_model.test_model()
    
    return ml_model,training_data,min_no,max_no,min_io,max_io

#function to do MCMC for the parameters
def v2_run_mcmc(ml_model,training_data,min_no,max_no,min_io,max_io):

    NSTEPS: int = 10000
    NCHAINS = 10
    IN_BOUND_COUNT = np.zeros(NCHAINS)
    OUT_BOUND_COUNT = np.zeros(NCHAINS)
    ACCEPT_COUNT = np.zeros(NCHAINS)
    REJECT_COUNT = np.zeros(NCHAINS)

    min_no = min_no.copy()-(0.1*min_no*np.sign(min_no))
    max_no = max_no.copy()+(0.1*max_no*np.sign(max_no))
    min_io = min_io.copy()-(0.1*min_io*np.sign(min_io))
    max_io = max_io.copy()+(0.1*max_io*np.sign(max_io))

    #print(min_io[0],min_io[25])
    #print(max_io[0],max_io[25])

    start_step = training_data.iloc[0].to_numpy() #but this has 29 items
    start_step_many = np.tile(start_step, (NCHAINS,1))

    stored_steps = np.empty((NSTEPS+1, NCHAINS, len(start_step)))

    #stored_steps[0] = start_step #so this initialises with 27 items
    stored_steps[0] = start_step_many
    #so when we accept new points we need to have all parameters considered and their new values included

    #print(start_step_many)

    current_L = ml_model.model_predict_unscale(start_step_many)
    #print(current_L)
    stored_L = np.empty((NSTEPS+1,NCHAINS,len(start_step)))
    stored_L[0] = current_L

    col_labels = list(training_data.columns)

    mu_1 = []
    mu_2 = []

    #find the means of the IO and NO peaks, mu_1 and mu_2 respectively, for each row
    for p in range(0,len(col_labels)):
        mu_1.append(get_mean_for_col(training_data,col_labels[p],False)) #IO
        mu_2.append(get_mean_for_col(training_data,col_labels[p],True)) #NO

    mu_1 = np.array(mu_1)
    mu_2 = np.array(mu_2)
    step_sizes = np.ones(len(start_step))*(2.38**2)/(len(start_step))*(np.array(mu_1)**2)*0.2

    flipturn = 0

    flip_random_nums = np.random.uniform(0,1, NSTEPS)
    accept_u = np.random.uniform(0,1,NSTEPS)

    #step_proposals = np.random.normal(0, step_sizes, (NSTEPS, len(step_sizes), NCHAINS))
    step_proposals = np.random.normal(0, step_sizes.reshape(1, 1, len(step_sizes)), (NSTEPS, NCHAINS, len(step_sizes)))


    for s in tqdm.tqdm_notebook(range(NSTEPS)):
        x_t = stored_steps[s]    
        new_priors = x_t + step_proposals[s] #proposed priors

        #use mu to flip the mass ordering on a 50% chance basis
        #this should stop the MCMC getting stuck in local IO min.
        # R = np.random.uniform(0,1)
        R = flip_random_nums[s]
        #print(R)

        if R<0.5: #FLIP MASS ORDERING
           new_priors = (mu_1 - new_priors) + mu_2
           #print('flipped priors: '+str(new_priors))
           flipturn += 1

        proposed_L = ml_model.model_predict(new_priors) #returns logL based on new set of priors
        #proposed_L = ml_model.model_predict_unscale(new_priors) #returns logL based on new set of priors
        current_L = current_L.flatten()
        proposed_L = proposed_L.flatten()

        #begin looping over multiple chains
        for ith_chain in range(0,NCHAINS):
            #print(np.all(min_io<new_priors[ith_chain]))

            in_io = np.all(min_io<new_priors[ith_chain]) and np.all(new_priors[ith_chain]<max_io)
            in_no = np.all(min_no<new_priors[ith_chain]) and np.all(new_priors[ith_chain]<max_no)

            if in_io or in_no:
                delta_L = current_L[ith_chain]-proposed_L[ith_chain]
                accept_prob = min(np.exp(delta_L), 1)

                #accept_prob = min(np.exp(current_L[ith_chain]-proposed_L[ith_chain]), 1)
                IN_BOUND_COUNT[ith_chain] += 1
                #print('in bounds')
            else:
                accept_prob = 0
                OUT_BOUND_COUNT[ith_chain] += 1
                #print('out of bounds')
        
        # u = np.random.uniform(0,1) #uniform dist of u to compare to acceptance prob
            u = accept_u[s]
            if u <= accept_prob: #accept proposal
                stored_steps[(s+1,ith_chain)] = new_priors[ith_chain]
                current_L[ith_chain] = proposed_L[ith_chain]
                #print('ACCEPTED')
                ACCEPT_COUNT[ith_chain] += 1
                
            if u > accept_prob: #reject proposal
                stored_steps[(s+1,ith_chain)] = stored_steps[(s,ith_chain)]
                #print('REJECTED')
                #print(f'param: {param}, io low: {iom}, io up: {iou}, no low: {nom}, no up: {nou}')
                REJECT_COUNT[ith_chain] += 1

            stored_L[(s+1,ith_chain)] = current_L[ith_chain] #update log-likelihood

    for i in range(0,NCHAINS):    
        print('\n')
        print(f'### {i} Chain ###')
        print('in bound count: '+str(IN_BOUND_COUNT[i]))
        print('out bound count: '+str(OUT_BOUND_COUNT[i]))
        print('accept count: '+str(ACCEPT_COUNT[i]))
        print('reject count: '+str(REJECT_COUNT[i]))
        print('flipped count: '+str(flipturn))

    return stored_steps, stored_L

def get_table_hacky():
    chain_name = 'posteriors'
    verbose=False
    chain_handler = ChainHandler(str(input_file), chain_name, verbose=verbose)
    # Firstly we set things we want to ignore, these are parameters we really don't care about/want to learn!
    chain_handler.ignore_plots(["LogL_systematic_xsec_cov", "Log", "LogL_systematic_osc_cov", "xsec_20", "xsec_19"])
    fitting_label = "LogL" #what target we want
    # Now we add parameters we care about. Note this only need to be a substring of the full name
    chain_handler.add_additional_plots(["sin2th", "delm2", "delta", "xsec"])
    # We need to make sure the chain handler knows this exists, passing true means it is looking for some with that exact name
    chain_handler.add_additional_plots(fitting_label, True)

    #specify the cut for the mass being tested
    chain_handler.add_new_cuts(["LogL_systematic_xsec_cov<1234", "step>10000"])
    chain_handler.convert_ttree_to_array()
    full_table = chain_handler.ttree_array

    return full_table.copy()

def do_bayes_factor(required_chain, label):
    #Find Bayes' factor for original MCMC fit
    NO_output = [p for p in required_chain if p>0]
    IO_output = [q for q in required_chain if q<0]
    bayes_factor = len(NO_output)/len(IO_output)
    print(f'{label} Bayes\' factor: {bayes_factor:.3f}')


#MAIN STARTS HERE ##############################
# Download file from google drive
file_url="https://drive.google.com/file/d/1iE6xFhn3BH_HnLUfQ7KFGy2wfeH52Rwf/view?usp=sharing"
# download the file
input_file = Path("../models/demo_chain.root")
if not input_file.exists():
    # download the file
    input_file.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(file_url, str(input_file), quiet=False, fuzzy=True)

#TRAIN BDT
ml_model,training_data,min_no,max_no,min_io,max_io = combo_ml_test_function()

#PERFORM MCMC USING BDT PREDICTIONS
mcmc_chain,mcmc_log_like = v2_run_mcmc(ml_model,training_data,min_no,max_no,min_io,max_io)

#PLOT TRACE AND HIST FOR MASS AND XSEC PARAMS
#adapt below for your own file path
file_save_path = '/Users/lucie/Documents/Summer25/MaCh3-PythonUtils/notebooks/output_plots/'

cutoff = 2000 #minimum burn in cutoff
x_0 = mcmc_chain[:,:,0][cutoff:,:].flatten()
delM_values = mcmc_chain[:,:,25][cutoff:,:].flatten() #apply same thing to the og data to compare

fig, (trace_ax, post_ax) = plt.subplots(nrows=1, ncols=2)
x = np.arange(0,len(delM_values),1)
trace_ax.plot(x, delM_values, linewidth=0.5, color='r')
trace_ax.set_ylabel(r'$\Delta \text{m}^2_{23}$')
trace_ax.set_xlabel('Step')
##
post_ax.hist(delM_values, bins=50, color='r', alpha=0.5, orientation='horizontal', density=True, label=' MCMC Result \n (Combined Chains)')
post_ax.set_xlabel('Probability Density')
# Merge the two axes
plt.setp(post_ax.get_yticklabels(), visible=False)
fig.subplots_adjust(left=0.15,wspace=.08)
post_ax.legend()
fig.savefig(file_save_path+'delm2_23.pdf',format='pdf')
plt.show()

figA, (trace_ax, post_ax) = plt.subplots(nrows=1, ncols=2)
x = np.arange(0,len(x_0),1)
trace_ax.plot(x, x_0, linewidth=0.5, color='darkorange')
trace_ax.set_ylabel(r'$\text{xsec}_{0}$')
trace_ax.set_xlabel('Step')
##
post_ax.hist(x_0, bins=50, color='darkorange', alpha=0.5, orientation='horizontal', density=True, label=' MCMC Result \n (Combined Chains)')
post_ax.set_xlabel('Probability Density')
# Merge the two axes
plt.setp(post_ax.get_yticklabels(), visible=False)
figA.subplots_adjust(left=0.13,wspace=.1)
post_ax.legend()
figA.savefig(file_save_path+'xsec_0.pdf',format='pdf')
plt.show()

#COMPARE TO TRAINING DATA
og_hist = training_data.iloc[:,25][cutoff:]
#og_hist = [val for val in og_hist if val>0] #use this if we are NOT FLIPPING
bin_vals = np.linspace(min(og_hist),max(og_hist),50)
#print(og_hist)
fig2, ax = plt.subplots(nrows=1, ncols=1)
ax.hist(delM_values, bins=bin_vals, color='r', alpha=0.5, orientation='horizontal', density=True, label=' MCMC Result \n (Combined Chains)')
ax.hist(og_hist, bins=bin_vals, color='b', alpha=0.5, orientation='horizontal', density=True, label='Training Data')
ax.set_xlabel('Probability Density')
ax.set_ylabel(r'$\Delta \text{m}^2_{23}$')
fig2.subplots_adjust(left=0.15)
ax.legend()
fig2.savefig(file_save_path+'delm2_23_training_comparison.pdf',format='pdf')
plt.show()

#COMPARE TO INITIAL MCMC RESULT
ttree_table = get_table_hacky()
#print(ttree_table)
desired_col = ttree_table.iloc[:,25][cutoff:]
#desired_col = [val for val in desired_col if val>0] #use this if we are NOT FLIPPING
bin_vals = np.linspace(min(desired_col),max(desired_col),50)

fig3, ax = plt.subplots(nrows=1, ncols=1)
ax.hist(delM_values, bins=bin_vals, color='r', alpha=0.5, orientation='horizontal', density=True, label=' MCMC Result \n (Combined Chains)')
ax.hist(desired_col, bins=bin_vals, color='#00A36C', alpha=0.5, orientation='horizontal', density=True, label='Original MCMC Fit')
ax.set_xlabel('Probability Density')
ax.set_ylabel(r'$\Delta m^2_{23}$')
ax.legend()
fig3.subplots_adjust(left=0.15)
fig3.savefig(file_save_path+'delm2_23_mcmc_comparison.pdf',format='pdf')
plt.show()

#PLOT HIST FOR EVERY PARAM
param_names = list(ttree_table.columns)
for p in range(0,27):
    #mcmc_col = [m[p] for m in mcmc_chain]
    mcmc_col = mcmc_chain[:,:,p][cutoff:,:].flatten()
    ttree_col = ttree_table.iloc[:,p][cutoff:]
    bin_vals = np.linspace(min(ttree_col),max(ttree_col),50)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.hist(mcmc_col, bins=bin_vals, color='r', alpha=0.5, orientation='horizontal', density=True, label=' MCMC Result \n (Combined Chains)')
    ax.hist(ttree_col, bins=bin_vals, color='#00A36C', alpha=0.5, orientation='horizontal', density=True, label='Original MCMC Fit')
    ax.set_xlabel('Probability Density')
    ax.set_ylabel(f'{param_names[p]}')
    ax.legend()
    fig.subplots_adjust(left=0.15)
    fig.savefig(file_save_path+f'{param_names[p]}_hist.pdf',format='pdf')
    plt.show()