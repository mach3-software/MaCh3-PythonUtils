import torch
from MaCh3PythonUtils.machine_learning.file_ml_interface import FileMLInterface
from tqdm.notebook import tqdm

class TorchMCMC:
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")


    def __init__(self, model: FileMLInterface, num_steps: int, num_chains: int,
                 start_adaption: int, end_adaption: int, adapt_step: int):
        self.model = model
        self.num_steps = num_steps
        self.num_chains = num_chains

        self.ndim = self.model.chain.ndim - 1  
        self.chains = torch.empty((self.num_chains, self.num_steps, self.ndim)).to(self.device)
        self.throw_matrix = torch.eye(self.ndim).to(self.device)*0.05

        self.adaptive_matrix = torch.eye(self.ndim).to(self.device)*1e-9

        self.current_step = torch.zeros((self.num_chains, self.ndim), dtype=torch.float32).to(self.device)
        self.proposed_step = torch.zeros((self.num_chains, self.ndim), dtype=torch.float32).to(self.device)

        self.current_log_prob = torch.Tensor(model.model_predict(self.current_step)).to(self.device)
        self.proposed_log_prob = torch.Tensor(model.model_predict(self.proposed_step)).to(self.device)

        self.start_adaption = start_adaption
        self.end_adaption = end_adaption
        self.adapt_step = adapt_step

    def propose_step(self):
        self.proposed_step = torch.distributions.MultivariateNormal(self.current_step, self.throw_matrix).sample().to(self.device)
        self.proposed_log_prob = torch.tensor(self.model.model_predict(self.proposed_step)).to(self.device)

    def accept_step(self, accept_mask: torch.Tensor):
        # Accept the proposed step if accept mask[i]==True
        self.current_step[accept_mask] = self.proposed_step[accept_mask].clone().to(self.device)
        self.current_log_prob[accept_mask] = self.proposed_log_prob[accept_mask].clone().to(self.device)

    def update_adaptive_covariance(self):
        '''
        Update Adaptive Covariance using only the first chain
        '''
        # No need to be here
        if self.num_steps > self.end_adaption:
            return
        
        chain_samples = self.chains[0]
        self.adaptive_matrix = torch.cov(chain_samples.T).to(self.device) + 1e-9 * torch.eye(self.ndim).to(self.device)

        if self.num_steps == self.start_adaption or\
           (self.num_steps % self.adapt_step == 0 and self.num_steps > self.start_adaption):
           # Update the throwing matrix
            self.throw_matrix = self.adaptive_matrix

    def acceptance_criterion(self):
        # Compute the acceptance criterion

        # No need to do computation, accept

        acceptance_prob = torch.min(torch.ones(self.num_chains).to(self.device), torch.exp(self.proposed_log_prob - self.current_log_prob)).to(self.device)
        acc_prob = torch.rand(self.num_chains).to(self.device)

        return acc_prob < acceptance_prob

    def __call__(self):
        print("Starting MCMC Sampling...")
        for _ in tqdm(range(self.num_steps)):
            self.propose_step()
            accept_mask = self.acceptance_criterion()
            self.accept_step(accept_mask)
            self.update_adaptive_covariance()

    def get_chains(self):
        return self.chains

    def get_parameter_chain(self, par_index: int, burnin: int=0):
        return self.chains[:, burnin:, par_index]