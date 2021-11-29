import torch
import sys


class PixelRDE:
    def __init__(self, model, device, batch_size=64, num_steps=2000, step_size=1e-3, l1lambda=20000,
                 distortion_measure="label", init_mask="ones", return_logs=False):
        """
        args:
            model: classifier to explain
            device: str - gpu or cpu
            batch_size: int - number of noise samples to approximate expected distortion
            num_steps: int  -  number of optimization steps
            l1lambda: float - Lagrange multiplier/weight of l_1 sparsity loss on mask
            distortion_measure: str - specifies which distortion function is used
            return_logs: bool - return logs of losses if true
        """
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.step_size = step_size
        self.l1lambda = l1lambda
        self.softmax = torch.nn.Softmax(dim=1)
        self.distortion_measure = distortion_measure
        self.init_mask = init_mask
        self.return_logs = return_logs
        
    def step(self, std_pixel, mean_pixel, x, s, target, score, num_mask_entries):
        """
        args:
            std_pixel: float - standard deviation for Gaussian perturbations
            mean_pixel: float - mean for Gaussian perturbations
            x: torch.Tensor - original input image
            s: torch.Tensor -  mask on input image
            target: int or torch.tensor - label index for int and  output probabilites for torch.Tensor
            score: float or torch.Tensor - target label probability for x if float, all probabilies for x if torch.Tensor
        """
        # Generate noise for yl coefficients with stand.dev. std_yh
        n = std_pixel * torch.randn((self.batch_size, *x.shape[1:]),
                              dtype=torch.float32,
                              device=self.device,
                              requires_grad=False) + mean_pixel

        # Get obfuscation for yl coefficients
        z = s*x + (1-s)*n

        # Get model score for z
        new_preds = self.model(z)
        if self.distortion_measure == "label" or self.distortion_measure == "maximize-target":
            # Measure distortion in target label probability as squared distance
            new_scores = self.softmax(new_preds)[:, target]
            # Approximate expected distortion with simple Monte-Carlo estimate
            distortion = torch.mean((score - new_scores)**2)
        elif self.distortion_measure == "l2":
            # Computes distortion as squared ell_2 norm between model outputs
            new_scores = self.softmax(new_preds)
            # Approximate expected distortion with simple Monte-Carlo estimate
            assert len(score.shape) == 2 and score.shape[-1]==1000, score
            distortion = torch.mean(torch.sqrt(((score - new_scores)**2).sum(dim=-1)))
        elif self.distortion_measure == "kl-divergence":
            new_scores = self.softmax(new_preds)
            # Compute average kl-divergence for prediction by obfuscations to original prediction
            distortion = torch.mean((new_scores *torch.log(new_scores/score)).sum(dim=-1)) 
        elif self.distortion_measure ==  "weighted-l2":
            new_scores = self.softmax(new_preds)
            distortion = self.C*torch.mean(torch.sqrt(((score - new_scores)**2).sum(dim=-1)))
        else:
            raise NotImplementedError(f"distortion measure {self.distortion_measure} was not implemented.")





        # Compute sparsity loss for s
        sparsity = torch.sum(torch.abs(s))

        # Divide sparsity with number of mask entries: adjust lambda if you don't want to normalize
        sparsity /= num_mask_entries

        return distortion, sparsity
    
    def __call__(self, x, target):
        """
        args:
            x: torch.Tensor - input image
            target: int or torch.Tensor - target label if int otherwise target output probabilities
        """
        # Assert image has shape (1,C,H,W)
        assert len(x.shape) == 4
        x.requires_grad_(False)
        
        # Initialize list for logs
        logs = {"l1-norm": [], "distortion": [], "loss": []}
        
        """
        Compute standard deviation and mean for Gaussian noise (the obfuscation strategy used in the paper)
        """
        std_pixel = torch.std(x)
        mean_pixel = torch.mean(x)
        
        """
        Initialize pixel mask
        """
        if self.init_mask=="ones":
            s = torch.ones((1, *x.shape[2:]),
                           dtype=torch.float32,
                           device=self.device,
                           requires_grad=True)
        elif self.init_mask == "rand":
            s = torch.rand((1, *x.shape[2:]),
                           dtype=torch.float32,
                           device=self.device,
                           requires_grad=True)
        else:
            raise NotImplementedError(f"mask initialzation {self.init_mask} was not implemented.")


        # Get total number of mask entries
        num_mask_entries = s.shape[-1] * s.shape[-2]
        
        """
        Initialize Optimizer for mask
        """
        optimizer = torch.optim.Adam([s], lr=self.step_size)

        """
        Compute score for input image
        """
        if self.distortion_measure == "label":
            # Measure distortion as squared difference in target label from obfuscation to original
            score = self.softmax(self.model(x.detach()).detach())[:, target]
        elif self.distortion_measure == "maximize-target":
            # Measure distortion as squared difference in target label from obfuscation to 1, i.e. the maximal possible score
            score = 1
        elif self.distortion_measure == "l2":
            # Measure distortion as squared ell_2 in output probabilities 
            score = self.softmax(self.model(x.detach()).detach())
        elif self.distortion_measure ==  "kl-divergence":
            score = self.softmax(self.model(x.detach()).detach())
            assert target is None
        elif self.distortion_measure ==  "weighted-l2":
            score = target[0]
            self.C = target[1]
        else:
            raise NotImplementedError(f"distortion measure {self.distortion_measure} was not implemented.")
        
        
        """
        Start optimizing masks
        """
        for i in range(self.num_steps):

            sys.stdout.write("\r iter %d" % i)
            sys.stdout.flush()


            """
            Compute distortion and sparsity
            """
            distortion, sparsity = self.step(std_pixel, mean_pixel, x, s, target, score, num_mask_entries)

            # Compute loss
            loss = distortion + self.l1lambda * sparsity
            
            # Log loss terms
            logs["distortion"].append(distortion.detach().clone().item())
            logs["l1-norm"].append(sparsity.detach().clone().item())
            logs["loss"].append(loss.detach().clone().item())

            # Perform optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Clamp masks into [0,1]
            with torch.no_grad():
                s.clamp_(0,1)
         
        if self.return_logs:
            # Returning the same thing twice for the makedataset.py script for the adversarial detector
            return s.detach().squeeze(0), s.detach().squeeze(0), logs
        else:
            return s.detach().squeeze(0)

    
    
    



