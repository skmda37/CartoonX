import sys
import torch
from pytorch_wavelets import DWTForward, DWTInverse


class CartoonRDE:
    def __init__(self, model, device, batch_size, num_steps, step_size, l1lambda, wave, mode, J,
                 distortion_measure="label", obfuscation_strategy="gaussian-adaptive-noise", init_mask="ones", return_logs=False):
        """
        args:
            model: classifier to be explained
            device: gpu or cpu
            batch_size: int - number of samples to approximate expected distortion
            num_steps: int - number of optimization steps for mask
            step_size: float - step size for adam optimizer on mask
            l1lambda: float - Lagrange multiplier for l1 norm of mask
            wave: str - wave type for DWT e.g. "db3"
            mode: str - mode for DWT e.g. "zero"
            J: int - number of scales for DWT
            distortion_measure: str - identifier of distortion measure function
            obfuscation_strategy: str - either "gaussian-adaptive-noise" or "zero"
            return_logs: bool - return logs for losses besides explanation if true
        """
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.step_size = step_size
        self.l1lambda = l1lambda
        self.forward_dwt = DWTForward(J=J, mode=mode, wave=wave).to(device)
        self.inverse_dwt = DWTInverse(mode=mode, wave=wave).to(device)
        self.softmax = torch.nn.Softmax(dim=1)
        self.distortion_measure = distortion_measure
        self.obfuscation_strategy = obfuscation_strategy
        self.init_mask = "ones"
        self.return_logs = return_logs

    def step(self, std_yl, mean_yl, std_yh, mean_yh, yl, yh, s_yl, s_yh, score, target, num_mask_entries):
        """
        args:
            std_yl: float - standard deviation for noise perturbation of yl coefficients
            mean_yl: float - mean for noise perturbation of yl coefficients
            std_yh: list - list of standard deviations for noise perturbation of yh coefficients
            mean_yh: list - list of means for noise perturbation of yh coefficients
            yl: torch.Tensor -  LL band DWT coefficients
            yh: list of torch.Tensor instances -  YH band DWT coefficients
            s_yl: torch.Tensor - mask over coefficients in yl
            s_yh: list of torch.Tensor instances - list of masks over coefficients in yh respectively
            score: float or torch.Tensor - typical choice is label probability for original image
            or all probabilities of original image
            target: int or torch.Tensor - index for target label if int else output probabilities
            num_mask_entries: int - number of entries of the mask
        """
        # Generate noise for yl coefficients with standard deviation std_yl
        n_yl = std_yl * torch.randn((self.batch_size, *yl.shape[1:]),
                                     dtype=torch.float32,
                                     device=self.device,
                                     requires_grad=False) + mean_yl

        # Get obfuscation for yl coefficients
        obf_yl = s_yl*yl + (1-s_yl)*n_yl

        # Initialize list of obfuscations for yh coefficients
        obf_yh = []
        for count, y in enumerate(yh):

            # Generate noise for yh coefficients with standard deviation in  std_yh and mean in_mean_yh
            n_y = std_yh[count] * torch.randn((self.batch_size, *y.shape[1:]),
                                               dtype=torch.float32,
                                               device=self.device,
                                               requires_grad=False) + mean_yh[count]

            # Get obfuscation for yh coefficients
            obf_yh.append(s_yh[count]*y + (1-s_yh[count])*n_y)

        # Inverse wavelet transform of obfuscation
        assert obf_yl.shape == (self.batch_size, *yl.shape[1:]) and obf_yh[0].shape == (self.batch_size, *yh[0].shape[1:])
        z = self.inverse_dwt((obf_yl,obf_yh))
        z = z.clamp(0,1) # We clamp overflowing values after inverse DWT back to [0,1] 

        # Get model score for the obfuscation z
        new_preds = self.model(z)
        assert len(new_preds.shape) == 2

        # Compute distortion between score for obfuscation z and and score for original image
        if self.distortion_measure == "label" or self.distortion_measure == "maximize-target":
            # Compute distortion in the predicted target label (for maximize-target score=1 otherwise "score=labelprobabilty")
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



        # Compute sparsity loss for s_yl 
        sparsity = torch.sum(torch.abs(s_yl))

        # Compute sparsity loss for masks in s_yh
        for s in s_yh:
            sparsity += torch.sum(torch.abs(s))

        # Divide sparsity loss with number of mask entries: adjust lambda if you don't want to normalize
        sparsity /= num_mask_entries

        return distortion, sparsity
    
    def __call__(self, x, target):
        """
        args:
            x: torch.Tensor of shape (1,C,H,W) - input image to be explained
            label: int or None - label index where distortion is measured or None if distortion is measured in all output probabilities
        """
        # Assert image has shape (1,C,H,W)
        assert len(x.shape) == 4
        x.requires_grad_(False)
        
        # Initialize list for logs
        logs = {"l1-norm": [], "distortion": [], "loss": []}
        
        """
        Do forward DWT
        """

        # Do forward DWT for images
        yl, yh = self.forward_dwt(x)  # yl.shape = (1, 3, ?, ?) yh[i].shape = (1, 3, 3 ?, ?)

        # Get DWT for greyscale image
        yl_grey, yh_grey = self.forward_dwt(x.sum(dim=1, keepdim=True)/3)
        
        """
        Initialize obfuscation strategy
        """
        if self.obfuscation_strategy == "gaussian-adaptive-noise":
            # Compute standard deviation and mean for adaptive Gaussian noise (this is the obfuscation strategy we use in our paper)
            std_yl = torch.std(yl)
            mean_yl = torch.mean(yl)
            std_yh = []
            mean_yh = []
            for y in yh:
                std_yh.append(torch.std(y))
                mean_yh.append(torch.mean(y))
        elif self.obfuscation_strategy == "zero":
            std_yl = 0
            mean_yl = 0 
            std_yh = []
            mean_yh = []
            for y in yh:
                std_yh.append(0)
                mean_yh.append(0)
        else:
            raise NotImplementedError(f"Obfuscation strategy {self.obfuscation_strategy} was not implemented")


        """
        Initialize DWT mask 
        """
        if self.init_mask == "ones":
            # Get mask for yl coefficients
            yl.requires_grad_(False).to(self.device)
            s_yl = torch.ones((1, *yl.shape[2:]),
                              dtype=torch.float32,
                              device=self.device,
                              requires_grad=True)

            # Get mask for yh coefficients
            s_yh = []
            for y in yh:
                y.requires_grad_(False).to(self.device)
                s_yh.append(torch.ones((1, *y.shape[2:]),
                                       dtype=torch.float32,
                                       device=self.device,
                                       requires_grad=True))

            # Get total number of mask entries
            num_mask_entries = s_yl.shape[-1] * s_yl.shape[-2]
            for s in s_yh: 
                num_mask_entries += s.shape[-1] * s.shape[-2] * s.shape[-3]
        elif self.init_mask == "rand":
            # Get mask for yl coefficients
            yl.requires_grad_(False).to(self.device)
            s_yl = torch.rand((1, *yl.shape[2:]),
                              dtype=torch.float32,
                              device=self.device,
                              requires_grad=True)

            # Get mask for yh coefficients
            s_yh = []
            for y in yh:
                y.requires_grad_(False).to(self.device)
                s_yh.append(torch.rand((1, *y.shape[2:]),
                                       dtype=torch.float32,
                                       device=self.device,
                                       requires_grad=True))

            # Get total number of mask entries
            num_mask_entries = s_yl.shape[-1] * s_yl.shape[-2]
            for s in s_yh: 
                num_mask_entries += s.shape[-1] * s.shape[-2] * s.shape[-3]
        else:
            raise NotImplementedError(f"mask initialization {self.init_mask} was not implemented.")
            

        """
        Initialize Optimizer for mask
        """
        optimizer = torch.optim.Adam([s_yl]+s_yh, lr=self.step_size)
        
        
        """
        Get score for original image
        """
        if self.distortion_measure == "label":
            # Measure distortion as squared difference in target label from obfuscation to original
            score = self.softmax(self.model(x.detach()).detach())[:, target].detach()
        elif self.distortion_measure == "maximize-target":
            # Measure distortion as squared difference in target label from obfuscation to 1, i.e. the maximal possible score
            score = 1
        elif self.distortion_measure == "l2":
            # Measure distortion as 
            assert target is None
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
            Compute distortion and sparasity
            """
            distortion, sparsity = self.step(std_yl, mean_yl, std_yh, mean_yh, yl,
                                             yh, s_yl, s_yh, score, target, num_mask_entries)

            # Compute loss
            loss = distortion + self.l1lambda * sparsity 
            
            # Log loss terms
            logs["distortion"].append(distortion.detach().item())
            logs["l1-norm"].append(sparsity.detach().item())
            logs["loss"].append(loss.detach().item())
            
            # Perform optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Clamp masks into [0,1]
            with torch.no_grad():
                s_yl.clamp_(0,1)
                for s_y in s_yh:
                    s_y.clamp_(0,1)
                    
        """
        Invert DWT mask back to pixel space
        """
        cartoonRDE= self.inverse_dwt((s_yl.detach()*yl_grey, 
                                       [s.detach()*y for s,y in zip(s_yh, yh_grey)]))
        # We take absolute value since 0 values in pixel space needs to be smallest values. 
        # We also clamp into 0,1 in case there was an overflow, i.e. pixel values larger than 1 after the inverse dwt
        cartoonRDE= cartoonRDE.squeeze(0).clamp_(0,1)
        
        if self.return_logs:
            return cartoonRDE, logs
        else: 
            return cartoonRDE 
