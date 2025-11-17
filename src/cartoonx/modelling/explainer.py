from collections import namedtuple, defaultdict
from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Union

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse


class AbstractCartoonX(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def explain(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> dict:
        pass

    @abstractmethod
    def _init_mask(self, mode: str):
        pass

    @abstractmethod
    def _sample_perturbations(self, samplesize: int, mode: str):
        pass


    @abstractmethod
    def _get_spatial_penalty(
        self,
        spatial_penalty_type: Union[None, str],
        **kwargs
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def _clamp_mask(self) -> None:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


class WaveletBasedCartoonX(AbstractCartoonX):

    def __init__(
        self,
        device: str,
        dwtmode: str = 'zero',
        dwtJ: int = 5,
        dwtwave: str = 'db3'
    ):
        super().__init__()
        # Discrete Wavelet Transform (Forward Transform)
        self.fwd_dwt = DWTForward(wave=dwtwave, J=dwtJ).to(device)
        # Discrete Wavelet Transform (Inverse Transform)
        self.inv_dwt = DWTInverse(mode=dwtmode, wave=dwtwave).to(device)

    def apply_wavelet_mask(
        self,
        ml: torch.Tensor,
        mh: List[torch.Tensor]
    ) -> torch.Tensor:
        return self.inv_dwt((
            ml * self.dl, [m * d for m, d in zip(mh, self.dh)]
        ))

    def explain(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        lr: float = 1e-2,
        N: int = 300,
        mask_init: str = 'ones',
        perturbations: str = 'gaussian',
        samplesize: int = 64,
        spatial_reg_type: Union[None, str] = 'l1',
        spatial_reg: float = 1e-4,
        wavelet_reg: float = 1e-4
    ) -> dict:
        if len(x.shape) != 4:
            raise ValueError(
                f'Expected to have 4 dimensions but got input shape {x.shape}'
            )

        # dl: lowpass wavelet coeffs; dh: highpass wavelet coeffs
        dl, dh = self.fwd_dwt(x)
        self.dl, self.dh = dl, dh  # wavelet coeffs of x are constant

        # Wavelet coefficients of grayscale image
        self.dl_gray, self.dh_gray = self.fwd_dwt(
            x.sum(dim=1, keepdim=True) / 3
        )

        # Init mask for lowpass and highpass wavelet coeffs
        ml, mh = self._init_mask(mode=mask_init)

        # Initialize optimizer
        opt = torch.optim.Adam([ml] + mh, lr=lr)

        stats = defaultdict(list)
        with tqdm(total=N, colour='red', desc='Mask Optim.') as pbar:
            for i in range(N):
                opt.zero_grad()
                # Get perturbation of coeffs
                pl, ph = self._sample_perturbations(
                    samplesize=samplesize,
                    mode=perturbations
                )
                # Obfuscate coeffs with perturbation
                ol = ml[:, None] * dl[:, None] + (1 - ml[:, None]) * pl
                oh = [
                    m[:, None] * d[:, None] + (1 - m[:, None]) * p
                    for d, m, p in zip(dh, mh, ph)
                ]
                ox = self.inv_dwt((
                    ol.reshape(-1, *ol.shape[2:]),
                    [o.reshape(-1, *o.shape[2:]) for o in oh]
                )).clamp(0, 1)  # Clamp pixels that fall out of [0, 1] range
                # Cross-Entropy of target labels for obfuscated coeffs
                targets = torch.stack(samplesize * [y]).T.reshape(-1)
                CE = F.cross_entropy(F.softmax(model(ox)), targets)
                # Wavelet Mask L1 Penalty
                W = ml.abs().sum() + sum([m.abs().sum() for m in mh])
                W *= wavelet_reg / self._bs
                # Spatial Regularization penalty
                S = self._get_spatial_penalty(spatial_reg_type, ml=ml, mh=mh)
                S *= spatial_reg / self._bs
                # Loss
                L = CE + W + S
                # Optimization step
                L.backward()
                opt.step()
                # Clamp mask values into [0,1] interval
                self._clamp_mask(ml, mh)
                # Update stats
                stats['loss'].append(L.item())
                stats['cross_entropy'].append(CE.item())
                stats['spatial_penalty'].append(S.item())
                stats['wavelet_l1'].append(W.item())

                pbar.set_description_str(
                    f"CE {CE.item()} - "
                    f"WAVELET MASK L1 {ml.sum() + sum([m.sum() for m in mh])}"
                )
                pbar.update(1)

        with torch.no_grad():
            cartoonx = self.apply_wavelet_mask(ml, mh).cpu()
        return {'cartoonx': cartoonx, 'ml': ml, 'mh': mh, **stats}

    def _init_mask(self, mode: str) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if mode == 'ones':
            ml = torch.ones_like(self.dl_gray, requires_grad=True)
            mh = [torch.ones_like(d, requires_grad=True) for d in self.dh_gray]
        else:
            raise ValueError(
                'Only mask_init="ones" is implemented.'
                'Other mask inits did not work as well.'
            )
        return ml, mh

    def _sample_perturbations(
        self,
        samplesize: int,
        mode: str
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        device = self.dl.device
        if mode == 'gaussian':
            pl = torch.randn(
                (self._bs, samplesize, *self.dl.shape[1:]),
                device=device,
                requires_grad=False
            )
            ph = [
                torch.randn(
                    (self._bs, samplesize, *d.shape[1:]),
                    device=device,
                    requires_grad=False
                ) for d in self.dh
            ]
        else:
            raise ValueError(
                'We only implemented mode="gaussian" for the perturbations.'
                f' But you passed mode={mode}.'
            )
        return pl, ph

    def _get_spatial_penalty(
        self,
        spatial_penalty_type: Union[None, str],
        ml: torch.Tensor,
        mh: List[torch.Tensor]
    ) -> torch.Tensor:
        if spatial_penalty_type is None:
            S = 0.
        elif spatial_penalty_type == 'l1':
            S = self.apply_wavelet_mask(
                ml, mh
            ).reshape(self._bs, -1).norm(p=1, dim=1).sum()
        elif spatial_penalty_type == 'l2':
            S = self.apply_wavelet_mask(
                ml, mh
            ).reshape(self._bs, -1).norm(p=2, dim=1).sum()
        else:
            raise ValueError(
                f'spatial_penalty_type={spatial_penalty_type}'
                'but only None, "l1", and "l2" are supported.'
            )
        return S

    def _clamp_mask(self, ml: torch.Tensor, mh: List[torch.Tensor]):
        with torch.no_grad():
            ml.clamp_(0, 1)
            for m in mh:
                m.clamp_(0, 1)

    @property
    def _bs(self) -> int: return self.dl.shape[0]

    def __repr__(self) -> str: return 'WaveletBasedCartoonX'

"""
TODO: implement shearlet based cartoonx
"""

class ShearletBasedCartoonX(AbstractCartoonX):

    def __init__(self, device: str):
        pass

    def explain(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        lr: float = 1e-2
    ) -> dict:
        return {}

    def _init_mask(self):
        raise NotImplementedError()

    def _sample_perturbations(
        self,
        samplesize: int,
        mode: str
    ):
        raise NotImplementedError()

    @property
    def _cartoonx(self) -> torch.Tensor:
        raise NotImplementedError()

    def _get_spatial_penalty(
        self,
        spatial_penalty_type: Union[None, str]
    ) -> torch.Tensor:
        raise NotImplementedError()

    def _clamp_mask(self) -> None:
        raise NotImplementedError()

    def __repr__(self) -> str: return 'ShearletBasedCartoonX'