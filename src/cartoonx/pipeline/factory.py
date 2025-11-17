from typing import Union

from cartoonx.modelling.explainer import WaveletBasedCartoonX
from cartoonx.modelling.explainer import ShearletBasedCartoonX


class CartoonXFactory:

    @staticmethod
    def create(
        system: str,
        **kwargs
    ) -> Union[WaveletBasedCartoonX, ShearletBasedCartoonX]:
        if system == 'wavelets':
            xpl = WaveletBasedCartoonX(**kwargs)
        elif system == 'shearlets':
            xpl = ShearletBasedCartoonX(**kwargs)
        else:
            raise ValueError(
                f'system is {system}; must be "wavelets" or "shearlets"!'
            )
        return xpl