import torch


class AdaptiveLRClipping:
    """Adaptive Learning Rate Clipping


    Clip loss spikes, from small batches, high order loss functions, and poor
    data.
    """

    def __init__(self, beta1=0.999, beta2=0.999, n=3, mu1=25, mu2=30**2):
        assert (
            0 < beta1 < 1 and 0 < beta2 < 1
        ), "decay rates beta1 and beta2 must be between 0 and 1"
        assert mu1**2 < mu2, "mu1**2 must be less than mu2"

        self._beta1 = beta1
        self._beta2 = beta2
        self._n = n
        self._mu1 = mu1
        self._mu2 = mu2
        self._prev_loss: Optional[float] = None

    def clip(self, loss: torch.Tensor) -> torch.Tensor:
        # TODO: add support for neg losses - reqs careful mgmt of div by 0
        assert loss > 0, "ALRC currently only supports positive losses"
        sigma = torch.sqrt(self._mu2 - self._mu1**2)
        max_loss = self._mu1 + self._n * sigma + 1e-8
        if loss > max_loss:
            # do we do this rigamarol to add max_loss to the graph?
            with torch.no_grad():
                ldiv = max_loss / loss
            return ldiv * loss
        self._prev_loss = loss.item()
        return loss

    def update(self):
        assert (
            self._prev_loss is not None
        ), "AdaptiveLRClipping.clip must be called before AdaptiveLRClipping.update"
        self._mu1 = self._beta1 * self._mu1 + (1 - self._beta1) * self._prev_loss
        self._mu2 = self._beta2 * self._mu2 + (1 - self._beta2) * self._prev_loss**2
