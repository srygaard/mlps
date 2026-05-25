"""Repeated instantiation of heavily-documented classes — tests -OO docstring removal."""

import time


class LinearLayer:
    """
    A fully-connected linear layer: y = xW^T + b.

    Parameters
    ----------
    in_features : int
        Number of input features per sample.
    out_features : int
        Number of output features per sample.
    bias : bool, optional
        Whether to add a bias term. Default: True.

    Attributes
    ----------
    weight : list[list[float]]
        Weight matrix of shape (out_features, in_features), initialised to zero.
    bias_vec : list[float] or None
        Bias vector of shape (out_features,), or None when bias=False.

    Notes
    -----
    This is a pure-Python reference implementation used for benchmarking.
    No gradient tracking or GPU dispatch is performed.

    Examples
    --------
    >>> layer = LinearLayer(4, 2)
    >>> layer.forward([1.0, 2.0, 3.0, 4.0])
    [0.0, 0.0]
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Initialise weight and bias tensors to zero.

        Parameters
        ----------
        in_features : int
            Dimensionality of the input vector.
        out_features : int
            Dimensionality of the output vector.
        bias : bool
            If True a zero bias vector is created; otherwise bias_vec is None.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.weight = [[0.0] * in_features for _ in range(out_features)]
        self.bias_vec = [0.0] * out_features if bias else None

    def forward(self, x: list[float]) -> list[float]:
        """
        Apply the linear transformation to input vector x.

        Parameters
        ----------
        x : list[float]
            Input of length in_features.

        Returns
        -------
        list[float]
            Output of length out_features equal to xW^T + b.
        """
        out = [sum(w * xi for w, xi in zip(row, x)) for row in self.weight]
        if self.bias_vec:
            out = [o + b for o, b in zip(out, self.bias_vec)]
        return out


def main():
    start = time.perf_counter()
    for _ in range(20_000):
        layer = LinearLayer(32, 16)
        layer.forward([0.5] * 32)
    elapsed = time.perf_counter() - start
    print(f"{elapsed:.6f}")


if __name__ == "__main__":
    main()
