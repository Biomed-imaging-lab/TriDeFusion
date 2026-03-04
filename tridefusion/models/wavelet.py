import cupy as cp
from typing import Tuple, Optional
from cuml.pipeline import Pipeline
from cuml.preprocessing import StandardScaler


class CuMLWaveletFilter:
    def __init__(
        self,
        wavelet: str = "haar",
        level: int = 1,
        threshold: Optional[float] = None,
        mode: str = "soft",
    ):
        self.wavelet = wavelet
        self.level = level
        self.threshold_value = threshold
        self.mode = mode
        self._init_filters()

    def _init_filters(self):
        if self.wavelet == "haar":
            h = cp.array([1 / cp.sqrt(2), 1 / cp.sqrt(2)])
            g = cp.array([1 / cp.sqrt(2), -1 / cp.sqrt(2)])
        else:
            raise NotImplementedError("Currently only Haar implemented")
        self.low_pass = h
        self.high_pass = g

    def _conv1d(self, x: cp.ndarray, filt: cp.ndarray):
        return cp.apply_along_axis(
            lambda m: cp.convolve(m, filt, mode="same"), axis=-1, arr=x
        )

    def _dwt2(self, x: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
        low = self._conv1d(x, self.low_pass)
        high = self._conv1d(x, self.high_pass)
        low = low[..., ::2]
        high = high[..., ::2]
        low_low = self._conv1d(low.transpose(0, 2, 1), self.low_pass).transpose(0, 2, 1)
        low_high = self._conv1d(low.transpose(0, 2, 1), self.high_pass).transpose(0, 2, 1)
        high_low = self._conv1d(high.transpose(0, 2, 1), self.low_pass).transpose(0, 2, 1)
        high_high = self._conv1d(high.transpose(0, 2, 1), self.high_pass).transpose(0, 2, 1)
        return (
            low_low[..., ::2, :],
            low_high[..., ::2, :],
            high_low[..., ::2, :],
            high_high[..., ::2, :],
        )

    def forward_dwt(self, x: cp.ndarray):
        coeffs = []
        current = x
        for _ in range(self.level):
            LL, LH, HL, HH = self._dwt2(current)
            coeffs.append((LH, HL, HH))
            current = LL
        coeffs.insert(0, current)  # final approximation
        return coeffs

    def threshold(self, coeffs):
        if self.threshold_value is None:
            return coeffs

        def soft(x):
            return cp.sign(x) * cp.maximum(cp.abs(x) - self.threshold_value, 0)

        def hard(x):
            return x * (cp.abs(x) > self.threshold_value)
        thresh_func = soft if self.mode == "soft" else hard
        new_coeffs = [coeffs[0]]
        for (LH, HL, HH) in coeffs[1:]:
            new_coeffs.append(
                (
                    thresh_func(LH),
                    thresh_func(HL),
                    thresh_func(HH),
                )
            )
        return new_coeffs

    def inverse_dwt(self, coeffs):
        current = coeffs[0]
        for level in reversed(coeffs[1:]):
            LH, HL, HH = level
            up_LL = cp.repeat(current, 2, axis=-2)
            up_LH = cp.repeat(LH, 2, axis=-2)
            up_HL = cp.repeat(HL, 2, axis=-2)
            up_HH = cp.repeat(HH, 2, axis=-2)
            current = up_LL + up_LH + up_HL + up_HH
        return current

    def fit_transform(self, x: cp.ndarray):
        coeffs = self.forward_dwt(x)
        coeffs = self.threshold(coeffs)
        return self.inverse_dwt(coeffs)

    def transform(self, x: cp.ndarray):
        return self.fit_transform(x)
    

if __name__ == "__main__":
    wavelet = CuMLWaveletFilter(level=1, threshold=0.05)
    pipeline = Pipeline([
        ("wavelet", wavelet),
        ("scaler", StandardScaler())
    ])