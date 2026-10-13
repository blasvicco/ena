"""Features of a chirp's response: the raw trajectory, or wavelet bands (PyWavelets) with one energy or all coefficients each."""

# Method: wavelet time-frequency features of a structure's response to a known excitation, as in structural
# damage detection and machine fault diagnosis. References: Kijewski & Kareem, "Wavelet Transforms for System
# Identification in Civil Engineering", Computer-Aided Civil and Infrastructure Engineering 18(5) (2003); Peng &
# Chu, "Application of the wavelet transform in machine condition monitoring and fault diagnostics: a review
# with bibliography", Mechanical Systems and Signal Processing 18(2) (2004). Transforms: PyWavelets (Lee et al.,
# JOSS 2019).

# General imports
import math

# Libs imports
import numpy as np
import pywt

TRANSFORMS = ("dwt", "swt", "packet", "cwt")
FEATURES = ("energy", "coefficients")


def _bands(signal: np.ndarray, wavelet: str, transform: str, level, scales) -> list:
	"""One channel's bands: dwt (dyadic, non-overlapping), swt (undecimated: overlapping windows), packet, or cwt."""
	continuous = wavelet in pywt.wavelist(kind="continuous")
	if (transform == "cwt") != continuous:
		raise ValueError(f"Wavelet {wavelet!r} needs {'the cwt' if continuous else 'a discrete transform (dwt, swt, packet)'}")
	if transform == "cwt":
		scales = scales if scales is not None else [2**power for power in range(1, int(math.log2(len(signal))))]
		return list(pywt.cwt(signal, scales, wavelet)[0])
	if transform == "dwt":
		return pywt.wavedec(signal, wavelet, mode="periodization", level=level)
	if transform == "swt":
		return pywt.swt(signal, wavelet, level=level or pywt.swt_max_level(len(signal)), trim_approx=True, norm=True)
	maxlevel = level or pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
	packet = pywt.WaveletPacket(signal, wavelet, mode="periodization", maxlevel=maxlevel)
	return [node.data for node in packet.get_level(maxlevel, order="freq")]


# pylint: disable=too-many-arguments
def features(
	response: np.ndarray, *, kind: str = "wavelet", wavelet: str = "haar", transform: str = "dwt", level=None, feature: str = "energy", scales=None
):
	"""Feature vector of a (steps, channels) response: raw, or per channel and band an energy (log) or its coefficients."""
	response = np.asarray(response, dtype=np.float64)
	if kind == "raw":
		return response.ravel()
	if transform not in TRANSFORMS:
		raise ValueError(f"Unknown wavelet transform: {transform!r}; expected one of {TRANSFORMS}")
	if feature not in FEATURES:
		raise ValueError(f"Unknown wavelet feature: {feature!r}; expected one of {FEATURES}")
	values = []
	for channel in response.T:
		bands = _bands(channel, wavelet, transform, level, scales)
		if feature == "energy":
			values.extend(math.log1p(float(np.sum(np.abs(band) ** 2))) for band in bands)
		else:
			values.extend(np.concatenate([np.real(band) for band in bands]))
	return np.asarray(values)
