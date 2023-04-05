import numpy as np


def remove_cosmic_ray_1d(spectrum: np.ndarray, width: int, threshold: float):
    spectrum = spectrum.copy()
    intensity = np.diff(spectrum)
    median_int = np.median(intensity)
    mad_int = np.median([np.abs(intensity - median_int)])
    if mad_int == 0:
        mad_int = 1e-4
    modified_scores = 0.6745 * (intensity - median_int) / mad_int
    spikes = abs(modified_scores) > threshold

    for i in np.arange(len(spikes)):
        if spikes[i]:
            w = np.arange(i - width, i + 1 + width)  # スパイク周りの2 m + 1個のデータを取り出す
            w = w[(0 <= w) & (w < (spectrum.shape[0] - 1))]  # 範囲を超えないようトリミング
            w2 = w[spikes[w] == False]  # スパイクでない値を抽出し，
            if len(w2) > 0:
                spectrum[i] = np.mean(spectrum[w2])  # 平均を計算し補完
    return spectrum


def remove_cosmic_ray(spectrum: np.ndarray, width: int = 3, threshold: float = 7):
    if len(spectrum.shape) == 1:
        return remove_cosmic_ray_1d(spectrum, width, threshold)

    if len(spectrum.shape) == 2:
        data_removed = []
        for spec in spectrum:
            data_removed.append(remove_cosmic_ray_1d(spec, width, threshold))
        return np.array(data_removed)
