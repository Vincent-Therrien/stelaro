"""K-mer transformation module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: October 2024
    - License: MIT
"""

import stelaro.stelaro as stelaro_rust


def profile(index: str, directory: str, k: int) -> dict:
    """Obtain the K-mer profile of a reference genome set.

    Args:
        index: File path of an index file.
        directory: Directory containing the reference genomes.
        k: K value.
    """
    return stelaro_rust.profile_kmer(index, directory, k)


def spectrum(kmers: dict, ax) -> None:
    """Plot K-mer multiplicity against count.

    Args:
        kmers: K-mer count returned by `stelaro.transform.kmer.profile`.
        ax: Pyplot axis.
    """
    kmers = kmers.copy()
    x = []
    y = []
    counts = set(kmers.values())
    for count in sorted(counts, reverse=True):
        multiplicity = 0
        keys = []
        for kmer, n in kmers.items():
            if n == count:
                multiplicity += 1
                keys.append(kmer)
        if multiplicity:
            x.append(multiplicity)
            y.append(count)
            for key in keys:
                del kmers[key]
    ax.scatter(x, y)


def frequency(kmers: dict, ax) -> None:
    """Plot the count of each K-mer.

    Args:
        kmers: K-mer count returned by `stelaro.transform.kmer.profile`.
        ax: Pyplot axis.
    """
    x = list(range(0, len(kmers)))
    y = sorted(kmers.values())
    ax.scatter(x, y)


def extract(kmers: dict, n: int, m: int = None) -> list[str]:
    """Extract the `n` most frequent K-mers from a K-mer profile that overlap
    by less than `m` characters.

    Args:
        kmers: K-mer profile.
        n: Number of K-mers to fetch.
        m: Number of contiguous characters that must differ to consider two
            K-mers non-overlapping. Not used if `None`.
    """
    counts = []
    for kmer, count in kmers.items():
        counts.append((kmer, count))
    counts = sorted(counts, key=lambda x: x[1])
    selected_kmers = []
    index = 0
    while len(selected_kmers) < n and index < len(kmers):
        index += 1
        candidate, _ = counts[-index]
        if m is None:
            selected_kmers.append(candidate)
        else:
            overlaps = False
            for previous in selected_kmers:
                for i in range(1, m + 1):
                    left_overlap = previous[:-i] == candidate[i:]
                    right_overlap = previous[i:] == candidate[:-i]
                    if left_overlap or right_overlap:
                        overlaps = True
                        break
                if overlaps:
                    break
            if not overlaps:
                selected_kmers.append(candidate)
    return selected_kmers


def compress(sequence: str, scheme: dict) -> list[int]:
    """Compress a sequence according to a K-mer scheme.

    Args:
        sequence: Sequence to compress.
        scheme: Compression scheme formatted as `{kmer: symbol}`.
    """
    keys = list(scheme.keys())
    lengths = set([len(k) for k in keys])
    length_bins = [[] for _ in lengths]
    reversed_lengths = sorted(lengths, reverse=True)
    for i, length in zip(range(len(length_bins)), reversed_lengths):
        for k in scheme:
            if len(k) == length:
                length_bins[i].append(k)
    scratchpad = list(sequence)
    for length_bin in length_bins:
        window_size = len(length_bin[0])
        for i in range(0, len(sequence) - window_size + 1):
            window = scratchpad[i:i + window_size]
            try:
                window = "".join(window)
            except TypeError as _:
                continue
            if window in length_bin:
                scratchpad[i] = scheme[window]
                for j in range(1, window_size):
                    scratchpad[i + j] = None
                i += window_size
    compression = [s for s in scratchpad if s]
    return compression


def decompress(sequence: list[int], scheme: dict) -> str:
    """Restore a compressed sequence into the original sequence."""
    reverse_scheme = {}
    for k, v in scheme.items():
        reverse_scheme[v] = k
    symbols = [reverse_scheme[s] for s in sequence]
    return "".join(symbols)
