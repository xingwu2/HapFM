import numpy as np

def geweke(x, first=.1, last=.5, intervals=20):
    R"""Return z-scores for convergence diagnostics.

    Compare the mean of the first % of series with the mean of the last % of
    series. x is divided into a number of segments for which this difference is
    computed. If the series is converged, this score should oscillate between
    -1 and 1.

    Parameters
    ----------
    x : array-like
      The trace of some stochastic parameter.
    first : float
      The fraction of series at the beginning of the trace.
    last : float
      The fraction of series at the end to be compared with the section
      at the beginning.
    intervals : int
      The number of segments.

    Returns
    -------
    scores : list [[]]
      Return a list of [i, score], where i is the starting index for each
      interval and score the Geweke score on the interval.

    Notes
    -----

    The Geweke score on some series x is computed by:

      .. math:: \frac{E[x_s] - E[x_e]}{\sqrt{V[x_s] + V[x_e]}}

    where :math:`E` stands for the mean, :math:`V` the variance,
    :math:`x_s` a section at the start of the series and
    :math:`x_e` a section at the end of the series.

    References
    ----------
    Geweke (1992)
    """

    if np.ndim(x) > 1:
        return [geweke(y, first, last, intervals) for y in np.transpose(x)]

    # Filter out invalid intervals
    for interval in (first, last):
        if interval <= 0 or interval >= 1:
            raise ValueError(
                "Invalid intervals for Geweke convergence analysis",
                (first,
                 last))
    if first + last >= 1:
        raise ValueError(
            "Invalid intervals for Geweke convergence analysis",
            (first,
             last))

    # Initialize list of z-scores
    zscores = []

    # Last index value
    end = len(x) - 1

    # Start intervals going up to the <last>% of the chain
    last_start_idx = (1 - last) * end

    # Calculate starting indices
    start_indices = np.arange(0, int(last_start_idx), step=int(
        (last_start_idx) / (intervals - 1)))

    # Loop over start indices
    for start in start_indices:
        # Calculate slices
        first_slice = x[start: start + int(first * (end - start))]
        last_slice = x[int(end - last * (end - start)):]

        z = first_slice.mean() - last_slice.mean()
        z /= np.sqrt(first_slice.var() + last_slice.var())

        zscores.append([start, z])

    if intervals is None:
        return np.array(zscores[0])
    else:
        return np.array(zscores)