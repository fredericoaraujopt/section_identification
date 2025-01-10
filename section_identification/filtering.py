import numpy as np
from sklearn.cluster import DBSCAN

def filtering(sorted_masks, eps_values, min_samples_values):
    """
    Filters the identified masks by clustering their areas using DBSCAN.

    This function systematically runs DBSCAN over a grid of parameters (eps_values, min_samples_values),
    identifies the largest cluster for each combination, and tracks which largest cluster (as a set of
    mask indices) occurs most frequently.

    Parameters
    ----------
    sorted_masks : list of dict
        Each dict should have at least the key 'area', e.g., mask['area'].
    eps_values : list or array-like
        Values of eps to try in DBSCAN.
    min_samples_values : list or array-like
        Values of min_samples to try in DBSCAN.

    Returns
    -------
    chosen_filtered_masks : list of dict
        The masks corresponding to the most commonly occurring largest cluster.
    chosen_params : tuple
        The (eps, min_samples) pair that produced this largest cluster.
    """

    # Prepare mask areas for clustering
    mask_areas = np.array([mask['area'] for mask in sorted_masks]).reshape(-1, 1)

    # Dictionary to track frequency of each "largest cluster" set
    cluster_frequency = {}

    # Iterate over all eps and min_samples values
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(mask_areas)

            # Identify largest cluster label (exclude noise = -1)
            unique_clusters = np.unique(clusters)
            largest_cluster_label = None
            largest_cluster_size = -1
            for c in unique_clusters:
                if c == -1:
                    continue
                size_c = np.sum(clusters == c)
                if size_c > largest_cluster_size:
                    largest_cluster_size = size_c
                    largest_cluster_label = c

            # Handle cases with no valid clusters
            if largest_cluster_label is None:
                continue

            # Get indices of the largest cluster
            indices_in_largest_cluster = {
                idx for idx, clbl in enumerate(clusters) if clbl == largest_cluster_label
            }
            cluster_key = frozenset(indices_in_largest_cluster)

            # Update frequency dictionary
            if cluster_key not in cluster_frequency:
                cluster_frequency[cluster_key] = {
                    'count': 0,
                    'params': []
                }
            cluster_frequency[cluster_key]['count'] += 1
            cluster_frequency[cluster_key]['params'].append((eps, min_samples))

    if not cluster_frequency:
        print("No non-noise clusters found for any parameter combination.")
        return [], (None, None)

    # Find the most frequent largest cluster
    max_count = 0
    most_common_cluster_key = None
    for c_key, info in cluster_frequency.items():
        if info['count'] > max_count:
            max_count = info['count']
            most_common_cluster_key = c_key

    most_common_cluster_info = cluster_frequency[most_common_cluster_key]
    chosen_params = most_common_cluster_info['params'][0]

    # Build the final list of filtered masks
    most_common_indices = list(most_common_cluster_key)
    chosen_filtered_masks = [sorted_masks[i] for i in most_common_indices]

    # Print cluster size and chosen parameters
    total_masks = len(sorted_masks)
    chosen_count = len(chosen_filtered_masks)
    print(
        f"Most common largest cluster size: {chosen_count} "
        f"out of {total_masks} total masks.\n"
        f"Chosen parameters: eps={chosen_params[0]}, min_samples={chosen_params[1]}"
    )

    return chosen_filtered_masks, chosen_params