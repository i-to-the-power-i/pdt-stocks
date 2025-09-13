
import numpy as np
from collections import Counter
from tqdm import tqdm
from graphviz import Digraph
import matplotlib.pyplot as plt
from matplotlib import image as mpimg

# ------------------------------------------------------------------------
# 1. ETC Calculation
# ------------------------------------------------------------------------

def calculate_etc(seq, deb = False):
    """
    Calculate the ETC (Effort To Compress) value for a given sequence of numbers.

    Args:
    seq (list or string): Input sequence for which ETC needs to be calculated.

    Returns:
    int: The value of ETC (number of iterations to compress the sequence).
    """
    itrs = 0
    seq = list(map(int, seq))  # Convert elements to integers
    max_symbol = max(seq)  # Keep track of the maximum symbol used
    while len(set(seq)) > 1:  # Loop until the sequence is uniform
        # Count pairs of consecutive elements
        pair_count = Counter((seq[i], seq[i + 1]) for i in range(len(seq) - 1))

        # Find the most frequent pair
        highest_freq_pair = max(pair_count, key=pair_count.get)

        # Create a new symbol that is larger than the current max value in the sequence
        max_symbol += 1
        new_symbol = max_symbol

        # Replace the most frequent pairs with the new symbol
        temp_seq = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and (seq[i], seq[i + 1]) == highest_freq_pair:
                temp_seq.append(new_symbol)
                i += 2  # Skip next element as it's part of the replaced pair
            else:
                temp_seq.append(seq[i])
                i += 1

        # Update the sequence and iteration count
        seq = temp_seq
        if deb:
            print(seq)
        itrs += 1

    return itrs
# ------------------------------------------------------------------------
# 2. ETC Gain
# ------------------------------------------------------------------------

def etc_gain(data, labels, feature_index):
    """
    Calculate the ETC information gain for splitting on a given feature index.
    
    data: 2D numpy array of shape (n_samples, n_features)
    labels: 1D numpy array of shape (n_samples,)
    feature_index: which column in data to split on
    """
    # Convert labels to int array (if not already)
    labels = np.array(labels, dtype=int)

    # Calculate total ETC for the entire label set
    total_etc = calculate_etc(labels)

    # Get all values of the specified feature
    feature_values = data[:, feature_index]

    # Sort unique feature values
    unique_vals = np.unique(feature_values)
    
    # If there's only one unique value, we can't split; return no gain
    if len(unique_vals) == 1:
        return -float('inf'), None

    # Candidate thresholds = midpoints between consecutive unique values
    thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

    best_gain = -float('inf')
    best_threshold = None

    # For each threshold, split labels into two groups: <= threshold, > threshold
    for t in thresholds:
        left_mask = feature_values <= t
        right_mask = ~left_mask

        left_labels = labels[left_mask]
        right_labels = labels[right_mask]

        left_etc = calculate_etc(left_labels) if len(left_labels) else 0
        right_etc = calculate_etc(right_labels) if len(right_labels) else 0

        # Weighted ETC of the split
        weighted_etc = (len(left_labels)/len(labels))*left_etc \
                     + (len(right_labels)/len(labels))*right_etc

        # Gain is how much ETC is reduced
        gain = total_etc - weighted_etc

        if gain > best_gain:
            best_gain = gain
            best_threshold = t

    return best_gain, best_threshold

# ------------------------------------------------------------------------
# 3. Find Best Feature
# ------------------------------------------------------------------------

def find_best_feature(data, labels):
    """
    Find the best (feature_index, threshold) that yields maximum ETC gain.
    """
    data = np.array(data)
    labels = np.array(labels)

    num_features = data.shape[1]
    best_feature_index = None
    best_threshold = None
    best_gain = -float('inf')

    for f_idx in range(num_features):
        gain, threshold = etc_gain(data, labels, f_idx)
        if gain > best_gain:
            best_gain = gain
            best_feature_index = f_idx
            best_threshold = threshold

    return best_feature_index, best_threshold

# ------------------------------------------------------------------------
# 4. Build the PDT (with optional Progress Bar)
# ------------------------------------------------------------------------

def build_pdt(
    data,
    labels,
    depth=0,
    max_depth=10,
    pbar=None,
    total_nodes=None
):
    """
    Recursively build the Permutation Decision Tree using ETC-based splits.
    Includes optional integration with a tqdm progress bar.

    Args:
        data (array-like): Feature matrix.
        labels (array-like): Labels.
        depth (int): Current depth of the tree.
        max_depth (int): Maximum depth of the tree.
        pbar (tqdm.tqdm or None): Progress bar object to update (optional).
        total_nodes (int or None): An estimate of total nodes to build (optional).
    """
    data = np.array(data)
    labels = np.array(labels, dtype=int)

    # Base cases
    if len(data) == 0:
        # Update progress bar
        if pbar is not None:
            pbar.update(1)
        return None
    if np.unique(labels).size == 1:
        if pbar is not None:
            pbar.update(1)
        return labels[0]
    if depth >= max_depth:
        if pbar is not None:
            pbar.update(1)
        # Return most frequent label
        label_counts = Counter(labels)
        return label_counts.most_common(1)[0][0]

    # Find best split
    best_feature, best_threshold = find_best_feature(data, labels)

    # If no threshold is found or no improvement, return majority label
    if best_threshold is None:
        if pbar is not None:
            pbar.update(1)
        label_counts = Counter(labels)
        return label_counts.most_common(1)[0][0]

    # Create current node
    tree = {
        'feature_index': best_feature,
        'threshold': best_threshold,
        'children': {}
    }

    # Split data
    left_mask = data[:, best_feature] <= best_threshold
    right_mask = ~left_mask

    left_data, left_labels = data[left_mask], labels[left_mask]
    right_data, right_labels = data[right_mask], labels[right_mask]

    # Recursively build subtrees
    tree['children']['left'] = build_pdt(
        left_data, left_labels,
        depth=depth + 1,
        max_depth=max_depth,
        pbar=pbar,
        total_nodes=total_nodes
    )
    tree['children']['right'] = build_pdt(
        right_data, right_labels,
        depth=depth + 1,
        max_depth=max_depth,
        pbar=pbar,
        total_nodes=total_nodes
    )

    # Update progress bar once we've created this node
    if pbar is not None:
        pbar.update(1)

    return tree


def train_with_progress(data, labels, max_depth=10):
    """
    Initialize a tqdm progress bar and build the PDT with an estimated 
    upper bound on the number of nodes (worst-case full binary tree).
    """
    # Estimating total nodes in a worst-case full binary tree
    estimated_nodes = 2 ** (max_depth + 1) - 1

    with tqdm(total=estimated_nodes, desc="Building PDT") as pbar:
        tree = build_pdt(
            data,
            labels,
            depth=0,
            max_depth=max_depth,
            pbar=pbar,
            total_nodes=estimated_nodes
        )
    return tree

# ------------------------------------------------------------------------
# 5. Prediction
# ------------------------------------------------------------------------

def predict(tree, x):
    """
    Predict label for a single data point x using the built tree.
    """
    # If leaf node
    if not isinstance(tree, dict):
        return tree

    feature_index = tree['feature_index']
    threshold = tree['threshold']

    if x[feature_index] <= threshold:
        return predict(tree['children']['left'], x)
    else:
        return predict(tree['children']['right'], x)

def prediction(data, tree):
    """
    Predict labels for the entire dataset using the built tree.
    """
    data = np.array(data)
    return [predict(tree, row) for row in data]

# ------------------------------------------------------------------------
# 6. Visualization
# ------------------------------------------------------------------------

def generate_tree_visuals(tree, name, graph=None):
    """
    Generate a visual representation of the decision tree and save it to a file.
    """
    def visualize_tree(node, node_id=0, graph_obj=None):
        if graph_obj is None:
            graph_obj = Digraph()

        # If it's an internal node
        if isinstance(node, dict) and 'feature_index' in node and 'threshold' in node:
            label = f"f{node['feature_index']} <= {round(node['threshold'], 3)}"
            graph_obj.node(str(node_id), label)

            # Left Child
            left_child = node['children'].get('left', None)
            left_id = node_id * 2 + 1
            if isinstance(left_child, dict):
                graph_obj.edge(str(node_id), str(left_id), label="True")
                visualize_tree(left_child, left_id, graph_obj)
            else:
                # Leaf
                graph_obj.node(str(left_id), f"{left_child}")
                graph_obj.edge(str(node_id), str(left_id), label="True")

            # Right Child
            right_child = node['children'].get('right', None)
            right_id = node_id * 2 + 2
            if isinstance(right_child, dict):
                graph_obj.edge(str(node_id), str(right_id), label="False")
                visualize_tree(right_child, right_id, graph_obj)
            else:
                # Leaf
                graph_obj.node(str(right_id), f"{right_child}")
                graph_obj.edge(str(node_id), str(right_id), label="False")

        else:
            # Leaf node
            graph_obj.node(str(node_id), f"{node}")

        return graph_obj

    # Build and save the Graphviz object
    graph = visualize_tree(tree, 0, graph)
    graph.render(name, format='png', cleanup=True)

def plot(tree, name="PDT_Visualization"):
    """
    Generate and display the visual representation of the decision tree.
    """
    generate_tree_visuals(tree, name)
    img = mpimg.imread(f'{name}.png')
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis('off')
    plt.show()