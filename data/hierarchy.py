import yaml
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from utils.utils import Dotdict

def count_all_nodes(hierarchy):
    total_nodes = 0

    def recursive_count(subtree):
        nonlocal total_nodes
        total_nodes += len(subtree)  # Count the current level keys
        for key in subtree:
            if subtree[key]:
                recursive_count(subtree[key])

    recursive_count(hierarchy)
    return total_nodes


# Function to traverse the hierarchy and get the path for each leaf and intermediate label
def extract_paths(node, code_prefix=""):
    leaf_paths = {}
    intermediate_paths = {}
    code_index = 1
    
    for key, value in node.items():
        # Generate the current code for this level
        current_code = code_prefix + str(code_index)
        
        if isinstance(value, dict) and value:
            # Add intermediate label
            intermediate_paths[key] = current_code
            
            # If it's a leaf node (no further children), assign the current code (no additional '1')
            if not value:  # Check if value is an empty dict (no children)
                leaf_paths[key] = current_code
            else:
                # Traverse deeper for intermediate and leaf nodes
                leafs, intermediates = extract_paths(value, current_code)
                leaf_paths.update(leafs)
                intermediate_paths.update(intermediates)
        else:
            # Add leaf label, but without appending '1' if it's directly a leaf under a higher level node
            leaf_paths[key] = current_code

        code_index += 1

    return leaf_paths, intermediate_paths



def build_hierarchy_mapping(final_hierarchy_strings):
    """
    Build a mapping from each label to its hierarchical ancestors based on the hierarchy strings.
    """
    hierarchy = {}
    
    for label, path in final_hierarchy_strings.items():
        ancestors = []
        # Add all ancestors by iterating through the path
        for i in range(1, len(path) + 1):
            ancestor_path = path[:i]
            ancestors.append(ancestor_path)
        hierarchy[label] = ancestors
    
    return hierarchy

def extend_ys(ys, leaf_paths, intermediate_paths):
    """
    Extend the one-hot encoded labels (ys) by including the intermediate labels based on the hierarchy.
    After extending, verify that the extension was correctly applied.
    """
    # Build the hierarchy of ancestors

    final_hierarchy_strings = {**leaf_paths, **intermediate_paths}
    
    hierarchy = build_hierarchy_mapping(final_hierarchy_strings)
    
    # Initialize the extended label matrix with zeros
    n_samples, num_leaf_labels = ys.shape
    num_classes = len(final_hierarchy_strings)
    ys_extended = np.zeros((n_samples, num_classes), dtype=np.float32)
    
    # Map leaf labels to their indices
    leaf_paths = [(k, v) for k, v in leaf_paths.items()]
    # label_to_index = {label: i for i, (label, _) in enumerate(leaf_labels)}
    path_to_index = {path: i for i, path in enumerate(final_hierarchy_strings.values())}

    # Step 1: Keep the leaf labels unchanged
    ys_extended[:, :num_leaf_labels] = ys  # Copy the leaf labels

    differences = np.where(ys_extended[:, :num_leaf_labels] != ys)
    print(f"Differences found at: {differences}")
    
    # Step 2: Extend the ys matrix by adding the ancestors
    for i in range(n_samples):
        for leaf_idx in range(num_leaf_labels):
            if ys[i, leaf_idx] > 0.0:  # Consider any positive value as active
                # Get the leaf label corresponding to this index
                leaf_label = leaf_paths[leaf_idx][0]
                # Get the ancestors from the hierarchy
                ancestors = hierarchy[leaf_label]
                # Activate all ancestors in the extended matrix using path_to_index
                for ancestor in ancestors:
                    if ancestor in path_to_index:
                        ancestor_index = path_to_index[ancestor]
                        ys_extended[i, ancestor_index] = 1.0  # Ensure ancestors are activated as 1


    return ys_extended




def process_hierarchy_config(yaml_path):
    """
    Processes the hierarchy configuration from a YAML file and generates the final hierarchy strings
    and grouped levels based on the leaf labels.

    Parameters:
    yaml_path (str): Path to the YAML configuration file.

    Returns:
    tuple: A tuple containing the length groups and hierarchy data frame.
    """
    # Load YAML data
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    # Access data
    hierarchy = config['hierarchy']
    leaf_labels = config['leaf_labels']
    num_classes = config['num_classes']
    dataset_name = config['name']

    leaf_paths, intermediate_paths = extract_paths(hierarchy)

    label_to_predecessors = encode_classes_and_ancestors(
        leaf_paths, intermediate_paths
    )

    final_hierarchy_strings = {**leaf_paths, **intermediate_paths}
    
    output = {
        "dataset_name": dataset_name,
        "hierarchy": hierarchy,
        "leaf_labels": leaf_labels,
        "leaf_paths": leaf_paths,
        "intermediate_paths": intermediate_paths,
        "final_hierarchy_strings": final_hierarchy_strings,
        "label_names": list(leaf_paths.values()),
        "num_classes": num_classes,
        "num_classes_extended": len(final_hierarchy_strings),
        "label_to_predecessors": label_to_predecessors
    }

    output = Dotdict(output)

    return output


def encode_classes_and_ancestors(leaf_paths, intermediate_paths):
    """
    Encodes leaf classes and their ancestors into integer representations.

    Args:
        leaf_paths (dict): A dictionary mapping class names to their string codes.
        intermediate_paths (dict): A dictionary mapping ancestor names to their string codes.

    Returns:
        dict: A dictionary where keys are encoded class indices (integers) and values are lists of 
        encoded ancestor indices (integers).
    """
    class_encoding = {class_code: idx for idx, class_code in enumerate(leaf_paths.values())}

    ancestor_encoding_start = len(leaf_paths)
    ancestor_encoding = {ancestor_code: idx + ancestor_encoding_start 
                         for idx, ancestor_code in enumerate(intermediate_paths.values())}

    def get_ancestor_codes(class_code):
        return [ancestor_encoding[ancestor_code] 
                for ancestor_code in intermediate_paths.values() 
                if class_code.startswith(ancestor_code)]

    return {class_encoding[class_code]: get_ancestor_codes(class_code) 
            for class_code in leaf_paths.values()}


def create_edge_index(hierarchy=None, labeled_data=None, threshold=0.5):
    if hierarchy is None and labeled_data is None:
        raise ValueError("Either hierarchy or labeled_data must be provided.")
    
    if hierarchy is not None:
        edge_index = []
        for child, ancestors in hierarchy.items():
            for i in range(len(ancestors) - 1):
                edge_index.append([ancestors[i], ancestors[i + 1]])
            edge_index.append([ancestors[-1], child])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        labeled_data = np.array(labeled_data['one_hot'].tolist())
        # Filter out labels that are not present in the labeled data
        present_labels = np.sum(labeled_data, axis=0) > 0
        filtered_data = labeled_data[:, present_labels]

        # Compute the correlation matrix using numpy's corrcoef
        correlation_matrix = np.corrcoef(filtered_data, rowvar=False)

        # Create edge list based on the threshold
        edge_index = []
        num_labels = correlation_matrix.shape[0]
        for i in range(num_labels):
            for j in range(i+1, num_labels):
                if abs(correlation_matrix[i, j]) >= threshold:
                    edge_index.append([i, j])
                    edge_index.append([j, i])  # Include both directions for undirected graph

        # Map indices back to original labels
        original_labels = np.where(present_labels)[0]
        mapped_edge_index = [[original_labels[edge[0]], original_labels[edge[1]]] for edge in edge_index]

        # Convert edge list to tensor
        edge_index = torch.tensor(mapped_edge_index, dtype=torch.long).t().contiguous()

    return edge_index


############################## Methods for hierarchy verification ########################################

def edge_case_tests():
    # Test 1: Empty input (No hierarchy and no ys)
    print("Test 1: Empty Input")
    try:
        empty_hierarchy = {}
        empty_ys = np.zeros((0, 0))
        leaf_paths, intermediate_paths = extract_paths(multi_class_hierarchy)
        ys_extended = extend_ys(empty_ys, leaf_paths, intermediate_paths)
        assert ys_extended is not None and ys_extended.shape == (0, 0), "Test 1 Failed"
        print("Test 1 Passed")
    except Exception as e:
        print(f"Test 1 Failed: {e}")
    
    # Test 2: Single leaf node with no ancestors
    print("\nTest 2: Single Leaf Node")
    try:
        single_hierarchy = {"leaf1": {}}
        ys_single = np.array([[1]])
        leaf_paths, intermediate_paths = extract_paths(multi_class_hierarchy)
        ys_extended = extend_ys(ys_single, leaf_paths, intermediate_paths)
        assert np.array_equal(ys_extended, np.array([[1]])), "Test 2 Failed"
        print("Test 2 Passed")
    except Exception as e:
        print(f"Test 2 Failed: {e}")
    
    # Test 3: Deep hierarchy with multiple ancestor levels
    print("\nTest 3: Deep Hierarchy")
    try:
        deep_hierarchy = {
            "level1": {
                "level2": {
                    "level3": {
                        "leaf": {}
                    }
                }
            }
        }
        ys_deep = np.array([[1]])
        leaf_paths, intermediate_paths = extract_paths(multi_class_hierarchy)
        ys_extended = extend_ys(ys_deep, leaf_paths, intermediate_paths)
        expected_output = np.array([[1, 1, 1, 1]])  # 'leaf', 'level3', 'level2', 'level1'
        assert np.array_equal(ys_extended, expected_output), "Test 3 Failed"
        print("Test 3 Passed")
    except Exception as e:
        print(f"Test 3 Failed: {e}")
    
    # Test 4: Multiple root nodes
    print("\nTest 4: Multiple Root Nodes")
    try:
        multi_root_hierarchy = {
            "root1": {
                "leaf1": {}
            },
            "root2": {
                "leaf2": {}
            }
        }
        ys_multi_root = np.array([[1, 0], [0, 1]])
        leaf_paths, intermediate_paths = extract_paths(multi_class_hierarchy)
        ys_extended = extend_ys(ys_multi_root, leaf_paths, intermediate_paths)
        expected_output = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])  # root1, leaf1 and root2, leaf2
        assert np.array_equal(ys_extended, expected_output), "Test 4 Failed"
        print("Test 4 Passed")
    except Exception as e:
        print(f"Test 4 Failed: {e}")
    
    # Test 5: Multiple leaf labels pointing to the same intermediate label
    print("\nTest 5: Multiple Leaf Labels with Common Intermediate")
    try:
        common_ancestor_hierarchy = {
            "root": {
                "intermediate": {
                    "leaf1": {},
                    "leaf2": {}
                }
            }
        }
        ys_common_ancestor = np.array([[1, 1]])  # leaf1 and leaf2
        leaf_paths, intermediate_paths = extract_paths(multi_class_hierarchy)
        ys_extended = extend_ys(ys_common_ancestor, leaf_paths, intermediate_paths)
        expected_output = np.array([[1, 1, 1, 1]])  # leaf1, leaf2, intermediate, root
        assert np.array_equal(ys_extended, expected_output), "Test 5 Failed"
        print("Test 5 Passed")
    except Exception as e:
        print(f"Test 5 Failed: {e}")
    
    # Test 6: Non-binary ys values
    print("\nTest 6: Non-binary ys values (Multi-class)")
    try:
        multi_class_hierarchy = {
            "root": {
                "leaf1": {},
                "leaf2": {}
            }
        }
        ys_multi_class = np.array([[1, 2]])  # Multi-class labels
        leaf_paths, intermediate_paths = extract_paths(multi_class_hierarchy)
        ys_extended = extend_ys(ys_multi_class, leaf_paths, intermediate_paths)
        expected_output = np.array([[1, 1, 1, 1]])  # We expect all related labels to be 1
        assert np.array_equal(ys_extended, expected_output), "Test 6 Failed"
        print("Test 6 Passed")
    except Exception as e:
        print(f"Test 6 Failed: {e}")


def verify_hierarchy_extension(ys_extended, ys, leaf_paths, intermediate_paths):
    """
    Verify that the one-hot encoded label extension is correctly done based on the hierarchy.
    """
    # Step 1: Build the hierarchy of ancestors
    final_hierarchy_strings = {**leaf_paths, **intermediate_paths}
    hierarchy = build_hierarchy_mapping(final_hierarchy_strings)
    
    # Step 2: Extract the number of leaf labels and total classes
    n_samples, num_classes = ys_extended.shape
    num_leaf_labels = ys.shape[1]
    
    # Step 3: Map leaf labels and paths to their indices
    label_to_index = {label: i for i, (label, _) in enumerate(leaf_paths)}
    path_to_index = {path: i for i, path in enumerate(final_hierarchy_strings.values())}

    # Step 4: Check if the original ys is preserved
    if not np.array_equal(ys_extended[:, :num_leaf_labels], ys):
        print("Leaf labels were modified during extension.")
        return False

    # Step 5: Validate each sample
    for i in range(n_samples):
        for leaf_idx in range(num_leaf_labels):
            if ys[i, leaf_idx] > 0.0:  # If the leaf label is active in the original ys
                # Get the leaf label and its ancestors
                leaf_label = leaf_paths[leaf_idx][0]
                ancestors = hierarchy[leaf_label]
                
                # Verify that all ancestors are active in ys_extended
                for ancestor in ancestors:
                    if ancestor in path_to_index:
                        ancestor_index = path_to_index[ancestor]
                        if ys_extended[i, ancestor_index] != 1.0:
                            print(f"Error: Ancestor '{ancestor}' not active for leaf label '{leaf_label}' in sample {i}.")
                            return False

    print("Verification successful: All ancestors are correctly activated for active leaf labels.")
    return True
