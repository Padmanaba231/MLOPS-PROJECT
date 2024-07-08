"""Transform module
"""

import tensorflow as tf
import tensorflow_transform as tft




# Define the numerical features
NUMERICAL_FEATURES = [
    "Age",
    "Gender",
    "BMI",
    "Smoking",
    "GeneticRisk",
    "PhysicalActivity",
    "AlcoholIntake",
    "CancerHistory" 
]

# Define the label key
LABEL_KEY = "Diagnosis"

def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"

def convert_num_to_one_hot(label_tensor, num_labels=2):
    """
    Convert a label (0 or 1) into a one-hot vector
    Args:
        int: label_tensor (0 or 1)
    Returns:
        label tensor
    """
    one_hot_tensor = tf.one_hot(label_tensor, num_labels)
    return tf.reshape(one_hot_tensor, [-1, num_labels])

# def replace_nan(tensor):
#     """Replace nan value with zero number

#     Args:
#         tensor (list): list data with na data that want to replace

#     Returns:
#         list with replaced nan value
#     """
#     tensor = tf.cast(tensor, tf.float64)
#     return tf.where(
#         tf.math.is_nan(tensor),
#         tft.mean(tensor),
#         tensor
#     )


def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features
    
    Args:
        inputs: map from feature keys to raw features.
    
    Return:
        outputs: map from feature keys to transformed features.    
    """
    
    outputs = {}

    # for keys, values in CATEGORICAL_FEATURES.items():
    #     int_value = tft.compute_and_apply_vocabulary(
    #         inputs[keys], top_k=values+1)
    #     outputs[transformed_name(keys)] = convert_num_to_one_hot(
    #         int_value, num_labels=values+1)

    for feature in NUMERICAL_FEATURES:
        # inputs[feature] = replace_nan(inputs[feature])
        outputs[transformed_name(feature)] = tft.scale_to_0_1(inputs[feature])
    
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)
    
    return outputs
