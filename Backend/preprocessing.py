import opensmile
import numpy as np
import os

# Initialize OpenSMILE once to avoid overhead
# IS10 feature set (1582 features) matches the training data
try:
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.IS10,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    print("✅ OpenSMILE initialized successfully (IS10)")
except Exception as e:
    print(f"❌ Failed to initialize OpenSMILE: {e}")
    smile = None

def extract_features(file_path):
    """
    Extracts 1582 features using OpenSMILE IS10 config.
    Returns a 1D numpy array.
    """
    if smile is None:
        print("OpenSMILE not available.")
        return None

    try:
        # process_file returns a DataFrame
        df = smile.process_file(file_path)
        
        # Convert to numpy array and flatten
        features = df.to_numpy().flatten()
        
        # ⚠️ CRITICAL FIX: The models were trained with 'id' (col 0) and 'class' (last col) included!
        # Total expected: 1584 (1 + 1582 + 1)
        # We must pad the 1582 OpenSMILE features to match 1584 to avoid crash.
        # We use 0.0 as dummy values for ID and Class.
        
        features_padded = np.zeros(1584)
        features_padded[1:1583] = features # Fill middle with real features
        # features_padded[0] is 0.0 (dummy id)
        # features_padded[1583] is 0.0 (dummy class)
        
        return features_padded
        
    except Exception as e:
        print(f"Error extracting features (OpenSMILE): {e}")
        return None