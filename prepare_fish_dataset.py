import os
import torch
import numpy as np
import pickle
from pathlib import Path
import random
from sklearn.model_selection import train_test_split

def prepare_fish_dataset():
    """
    Scans the fish dataset directory, splits the data, and saves it into
    the format expected by the main training script.
    """
    # --- Configuration ---
    SEED = 42
    DATA_DIR = Path("data/Fish/Fish_2025/Images")
    ATTRIBUTE_PKL_PATH = Path("attribute/w2v/Fish_attribute.pkl")
    OUTPUT_DIR = Path("info-files")
    OUTPUT_FILENAME = "x-Fish-data-image.pth"
    
    # Ratio for unseen classes
    UNSEEN_RATIO = 0.25 
    # Ratio for test set within seen classes
    TEST_SEEN_RATIO = 0.2

    random.seed(SEED)
    np.random.seed(SEED)
    
    print("--- Starting Fish Dataset Preparation ---")

    # 1. Get all species (class) names
    print(f"Scanning for species in: {DATA_DIR}")
    species_names = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
    num_species = len(species_names)
    print(f"Found {num_species} unique species.")

    if num_species == 0:
        print("Error: No species directories found. Please organize the images first.")
        return

    # 2. Create class mapping and split into seen/unseen
    species_to_id = {name: i for i, name in enumerate(species_names)}
    all_class_ids = list(range(num_species))
    
    unseen_class_ids = sorted(random.sample(all_class_ids, int(num_species * UNSEEN_RATIO)))
    seen_class_ids = sorted(list(set(all_class_ids) - set(unseen_class_ids)))

    print(f"Split: {len(seen_class_ids)} seen classes, {len(unseen_class_ids)} unseen classes.")

    # 3. Load attribute vectors
    print(f"Loading attribute vectors from: {ATTRIBUTE_PKL_PATH}")
    if not ATTRIBUTE_PKL_PATH.exists():
        print(f"Error: Attribute file not found at {ATTRIBUTE_PKL_PATH}")
        print("Please run 'python tools/extract_attribute_w2v_Fish.py' first.")
        return
        
    with open(ATTRIBUTE_PKL_PATH, 'rb') as f:
        attributes = pickle.load(f)
    attributes = torch.from_numpy(attributes).float()
    print(f"Loaded attributes with shape: {attributes.shape}")

    # 4. Process each species to create file and label lists
    print("Processing image files and creating splits...")
    all_files = []
    all_labels = []
    
    for species_name, species_id in species_to_id.items():
        species_dir = DATA_DIR / species_name
        for img_path in species_dir.glob('*.*'):
            all_files.append(str(img_path))
            all_labels.append(species_id)
            
    all_labels = np.array(all_labels)

    # --- Create splits ---
    # Unseen test set
    unseen_indices = np.where(np.isin(all_labels, unseen_class_ids))[0]
    test_unseen_files = [all_files[i] for i in unseen_indices]
    test_unseen_labels = all_labels[unseen_indices]

    # Seen set (to be split into train and test)
    seen_indices = np.where(np.isin(all_labels, seen_class_ids))[0]
    
    # Use sklearn's train_test_split to create a stratified split for the seen classes
    trainval_indices, test_seen_indices = train_test_split(
        seen_indices,
        test_size=TEST_SEEN_RATIO,
        random_state=SEED,
        stratify=all_labels[seen_indices] # Ensure proportion of classes is the same in train/test
    )
    
    trainval_files = [all_files[i] for i in trainval_indices]
    trainval_labels = all_labels[trainval_indices]
    
    test_seen_files = [all_files[i] for i in test_seen_indices]
    test_seen_labels = all_labels[test_seen_indices]

    print(f"Train/Val (seen) samples: {len(trainval_files)}")
    print(f"Test (seen) samples: {len(test_seen_files)}")
    print(f"Test (unseen) samples: {len(test_unseen_files)}")

    # 5. Assemble the final data dictionary
    dataset_dict = {
        'allclasses': all_class_ids,
        'train_classes': seen_class_ids,
        'unseen_classes': unseen_class_ids,
        'trainval_files': trainval_files,
        'trainval_label': torch.from_numpy(trainval_labels),
        'test_seen_files': test_seen_files,
        'test_seen_label': torch.from_numpy(test_seen_labels),
        'test_unseen_files': test_unseen_files,
        'test_unseen_label': torch.from_numpy(test_unseen_labels),
        'ori_attributes': attributes,
        'attributes': attributes, # The project seems to use both keys
    }

    # 6. Save to file
    output_path = OUTPUT_DIR / OUTPUT_FILENAME
    print(f"Saving final dataset file to: {output_path}")
    torch.save(dataset_dict, output_path)

    print("--- Dataset preparation complete! ---")
    print(f"You can now run training with '--dataset Fish'")

if __name__ == '__main__':
    prepare_fish_dataset()
