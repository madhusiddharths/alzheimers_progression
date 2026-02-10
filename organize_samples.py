import os
import shutil

SOURCE_DIR = os.path.join("data", "source_2")
DEST_DIR = "samples"
CATEGORIES = ['Non Demented', 'Very mild Dementia', 'Mild Dementia', 'Moderate Dementia']

def organize_samples():
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
        print(f"Created {DEST_DIR}")

    for category in CATEGORIES:
        src_path = os.path.join(SOURCE_DIR, category)
        dest_path = os.path.join(DEST_DIR, category)
        
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
            print(f"Created {dest_path}")
            
        if not os.path.exists(src_path):
            print(f"WARNING: Source category not found: {src_path}")
            continue
            
        files = [f for f in os.listdir(src_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        files.sort() # Ensure deterministic selection
        
        selected_files = files[:5]
        
        for f in selected_files:
            s = os.path.join(src_path, f)
            d = os.path.join(dest_path, f)
            shutil.copy2(s, d)
            print(f"Copied {f} to {dest_path}")
            
    print("Sample organization complete.")

if __name__ == "__main__":
    organize_samples()
