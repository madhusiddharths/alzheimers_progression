import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
from typing import Dict, Union

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
CHECKPOINT_DIR = "../gans/madhu/" # Relative to GAN generators
CLASSIFIER_PATH = "../model_classifier/efficientnet_b4_pytorch.pth" # Relative to classifier model
GAN_IMAGE_SIZE = (256, 256)
CLASSIFIER_IMAGE_SIZE = (256, 256)

CLASSES = ['Mild Dementia', 'Moderate Dementia', 'Non Demented', 'Very mild Dementia']
MILD_DEMENTED = CLASSES[0]
MODERATE_DEMENTED = CLASSES[1]
NON_DEMENTED = CLASSES[2]
VERY_MILD_DEMENTED = CLASSES[3]

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1), # Outputs 1 channel
            nn.Tanh() # Output range [-1, 1]
        )

    def forward(self, input):
        return self.main(input)

# Transforms
gan_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(GAN_IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

classifier_transform = transforms.Compose([
    transforms.Resize(CLASSIFIER_IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_and_preprocess_gan_image(image_path: str) -> torch.Tensor:
    """Loads and preprocesses an image for the GAN model."""
    try:
        image = Image.open(image_path)
        if image.mode == 'RGBA':
            image = image.convert('RGB') 
        return gan_transform(image)
    except Exception as e:
        print(f"Error loading or preprocessing image {image_path} for GAN: {e}")
        raise

classifier_model = None
def load_classifier_model():
    global classifier_model
    if classifier_model is None:
        try:
            script_dir = os.path.dirname(__file__)
            abs_classifier_path = os.path.abspath(os.path.join(script_dir, CLASSIFIER_PATH))
            if not os.path.exists(abs_classifier_path):
                # Don't error immediately, maybe training hasn't run.
                # But warn.
                print(f"WARNING: Classifier model not found at {abs_classifier_path}")
                return None
            
            # Rebuild model structure
            print("Loading PyTorch EfficientNetB4...")
            weights = models.EfficientNet_B4_Weights.DEFAULT
            model = models.efficientnet_b4(weights=weights)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, len(CLASSES))
            
            # Load weights
            state_dict = torch.load(abs_classifier_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state_dict)
            model.to(DEVICE)
            model.eval()
            
            classifier_model = model
            print("Classifier model loaded.") 
        except Exception as e:
            print(f"Error loading classifier model: {e}")
            classifier_model = None 
    return classifier_model


### NEED THIS FIRST
load_classifier_model()


def identify_stage(image_path: str) -> str | None:
    """
    Identifies the Alzheimer's stage of an image using the PyTorch classifier.
    Input: Path to the image file (str).
    Output: String representing the identified stage, or None on error.
    """
    model = load_classifier_model()
    if model is None:
        print("ERROR: Classifier model not loaded.") 
        return None

    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = classifier_transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted_idx = torch.max(outputs, 1)
        
        predicted_class_label = CLASSES[predicted_idx.item()]
        return predicted_class_label

    except Exception as e:
        print(f"ERROR identifying stage for {os.path.basename(image_path)}: {e}") 
        return None

generator_paths = {
    NON_DEMENTED: os.path.join(CHECKPOINT_DIR, "generator_vmild.pth"),
    VERY_MILD_DEMENTED: os.path.join(CHECKPOINT_DIR, "generator_mild.pth"),
    MILD_DEMENTED: os.path.join(CHECKPOINT_DIR, "generator.pth"),
    MODERATE_DEMENTED: None
}


def predict_full_progression(initial_image_path: str, start_stage_override: str = None) -> Dict[str, torch.Tensor]:
    """
    Predicts the progression of Alzheimer's stage-by-stage until Moderate Dementia.
    The classifier is used ONLY to determine the starting stage, unless start_stage_override is provided.
    Subsequent steps use the predefined generator sequence.
    Input: Path to the initial image file.
           start_stage_override: Optional string to force a starting stage (e.g. 'Non Demented').
    Output: Dictionary mapping stage names to corresponding image tensors.
            Returns an empty dictionary if the initial image cannot be processed or identified.
    """
    results: Dict[str, torch.Tensor] = {}

    print(f"--- Starting Progression: {os.path.basename(initial_image_path)} ---")

    if start_stage_override:
        start_stage = start_stage_override
        print(f"Forced Initial Stage: {start_stage}")
    else:
        start_stage = identify_stage(initial_image_path)
        if start_stage is None:
            print("ERROR: Failed to identify initial stage.") 
            return {}
        print(f"Initial Stage: {start_stage}")
    
    
    try:
        current_tensor = load_and_preprocess_gan_image(initial_image_path)
    except Exception as e:
        print(f"ERROR loading initial image {os.path.basename(initial_image_path)}: {e}") 
        return {}

    if start_stage == MODERATE_DEMENTED:
        print(f"Already at final stage ({MODERATE_DEMENTED}).")
        return results 

    # Define the correct disease progression order
    PROGRESSION_ORDER = [NON_DEMENTED, VERY_MILD_DEMENTED, MILD_DEMENTED, MODERATE_DEMENTED]

    try:
        # Find where we are starting in the progression
        start_index = PROGRESSION_ORDER.index(start_stage)
    except ValueError:
        print(f"ERROR: Stage '{start_stage}' not found in PROGRESSION_ORDER list.") 
        return {}

    # Iterate through the remaining stages in the progression
    for i in range(start_index, len(PROGRESSION_ORDER) - 1):
        current_expected_stage = PROGRESSION_ORDER[i]
        next_expected_stage = PROGRESSION_ORDER[i+1]
        print(f"\n--- Generating: {current_expected_stage} -> {next_expected_stage} ---") 

        generator_path_key = generator_paths.get(current_expected_stage)
        if generator_path_key is None:
            print(f"ERROR: No generator path for stage '{current_expected_stage}'.") 
            return results

        script_dir = os.path.dirname(__file__)
        generator_path = os.path.abspath(os.path.join(script_dir, generator_path_key))

        if not os.path.exists(generator_path):
            print(f"ERROR: Generator not found: {generator_path}. Stopping.") 
            return results

        generator = Generator().to(DEVICE)
        try:
            state_dict = torch.load(generator_path, map_location=DEVICE, weights_only=True)
            if not isinstance(state_dict, dict):
                 raise TypeError("Checkpoint is not a state_dict.")
            generator.load_state_dict(state_dict)
            generator.eval()
        except TypeError as e:
            print(f"ERROR: Checkpoint file {os.path.basename(generator_path)} not valid state_dict (use weights_only=False?): {e}") 
            return results
        except Exception as e:
            print(f"ERROR loading generator state_dict from {os.path.basename(generator_path)}: {e}. Stopping.") 
            return results

        try:
            with torch.no_grad():
                next_tensor = generator(current_tensor.unsqueeze(0).to(DEVICE))
                next_tensor = next_tensor.squeeze(0)

            current_tensor = next_tensor
            results[next_expected_stage] = current_tensor.cpu().clone()

        except Exception as e:
            print(f"ERROR during generation for stage {next_expected_stage}: {e}. Stopping.") # Emphasize error
            return results

    print(f"--- Progression Finished ---")
    return results

if __name__ == '__main__':
    
    
    image_to_process = "/Users/thuptenwangpo/Documents/GitHub/CS584-Project/data/source_2/Mild Dementia/OAS1_0028_MR1_mpr-1_117.jpg" 
    
    # Simple test check
    print(f"Using Device: {DEVICE}")

    if os.path.exists(image_to_process):
        print(f"Running prediction for: {image_to_process}") 
        progression_results = predict_full_progression(image_to_process)

        if progression_results:
            print(f"\n--- Results ({len(progression_results)} generated stages) ---")
            output_dir = "progression_output"
            os.makedirs(output_dir, exist_ok=True)
            print(f"Saving images to ./{output_dir}/")

            for stage_name, image_tensor in progression_results.items():
                
                try:
                    from torchvision.utils import save_image
                    filename_stage = stage_name.replace(" ", "_").lower()
                    output_filename = os.path.join(output_dir, f"predicted_{filename_stage}.png")
                    save_image(image_tensor.cpu() * 0.5 + 0.5, output_filename)
                    print(f"  Saved: {output_filename}")
                except Exception as e:
                    print(f"    ERROR saving image for stage {stage_name}: {e}")
        else:
             print("Prediction failed or produced no generated stages.") 

    else:
        print(f"ERROR: Input image not found: {image_to_process}") 
