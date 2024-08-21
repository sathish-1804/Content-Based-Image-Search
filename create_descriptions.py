import torch
from PIL import Image
import google.generativeai as genai
from datasets import load_dataset
import csv
import os
import base64
from io import BytesIO

# Configure the API key
genai.configure(api_key=os.getenv("GEMINI_KEY"))
model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")

# Set a seed for reproducibility
torch.manual_seed(0)

# Define datasets
datasets = [
    ("detection-datasets/fashionpedia", None, "val"),
    ("keremberke/nfl-object-detection", "mini", "test"),
    ("keremberke/plane-detection", "mini", "train"),
    ("Matthijs/snacks", None, "validation"),
    ("rokmr/mini_pets", None, "test"),
    ("keremberke/pokemon-classification", "mini", "train"),
]

counter = 0

# Ensure the images directory exists
os.makedirs("images", exist_ok=True)

# Open the CSV file for writing
with open("descriptions.csv", "w", newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    # Write the header
    csvwriter.writerow(['Image', 'Description'])

    # Process each dataset
    for name, config, split in datasets:
        d = load_dataset(name, config, split=split)
        num_images = min(len(d), 10)  # Only process up to 10 images per dataset
        for idx in range(num_images):
            # Load the image directly from the dataset
            image = d[idx]["image"].convert("RGB")

            # Save the image locally
            saved_image_path = f"images/{counter}.jpg"
            image.save(saved_image_path)

            # Convert image to base64 string
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # Create the input structure for the model
            content = [
                {
                    "parts": [
                        {"mime_type": "image/jpeg", "data": img_str},
                        {"text": "Create an extensive description of this image, describing the details of the image in full detail."}
                    ]
                }
            ]

            # Generate detailed description
            response = model.generate_content(content)

            # Check if there are candidates in the response
            if response is not None and response.candidates and len(response.candidates) > 0:
                # Extract the generated description text
                description = response.candidates[0].content.parts[0].text
            else:
                description = "No description available"

            # Print the description
            print(counter, description)

            # Write the description to the CSV file
            csvwriter.writerow([saved_image_path, description])

            counter += 1

            # Clear GPU memory
            torch.cuda.empty_cache()
