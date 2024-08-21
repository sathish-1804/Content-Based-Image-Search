# Image Description and Search System

This project is a machine learning-based system for generating detailed descriptions of images and enabling a search functionality based on these descriptions. The system leverages models like BERT for creating embeddings and integrates with the Google Generative AI model to generate image descriptions. The final output is a Streamlit-based web application where users can search for images based on textual queries.

## Project Overview

The project is divided into three main components:

### Image Description Generation (`create_descriptions.py`)

- Loads images from various datasets.
- Generates detailed descriptions for each image using Google Generative AI.
- Saves the descriptions and the corresponding images to a CSV file and local directory, respectively.

### Description Embedding Creation (`create_embeddings.py`)

- Uses BERT to create embeddings from the generated image descriptions.
- Saves the embeddings in a pickle file for later use in search queries.

### Image Search Application (`app.py`)

- Provides a user interface using Streamlit for searching images.
- Uses FAISS to perform efficient similarity searches over the BERT embeddings.
- Returns and displays the top matching images based on the user's search query.

## Requirements

- Python 3.x
- Libraries: `torch`, `PIL`, `google.generativeai`, `datasets`, `faiss`, `streamlit`, `transformers`, `pandas`, `numpy`, `pickle`

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/image-description-search.git
    cd image-description-search
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up your Google Generative AI API key:

    ```bash
    export GEMINI_KEY='your-api-key'
    ```

