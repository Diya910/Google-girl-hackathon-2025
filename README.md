# RTL to Timing Analysis Pipeline

Welcome to the RTL to Timing Analysis Pipeline! This project leverages advanced machine learning techniques, including Graph Neural Networks (GNNs), to analyze and predict timing delays in digital circuits. The pipeline is designed to convert RTL Verilog code and circuit images into a unified graph structure, enabling precise timing analysis.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Feature Engineering](#feature-engineering)
- [Why GNN?](#why-gnn)
- [Experiments with Other Models](#experiments-with-other-models)
- [Gemini API Integration](#gemini-api-integration)
- [Dataset](#dataset)
- [Scope for Improvement](#scope-for-improvement)
- [References](#references)

## Overview

This project aims to provide a robust framework for timing analysis of digital circuits. By converting RTL Verilog code and circuit images into a graph structure, we can leverage GNNs to predict timing delays with high accuracy. The pipeline supports various input formats and uses the Gemini API for AI-driven conversion of circuit images to Verilog code.

## Installation

To get started, ensure you have Python 3.10 installed. This version is chosen for its compatibility with the machine learning libraries and features.

1. Clone the repository:
   ```bash
   git clone https://github.com/Diya910/Google-girl-hackathon-2025
   cd Google-girl-hackathon-2025
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements_webapp.txt
   ```

3. Set up the Google API key in `.streamlit/secrets.toml`:
   ```toml
   GOOGLE_API_KEY = "your-api-key-here"
   ```

## Usage

1. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

2. **Upload Verilog Code or Circuit Image**:
   - The app allows you to upload Verilog files or circuit images.
   - The Gemini API processes images to generate Verilog code.

3. **Run Timing Analysis**:
   - The app converts Verilog to a graph and predicts timing delays.
   - Visualize the results with histograms of predicted delays.

## Model Training

The GNN model is trained using a dataset of digital circuits, focusing on key features like fan-in/fan-out and gate types. The training process involves:

- **Data Preparation**: Convert Verilog files to graph structures as .graph.bin files.
- **Feature Engineering**: Identify parameters influencing signal depth, such as fan-in/fan-out and synthesis optimizations.
- **Training**: Use the `gnn_model.py`, `graph_dataset.py` and `gnn_model_train.py` scripts to train the model.

## Feature Engineering

Feature engineering is crucial for accurate timing predictions. Key features include:

- **Fan-In/Fan-Out**: Measures of connectivity affecting signal propagation.
- **Gate Types**: Different gates have varying delays.
- **Synthesis Optimizations**: Potential improvements in circuit design.

## Why GNN?

GNNs are ideal for this task due to their ability to capture complex relationships in graph-structured data. They outperform traditional models like XGBoost in handling the intricacies of circuit topologies.

## Experiments with Other Models

Before settling on GNNs, we experimented with models like XGBoost and Random Forests. While these models provided some insights, they lacked the ability to fully capture the graph-based nature of digital circuits.

## Gemini API Integration

The Gemini API is used to convert circuit images into Verilog code. This AI-driven approach ensures consistent formatting and structure, facilitating seamless integration into the pipeline.

## Dataset

The dataset used for training is available on Google Drive: [Dataset Link](https://drive.google.com/drive/folders/1IVbEQ0au1zKZfHPNpyV1ZyT_2nHAF4pz?usp=sharing). It includes a variety of digital circuits in Verilog format.

## Scope for Improvement

While the current pipeline is robust, there are areas for enhancement:

- **Increased Dataset Diversity**: Incorporating more circuit types.
- **Advanced Feature Engineering**: Exploring additional parameters.
- **Model Optimization**: Fine-tuning hyperparameters for better performance.

## References

This project references various Verilog files and leverages the Gemini API for AI-driven conversions. The integration of these technologies provides a comprehensive solution for timing analysis in digital circuits. 