import streamlit as st
import google.generativeai as genai
import torch
import dgl
import numpy as np
import os
import tempfile
from PIL import Image
from time_delay import TimingPredictor
from verilog_to_graph import parse_verilog, extract_nets_and_instances, build_graph
import base64
import matplotlib.pyplot as plt
import subprocess
import sys
import time
from google.api_core import exceptions as google_exceptions

# Configure Gemini API using secrets
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')
vision_model = genai.GenerativeModel('gemini-pro-vision')

def process_image_with_retry(image, max_retries=3, initial_wait=5):
    """Process image with retry logic for rate limits"""
    prompt = """
    Analyze this digital circuit image and convert it to Verilog code. Follow these rules:
    1. Identify all logic gates and their connections
    2. Create module inputs and outputs
    3. Use standard cell library components (sky130_fd_sc_hd__)
    4. Include timing-relevant information
    5. Format the code similar to the example below:
    
    module example (
        input VGND,
        input VPWR,
        input clk,
        ...
    );
        // Internal wires
        wire _00000_;
        ...
        // Buffer instantiations
        sky130_fd_sc_hd__buf_8 buffer1 (
            .A(input_signal),
            .X(output_signal),
            ...
        );
        ...
    endmodule
    """
    
    for attempt in range(max_retries):
        try:
            response = vision_model.generate_content([prompt, image])
            return response.text
        except google_exceptions.ResourceExhausted as e:
            if attempt < max_retries - 1:
                wait_time = initial_wait * (2 ** attempt)  # Exponential backoff
                st.warning(f"Rate limit reached. Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
                continue
            else:
                st.error("Rate limit exceeded. Please try again later or contact support for increased quota.")
                raise
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            raise

def process_verilog(verilog_code, filename="temp_design"):
    """Save Verilog code to file and process it"""
    # Create a directory for temporary files if it doesn't exist
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_designs")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save Verilog code to file
    verilog_file = os.path.join(temp_dir, f"{filename}.v")
    with open(verilog_file, "w") as f:
        f.write(verilog_code)
    
    # Generate output graph filename
    graph_file = os.path.join(temp_dir, f"{filename}.graph.bin")
    
    try:
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        verilog_to_graph_path = os.path.join(current_dir, "verilog_to_graph.py")
        
        # Run verilog_to_graph.py using subprocess
        cmd = [sys.executable, verilog_to_graph_path, "--verilog", verilog_file, "--output", graph_file]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Verilog to graph conversion failed: {result.stderr}")
        
        if not os.path.exists(graph_file):
            raise Exception(f"Graph file was not created at {graph_file}")
            
        return graph_file
        
    except Exception as e:
        st.error(f"Error in processing Verilog: {str(e)}")
        raise
    finally:
        # Clean up Verilog file but keep the graph file for timing analysis
        if os.path.exists(verilog_file):
            os.remove(verilog_file)

def main():
    st.title("RTL to Timing Analysis Pipeline")
    st.write("Upload Verilog code or circuit image for timing analysis")
    
    # File upload section
    upload_type = st.radio("Choose input type:", ["Verilog Code", "Circuit Image", "Both"])
    
    verilog_code = None
    
    if upload_type in ["Verilog Code", "Both"]:
        verilog_file = st.file_uploader("Upload Verilog file", type=['v'])
        if verilog_file:
            verilog_code = verilog_file.getvalue().decode()
            st.code(verilog_code, language='verilog')
    
    if upload_type in ["Circuit Image", "Both"]:
        image_file = st.file_uploader("Upload circuit image", type=['png', 'jpg', 'jpeg'])
        if image_file:
            image = Image.open(image_file)
            st.image(image, caption="Uploaded Circuit")
            
            if st.button("Process Image"):
                with st.spinner("Processing image with Gemini Vision..."):
                    try:
                        verilog_code = process_image_with_retry(image)
                        if verilog_code:
                            st.code(verilog_code, language='verilog')
                    except Exception as e:
                        st.error("Failed to process image. Please try again later.")
    
    if verilog_code and st.button("Run Timing Analysis"):
        with st.spinner("Processing..."):
            try:
                # Generate a unique filename based on timestamp
                filename = f"design_{int(time.time())}"
                
                # Process Verilog to graph
                graph_path = process_verilog(verilog_code, filename)
                st.info(f"Graph generated at: {graph_path}")
                
                # Run timing prediction
                model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pth")
                predictor = TimingPredictor(model_path)
                delays = predictor.predict(graph_path)
                
                # Display results
                st.success("Analysis Complete!")
                st.write("Predicted Delays:")
                st.write(delays)
                
                # Visualize results
                fig = plt.figure(figsize=(10, 6))
                plt.hist(delays.numpy().flatten(), bins=50)
                plt.xlabel("Delay (s)")
                plt.ylabel("Frequency")
                st.pyplot(fig)
                
                # Cleanup
                if os.path.exists(graph_path):
                    os.remove(graph_path)
                
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    main() 