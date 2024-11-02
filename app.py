import streamlit as st
import pickle
import pandas as pd
from typing import Any
import os
import sys
import traceback
from pathlib import Path

class MLModelDeployer:
    def __init__(self):
        self.model = None
        self.model_info = {}
        self.allowed_extensions = ['.pkl']
        
    def validate_file(self, file) -> bool:
        """Validate file extension and permissions"""
        try:
            file_extension = Path(file.name).suffix.lower()
            if file_extension not in self.allowed_extensions:
                st.error(f"Invalid file type. Please upload a pickle file (.pkl)")
                return False
                
            # Check file size (limit to 100MB for safety)
            if file.size > 100 * 1024 * 1024:  
                st.error("File size too large. Please upload a file smaller than 100MB")
                return False
                
            return True
        except Exception as e:
            st.error(f"Error validating file: {str(e)}")
            return False
    
    def load_model(self, uploaded_file) -> bool:
        """Load a pickle file containing the ML model with enhanced error handling"""
        try:
            if not self.validate_file(uploaded_file):
                return False
                
            # Create a secure temporary file
            with uploaded_file as file:
                try:
                    self.model = pickle.load(file)
                except pickle.UnpicklingError:
                    st.error("Error: Invalid or corrupted pickle file")
                    return False
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
                    return False
            
            # Extract model information
            self.model_info = {
                'type': type(self.model).__name__,
                'features': self.get_model_features(),
                'file_name': uploaded_file.name,
                'file_size': f"{uploaded_file.size / 1024:.2f} KB"
            }
            return True
            
        except PermissionError:
            st.error("""
                Permission Error (403): Unable to access the file.
                Please check:
                1. File permissions
                2. File is not in use by another program
                3. You have read access to the file location
            """)
            return False
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
            return False
    
    def get_model_features(self) -> list:
        """Extract feature names from the model if available"""
        try:
            if hasattr(self.model, 'feature_names_in_'):
                return list(self.model.feature_names_in_)
            elif hasattr(self.model, 'feature_names'):
                return list(self.model.feature_names)
            elif hasattr(self.model, 'get_feature_names'):
                return list(self.model.get_feature_names())
            return []
        except Exception as e:
            st.warning(f"Could not extract feature names: {str(e)}")
            return []
    
    def predict(self, input_data: pd.DataFrame) -> Any:
        """Make predictions using the loaded model"""
        if self.model is None:
            raise ValueError("No model loaded")
            
        try:
            return self.model.predict(input_data)
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None

def main():
    st.set_page_config(page_title="ML Model Deployer", layout="wide")
    
    st.title("ML Model Deployment System")
    
    # Initialize the deployer
    deployer = MLModelDeployer()
    
    # File upload section with clear instructions
    st.header("1. Upload Your Model")
    st.markdown("""
        ### Instructions:
        1. Ensure your pickle file has appropriate permissions
        2. File size should be less than 100MB
        3. The model should be a standard ML model (scikit-learn, tensorflow, etc.)
    """)
    
    uploaded_file = st.file_uploader(
        "Choose a pickle file", 
        type=['pkl'],
        help="Upload a trained ML model in pickle format (.pkl)"
    )
    
    if uploaded_file is not None:
        with st.spinner("Loading model..."):
            if deployer.load_model(uploaded_file):
                st.success("✅ Model loaded successfully!")
                
                # Display model information in an expandable section
                with st.expander("Model Information", expanded=True):
                    st.json(deployer.model_info)
                
                # Input form for predictions
                st.header("2. Make Predictions")
                
                if deployer.model_info.get('features'):
                    # Create columns for better layout
                    cols = st.columns(3)
                    input_data = {}
                    
                    # Create input fields for each feature
                    for idx, feature in enumerate(deployer.model_info['features']):
                        with cols[idx % 3]:
                            input_data[feature] = st.number_input(
                                f"{feature}",
                                value=0.0,
                                help=f"Enter value for {feature}"
                            )
                    
                    if st.button("Generate Prediction", type="primary"):
                        try:
                            # Convert input data to DataFrame
                            df = pd.DataFrame([input_data])
                            
                            # Make prediction
                            with st.spinner("Generating prediction..."):
                                prediction = deployer.predict(df)
                            
                            if prediction is not None:
                                # Display prediction
                                st.header("3. Prediction Results")
                                st.success(f"Prediction: {prediction}")
                                
                                # Option to download prediction
                                prediction_df = pd.DataFrame({
                                    'Feature': list(input_data.keys()),
                                    'Value': list(input_data.values()),
                                    'Prediction': prediction
                                })
                                st.download_button(
                                    "Download Prediction Results",
                                    prediction_df.to_csv(index=False).encode('utf-8'),
                                    "prediction_results.csv",
                                    "text/csv"
                                )
                        
                        except Exception as e:
                            st.error(f"Error making prediction: {str(e)}")
                else:
                    st.warning("No feature information available in the model")

if __name__ == "__main__":
    main()