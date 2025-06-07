Disease Predictor Application
============================

This application uses machine learning to predict possible diseases based on selected symptoms. It provides a user-friendly interface for symptom selection and displays prediction results with reliable medical information sources.

Prerequisites
------------
1. Python 3.7 or higher
2. pip (Python package installer)

Required Files
-------------
1. app.py - Main Flask application
2. templates/index.html - Web interface template
3. static/ - Directory for storing generated plots
4. model.pkl - Pre-trained machine learning model
5. label_encoder.pkl - Label encoder for disease names
6. symptoms_list.pkl - List of available symptoms

Installation Steps
----------------
1. Create a virtual environment (recommended):
   ```
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows:
     ```
     venv\Scripts\activate
     ```
   - Linux/Mac:
     ```
     source venv/bin/activate
     ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

Running the Application
---------------------
1. Make sure all required files are in the correct directories:
   - app.py in the root directory
   - index.html in the templates directory
   - model.pkl, label_encoder.pkl, and symptoms_list.pkl in the root directory
   - Create a 'static' directory if it doesn't exist

2. Start the Flask application:
   ```
   python app.py
   ```

3. Open your web browser and go to:
   ```
   http://localhost:5000
   ```

Using the Application
-------------------
1. Select your symptoms from the categorized list
2. Click the "Predict Disease" button
3. View the prediction results, including:
   - Predicted disease
   - Top 5 possible conditions
   - Links to medical information
   - Symptom importance analysis
   - Probability distribution chart

Important Notes
-------------
- The application requires an internet connection to access medical information sources
- All predictions are for informational purposes only
- Always consult with healthcare professionals for medical advice
- The static directory needs write permissions for generating plots

Troubleshooting
-------------
1. If you get a "ModuleNotFoundError":
   - Make sure all required packages are installed
   - Verify you're using the correct Python environment

2. If the application doesn't start:
   - Check if port 5000 is available
   - Verify all required files are present
   - Ensure you have write permissions for the static directory

3. If plots aren't generating:
   - Check if the static directory exists
   - Verify write permissions
   - Make sure matplotlib is properly installed

Support
-------
For any issues or questions, please check the project documentation or create an issue in the project repository. 