# Disease Predictor Web Application

A Flask web application that predicts possible diseases based on selected symptoms using a trained machine learning model.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Make sure you have the following files in the root directory:
   - disease_model.pkl
   - label_encoder.pkl
   - symptoms_list.pkl

2. Run the Flask application:
```bash
python app.py
```

3. Open your web browser and navigate to:
```
http://localhost:5000
```

## Features

- Interactive symptom selection form
- Real-time disease prediction
- Probability distribution visualization
- Responsive design using Bootstrap
- Clean and user-friendly interface

## Project Structure

```
.
├── app.py              # Main Flask application
├── templates/          # HTML templates
│   └── index.html     # Main template
├── static/            # Static files (generated plots)
├── disease_model.pkl  # Trained ML model
├── label_encoder.pkl  # Label encoder for disease names
├── symptoms_list.pkl  # List of possible symptoms
└── requirements.txt   # Python dependencies
``` 
