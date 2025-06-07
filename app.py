from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pandas as pd

app = Flask(__name__)

# Create static directory if it doesn't exist
if not os.path.exists('static'):
    os.makedirs('static')

# Load the model and encoders
model = joblib.load('disease_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
symptoms_list = joblib.load('symptoms_list.pkl')

def prettify_symptom(symptom):
    return symptom.replace('_', ' ').title()

# Example categories (customize as needed)
symptom_categories = {
    'Respiratory': [
        'cough', 'breathlessness', 'phlegm', 'throat_irritation', 'runny_nose',
        'congestion', 'chest_pain', 'mucoid_sputum', 'rusty_sputum', 'blood_in_sputum',
        'continuous_sneezing', 'sinus_pressure', 'patches_in_throat'
    ],
    'Gastrointestinal': [
        'nausea', 'vomiting', 'diarrhoea', 'abdominal_pain', 'stomach_pain',
        'acidity', 'ulcers_on_tongue', 'indigestion', 'loss_of_appetite',
        'constipation', 'pain_during_bowel_movements', 'pain_in_anal_region',
        'bloody_stool', 'irritation_in_anus', 'passage_of_gases', 'belly_pain',
        'stomach_bleeding', 'distention_of_abdomen', 'swelling_of_stomach'
    ],
    'General': [
        'fever', 'high_fever', 'mild_fever', 'fatigue', 'sweating',
        'chills', 'shivering', 'malaise', 'dehydration', 'lethargy',
        'restlessness', 'weakness_in_limbs', 'muscle_weakness', 'altered_sensorium',
        'coma', 'toxic_look_(typhos)', 'family_history'
    ],
    'Neurological': [
        'headache', 'dizziness', 'pain_behind_the_eyes', 'blurred_and_distorted_vision',
        'visual_disturbances', 'lack_of_concentration', 'slurred_speech',
        'spinning_movements', 'loss_of_balance', 'unsteadiness',
        'weakness_of_one_body_side', 'loss_of_smell', 'movement_stiffness',
        'stiff_neck', 'depression', 'irritability', 'anxiety', 'mood_swings'
    ],
    'Skin': [
        'itching', 'skin_rash', 'nodal_skin_eruptions', 'yellowish_skin',
        'yellowing_of_eyes', 'red_spots_over_body', 'dischromic _patches',
        'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
        'silver_like_dusting', 'blister', 'red_sore_around_nose',
        'yellow_crust_ooze', 'internal_itching'
    ],
    'Musculoskeletal': [
        'joint_pain', 'muscle_wasting', 'muscle_pain', 'back_pain',
        'neck_pain', 'knee_pain', 'hip_joint_pain', 'cramps',
        'swelling_joints', 'bruising', 'brittle_nails', 'small_dents_in_nails',
        'inflammatory_nails', 'painful_walking'
    ],
    'Cardiovascular': [
        'chest_pain', 'palpitations', 'fast_heart_rate', 'swollen_blood_vessels',
        'prominent_veins_on_calf', 'cold_hands_and_feets'
    ],
    'Urinary': [
        'burning_micturition', 'spotting_ urination', 'dark_urine',
        'yellow_urine', 'bladder_discomfort', 'foul_smell_of urine',
        'continuous_feel_of_urine', 'polyuria'
    ],
    'Endocrine': [
        'weight_gain', 'weight_loss', 'obesity', 'excessive_hunger',
        'increased_appetite', 'irregular_sugar_level', 'enlarged_thyroid',
        'swollen_extremeties', 'puffy_face_and_eyes', 'swollen_legs'
    ],
    'Immune': [
        'swelled_lymph_nodes', 'fluid_overload', 'fluid_overload.1',
        'receiving_blood_transfusion', 'receiving_unsterile_injections',
        'history_of_alcohol_consumption'
    ],
    'Eyes': [
        'sunken_eyes', 'redness_of_eyes', 'watering_from_eyes'
    ],
    'Liver': [
        'acute_liver_failure', 'yellowish_skin', 'yellowing_of_eyes'
    ],
    'Other': []  # Will fill with truly uncategorized symptoms
}

# Assign symptoms to categories
categorized_symptoms = {cat: [] for cat in symptom_categories}
for symptom in symptoms_list:
    found = False
    for cat, syms in symptom_categories.items():
        if symptom in syms:
            categorized_symptoms[cat].append(symptom)
            found = True
            break
    if not found:
        categorized_symptoms['Other'].append(symptom)

# Debug route to see uncategorized symptoms
@app.route('/debug/uncategorized')
def show_uncategorized():
    return jsonify({
        'uncategorized_symptoms': categorized_symptoms['Other'],
        'total_symptoms': len(symptoms_list),
        'uncategorized_count': len(categorized_symptoms['Other'])
    })

def get_symptom_importance(symptom_vector, model, label_encoder):
    """Calculate how important each selected symptom is for the prediction"""
    base_prob = model.predict_proba([symptom_vector])[0]
    importance = {}
    
    for i, symptom in enumerate(symptoms_list):
        if symptom_vector[i] == 1:  # Only for selected symptoms
            # Create a copy of the vector and remove this symptom
            temp_vector = symptom_vector.copy()
            temp_vector[i] = 0
            # Get new probabilities without this symptom
            new_prob = model.predict_proba([temp_vector])[0]
            # Calculate the difference in probability for the predicted class
            pred_class = np.argmax(base_prob)
            importance[symptom] = base_prob[pred_class] - new_prob[pred_class]
    
    return importance

def get_prediction(input_data):
    """Get disease prediction and related information"""
    # Make prediction
    prediction = model.predict([input_data])[0]
    disease_name = label_encoder.inverse_transform([prediction])[0]
    
    # Get probabilities for all diseases
    probabilities = model.predict_proba([input_data])[0]
    disease_names = label_encoder.classes_
    
    # Get symptom importance
    symptom_importance = get_symptom_importance(input_data, model, label_encoder)
    
    # Sort probabilities and disease names
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_probs = probabilities[sorted_indices]
    sorted_diseases = disease_names[sorted_indices]
    
    # Get top 5 diseases with their probabilities
    top_diseases = list(zip(sorted_diseases[:5], sorted_probs[:5]))
    
    # Get the probability of the predicted disease
    probability = probabilities[prediction]
    
    return disease_name, probability, top_diseases, symptom_importance

def generate_plot(top_diseases):
    """Generate probability distribution plot"""
    # Create the plot
    plt.figure(figsize=(12, 6))
    diseases, probabilities = zip(*top_diseases)
    
    # Create bar plot
    plt.barh(range(len(probabilities)), probabilities)
    plt.yticks(range(len(probabilities)), diseases)
    plt.xlabel('Probability')
    plt.title('Top 5 Disease Probabilities')
    
    # Save plot with timestamp to avoid caching issues
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = f'static/probabilities_{timestamp}.png'
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_symptoms = request.form.getlist('symptoms')
        
        if not selected_symptoms:
            return render_template('index.html', 
                                symptoms_list=symptoms_list,
                                categorized_symptoms=categorized_symptoms,
                                error_message="Please select at least one symptom you are experiencing.",
                                prettify_symptom=prettify_symptom)
        
        # Create input array for prediction
        input_data = np.zeros(len(symptoms_list))
        for symptom in selected_symptoms:
            if symptom in symptoms_list:
                input_data[symptoms_list.index(symptom)] = 1
        
        # Get prediction
        disease_name, probability, top_diseases, symptom_importance = get_prediction(input_data)
        
        # Generate plot
        plot_path = generate_plot(top_diseases)
        
        return render_template('index.html', 
                            disease_name=disease_name,
                            probability=probability,
                            top_diseases=top_diseases,
                            plot_path=plot_path,
                            symptoms_list=symptoms_list,
                            categorized_symptoms=categorized_symptoms,
                            selected_symptoms=selected_symptoms,
                            symptom_importance=symptom_importance,
                            prettify_symptom=prettify_symptom)
    
    return render_template('index.html', 
                         symptoms_list=symptoms_list,
                         categorized_symptoms=categorized_symptoms,
                         prettify_symptom=prettify_symptom)

if __name__ == '__main__':
    app.run(debug=True) 