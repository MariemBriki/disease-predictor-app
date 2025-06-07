import joblib
import numpy as np
from itertools import combinations

# Load the model and data
print("Loading model and data...")
model = joblib.load('disease_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
symptoms_list = joblib.load('symptoms_list.pkl')

def get_prediction(symptom_indices):
    """Get prediction for a combination of symptoms"""
    symptom_vector = np.zeros(len(symptoms_list))
    for idx in symptom_indices:
        symptom_vector[idx] = 1
    
    probabilities = model.predict_proba([symptom_vector])[0]
    disease_names = label_encoder.classes_
    
    # Get top prediction
    top_idx = np.argmax(probabilities)
    if probabilities[top_idx] > 0.4:  # Only include if probability > 40%
        return {
            'disease': disease_names[top_idx],
            'probability': probabilities[top_idx]
        }
    return None

print("Analyzing 2-symptom combinations (showing up to 20 results >40%)...")
max_results = 20
found = 0
for combo in combinations(range(len(symptoms_list)), 2):
    result = get_prediction(combo)
    if result:
        symptoms = [symptoms_list[i] for i in combo]
        print(f"\nSymptoms: {', '.join(symptoms)}")
        print(f"Predicted Disease: {result['disease']}")
        print(f"Probability: {result['probability']*100:.1f}%")
        found += 1
        if found >= max_results:
            break
if found == 0:
    print("No 2-symptom combinations found with probability > 40%.") 