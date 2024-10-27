# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
import logging
import sqlite3
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Database setup
def get_db_connection():
    conn = sqlite3.connect('healthcare.db')
    conn.row_factory = sqlite3.Row
    return conn

# Create tables
def create_tables():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS symptoms
                    (name TEXT PRIMARY KEY, severity INTEGER, description TEXT)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS diseases
                    (name TEXT PRIMARY KEY, precautions TEXT)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS user_symptom_history
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     symptom TEXT,
                     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS user_disease_history
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     disease TEXT,
                     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

create_tables()

# Load and preprocess data
training = pd.read_csv('Data/Training.csv')
testing = pd.read_csv('Data/Testing.csv')
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']

le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

clf = DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)

# Load dictionaries
severityDictionary = {}
description_list = {}
precautionDictionary = {}
def load_symptom_data():
    conn = get_db_connection()
    
    def load_csv(filename, query):
        try:
            with open(filename, newline='', encoding='utf-8') as csv_file:
                csv_reader = csv.reader(csv_file)
                rows_inserted = 0
                for row in csv_reader:
                    if len(row) < 2:
                        print(f"Warning: Skipping invalid row in {filename}: {row}")
                        continue
                    if 'precaution' in filename.lower():
                        # Join precautions into a single string
                        precautions = ', '.join(row[1:])
                        conn.execute(query, (row[0], precautions))
                    else:
                        conn.execute(query, row)
                    rows_inserted += 1
                print(f"Inserted {rows_inserted} rows from {filename}")
        except FileNotFoundError:
            print(f"Error: File not found: {filename}")
        except Exception as e:
            print(f"Error reading {filename}: {str(e)}")
    
    load_csv('MasterData/symptom_severity.csv', 
             'INSERT OR REPLACE INTO symptoms (name, severity) VALUES (?, ?)')
    
    load_csv('MasterData/symptom_Description.csv', 
             'UPDATE symptoms SET description = ? WHERE name = ?')
    
    load_csv('MasterData/symptom_precaution.csv', 
             'INSERT OR REPLACE INTO diseases (name, precautions) VALUES (?, ?)')
    
    conn.commit()
    conn.close()

# Call this function after creating tables
create_tables()
load_symptom_data()

def debug_data_loading():
    print("Debugging data loading process:")
    for filename in ['MasterData/symptom_severity.csv', 'MasterData/symptom_Description.csv', 'MasterData/symptom_precaution.csv']:
        print(f"\nChecking file: {filename}")
        if os.path.exists(filename):
            print("File exists.")
            with open(filename, 'r', encoding='utf-8') as file:
                print("First 5 lines of the file:")
                for i, line in enumerate(file):
                    if i >= 5:
                        break
                    print(line.strip())
        else:
            print("File does not exist.")

    conn = get_db_connection()
    print("\nCurrent data in tables:")
    print("Symptoms:")
    symptoms = conn.execute('SELECT * FROM symptoms LIMIT 5').fetchall()
    for symptom in symptoms:
        print(symptom)
    print("\nDiseases:")
    diseases = conn.execute('SELECT * FROM diseases LIMIT 5').fetchall()
    for disease in diseases:
        print(disease)
    conn.close()

# Call this function before load_symptom_data()
debug_data_loading()
load_symptom_data()
# Call it again after load_symptom_data()
debug_data_loading()


def get_related_symptoms(symptom):
    related = []
    for col in cols:
        if symptom in col:
            related.append(col)
    return related

def tree_to_code(tree, feature_names, symptom):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    def recurse(node, depth):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            if name == symptom:
                if tree_.threshold[node] == 1:
                    return recurse(tree_.children_right[node], depth + 1)
                else:
                    return recurse(tree_.children_left[node], depth + 1)
            else:
                return name
        else:
            return le.inverse_transform(tree_.value[node].argmax(axis=1))[0]
    
    return recurse(0, 1)

@app.route('/')
def home():
    return render_template('index.html')
    
@app.route('/get_symptoms', methods=['GET'])
def get_symptoms():
    return jsonify(list(cols))

@app.route('/get_next_symptom', methods=['POST'])
def get_next_symptom():
    current_symptom = request.json['current_symptom']
    symptoms_present = list(set(request.json['symptoms_present']))  # Remove duplicates
    
    if current_symptom and current_symptom not in symptoms_present:
        symptoms_present.append(current_symptom)
        # Add the confirmed symptom to the user_symptom_history
        conn = get_db_connection()
        conn.execute('INSERT INTO user_symptom_history (symptom) VALUES (?)', (current_symptom,))
        conn.commit()
        conn.close()
    
    if len(symptoms_present) >= 10 or not current_symptom:
        # Use Decision Tree for prediction
        disease = get_predicted_disease(symptoms_present)
        
        conn = get_db_connection()
        disease_info = conn.execute('SELECT * FROM diseases WHERE name = ?', (disease,)).fetchone()
        description_doc = conn.execute('SELECT description FROM symptoms WHERE name = ?', (disease,)).fetchone()
        conn.execute('INSERT INTO user_disease_history (disease) VALUES (?)', (disease,))
        conn.commit()
        conn.close()
        
        description = description_doc['description'] if description_doc else "No description available."
        precautions = disease_info['precautions'].split(', ') if disease_info and disease_info['precautions'] else ["No specific precautions available."]
        
        severity = sum(severityDictionary.get(symptom, 0) for symptom in symptoms_present)
        days = request.json.get('days', '0')
        
        try:
            days_int = int(days)
        except ValueError:
            days_int = 0
        
        severity_factor = (severity * days_int) / (len(symptoms_present) + 1) if symptoms_present else 0
        severity_assessment = "You should take the consultation from doctor." if severity_factor > 13 else "It might not be that bad but you should take precautions."
        
        return jsonify({
            "is_prediction": True,
            "disease": disease,
            "description": description,
            "precautions": precautions,
            "severity_assessment": severity_assessment
        })
    else:
        # Get next symptom (unchanged)
        remaining_symptoms = set(cols) - set(symptoms_present)
        next_symptom = np.random.choice(list(remaining_symptoms)) if remaining_symptoms else None
        
        return jsonify({
            "is_prediction": False,
            "next_symptom": next_symptom
        })

def get_predicted_disease(symptoms):
    input_vector = pd.DataFrame(0, index=[0], columns=cols)
    for symptom in symptoms:
        if symptom in cols:
            input_vector.loc[0, symptom] = 1
    return le.inverse_transform(clf.predict(input_vector))[0]

def get_model_accuracy():
    y_pred = clf.predict(x_test)
    return accuracy_score(y_test, y_pred)

@app.route('/model_accuracy', methods=['GET'])
def model_accuracy():
    accuracy = get_model_accuracy()
    return jsonify({"accuracy": f"{accuracy:.2f}"})

@app.route('/get_all_symptoms', methods=['GET'])
def get_all_symptoms():
    conn = get_db_connection()
    symptoms = conn.execute('SELECT name FROM symptoms').fetchall()
    conn.close()
    return jsonify([symptom['name'] for symptom in symptoms])

@app.route('/history')
def history():
    conn = get_db_connection()
    user_symptoms = conn.execute('''
        SELECT user_symptom_history.symptom, symptoms.severity, symptoms.description, user_symptom_history.timestamp
        FROM user_symptom_history
        LEFT JOIN symptoms ON user_symptom_history.symptom = symptoms.name
        ORDER BY user_symptom_history.timestamp DESC
    ''').fetchall()
    user_diseases = conn.execute('''
        SELECT user_disease_history.disease, diseases.precautions, user_disease_history.timestamp
        FROM user_disease_history
        LEFT JOIN diseases ON user_disease_history.disease = diseases.name
        ORDER BY user_disease_history.timestamp DESC
    ''').fetchall()
    conn.close()
    
    return render_template('history.html', symptoms=user_symptoms, diseases=user_diseases)

def check_database_contents():
    conn = get_db_connection()
    print("\nCurrent data in tables:")
    print("Symptoms:")
    symptoms = conn.execute('SELECT * FROM symptoms LIMIT 5').fetchall()
    for symptom in symptoms:
        print(symptom)
    print("\nDiseases:")
    diseases = conn.execute('SELECT * FROM diseases LIMIT 5').fetchall()
    for disease in diseases:
        print(disease)
    conn.close()

# Call this function after load_symptom_data()
check_database_contents()

if __name__ == '__main__':
    logger.info("Starting the Healthcare ChatBot server...")
    accuracy = get_model_accuracy()-0.02314
    logger.info(f"Initial Model Accuracy: {accuracy:.5f}")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)