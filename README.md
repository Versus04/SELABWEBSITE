# Healthcare Symptom Checker Chatbot

## Description
A Flask-based web application that uses machine learning to help users identify potential health conditions based on their symptoms. The application implements a decision tree classifier to predict diseases from user-input symptoms and provides relevant precautions and severity assessments.

## Features
- Interactive symptom-based disease prediction
- Symptom severity assessment
- Disease precautions and descriptions
- User symptom and disease history tracking
- Model accuracy reporting
- SQLite database for persistent storage
- Comprehensive logging system

## Prerequisites
- Python 3.x
- Flask
- pandas
- scikit-learn
- numpy
- SQLite3

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd healthcare-chatbot
```

2. Install required packages:
```bash
pip install flask pandas scikit-learn numpy
```

3. Ensure you have the following data files in their respective directories:

Data folder:
- `Training.csv`
- `Testing.csv`

MasterData folder:
- `symptom_severity.csv`
- `symptom_Description.csv`
- `symptom_precaution.csv`

## Project Structure
```
healthcare-chatbot/
├── app.py
├── healthcare.db
├── Data/
│   ├── Training.csv
│   └── Testing.csv
├── MasterData/
│   ├── symptom_severity.csv
│   ├── symptom_Description.csv
│   └── symptom_precaution.csv
└── templates/
    ├── index.html
    └── history.html
```

## Database Schema

### Symptoms Table
```sql
CREATE TABLE symptoms (
    name TEXT PRIMARY KEY,
    severity INTEGER,
    description TEXT
)
```

### Diseases Table
```sql
CREATE TABLE diseases (
    name TEXT PRIMARY KEY,
    precautions TEXT
)
```

### User History Tables
```sql
CREATE TABLE user_symptom_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symptom TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)

CREATE TABLE user_disease_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    disease TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
```

## API Endpoints

### GET Endpoints
- `/`: Home page
- `/get_symptoms`: Returns list of all possible symptoms
- `/get_all_symptoms`: Returns all symptoms from the database
- `/model_accuracy`: Returns the current model accuracy
- `/history`: Displays user symptom and disease history

### POST Endpoints
- `/get_next_symptom`: Processes current symptoms and returns either:
  - Next symptom to check
  - Disease prediction with description and precautions

## Machine Learning Model
- Uses Decision Tree Classifier from scikit-learn
- Trained on symptom-disease dataset
- Features include various symptoms as binary indicators
- Target variable is the disease prognosis
- Model accuracy is tracked and available via API endpoint

## Key Functions

### Data Management
- `get_db_connection()`: Establishes SQLite database connection
- `create_tables()`: Initializes database schema
- `load_symptom_data()`: Loads symptom and disease data from CSV files

### Prediction System
- `get_related_symptoms()`: Finds symptoms related to input symptom
- `tree_to_code()`: Converts decision tree to executable code
- `get_predicted_disease()`: Predicts disease based on input symptoms

### Debug Functions
- `debug_data_loading()`: Validates data loading process
- `check_database_contents()`: Verifies database contents

## Running the Application

1. Start the server:
```bash
python app.py
```

2. Access the application:
```
http://localhost:5000
```

## Logging
The application uses Python's built-in logging module with INFO level logging. Logs include:
- Server start-up information
- Model accuracy metrics
- Database operations (when in debug mode)

## Security Considerations
- SQL injection prevention through parameterized queries
- Input validation for symptom data
- Error handling for file operations and database connections

## Performance
- The model achieves approximately 98% accuracy 
- Symptoms are processed in real-time
- Database operations are optimized with proper indexing


