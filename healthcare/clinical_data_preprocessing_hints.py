"""
Healthcare Data Preprocessing Hints
====================================
This file provides guidance for handling clinical data quality issues.
HIPAA Note: All data has been de-identified. No real patient information is present.
"""

import pandas as pd
import numpy as np

# HINT 1: Handling Missing Clinical Values
# =========================================
# Healthcare data often has missing values marked as '?'
# These are NOT random - they often indicate emergency admissions

def handle_missing_values(encounters_df):
    """
    Handle missing values in clinical data intelligently.
    Missing values often have clinical meaning!
    """
    # Identify columns with '?' values
    for col in encounters_df.columns:
        if encounters_df[col].dtype == 'object':
            encounters_df[col] = encounters_df[col].replace('?', np.nan)
    
    # Weight is often missing in emergency cases
    # Consider creating a 'weight_missing' flag instead of imputing
    encounters_df['weight_missing'] = encounters_df['weight'].isna()
    
    # For some fields, missing might mean "not tested" which is clinically relevant
    encounters_df['payer_code_missing'] = encounters_df['payer_code'].isna()
    
    return encounters_df


# HINT 2: Linking Patient Data Across Tables
# ===========================================
# patient_encounters uses 'patient_nbr'
# patient_medication_feedback uses 'Patient ID'
# These MAY or MAY NOT be the same!

def link_patient_data(encounters_df, medication_df):
    """
    Attempt to link patient data across tables.
    You may need to try different joining strategies.
    """
    # First, check if IDs match directly
    encounters_patients = set(encounters_df['patient_nbr'].unique())
    medication_patients = set(medication_df['Patient ID'].unique())
    
    overlap = encounters_patients.intersection(medication_patients)
    print(f"Direct ID matches: {len(overlap)}")
    
    if len(overlap) == 0:
        print("WARNING: No direct patient ID matches!")
        print("Consider:")
        print("1. IDs might be encoded differently")
        print("2. These might be different patient cohorts")
        print("3. You may need to work with each dataset separately")
    
    return overlap


# HINT 3: Processing Retinal Images and Labels
# =============================================
# Images are in retinal_scan_images/
# Labels are in retinal_labels.csv

def prepare_retinal_data(labels_path='retinal_labels.csv', 
                        image_dir='retinal_scan_images'):
    """
    Match retinal images with their diagnosis labels.
    """
    import os
    
    # Load labels
    labels_df = pd.read_csv(labels_path)
    
    # Check if image files exist
    labels_df['image_path'] = labels_df['id_code'].apply(
        lambda x: f'{image_dir}/{x}.png'
    )
    labels_df['image_exists'] = labels_df['image_path'].apply(os.path.exists)
    
    print(f"Images with labels: {labels_df['image_exists'].sum()}/{len(labels_df)}")
    
    # Create train/val split for CNN
    from sklearn.model_selection import train_test_split
    
    train_df, val_df = train_test_split(
        labels_df[labels_df['image_exists']], 
        test_size=0.2, 
        stratify=labels_df[labels_df['image_exists']]['diagnosis'],
        random_state=42
    )
    
    return train_df, val_df


# HINT 4: Encoding Diagnostic Codes
# ==================================
# diag_1, diag_2, diag_3 contain ICD-9 codes
# These need special handling

def process_diagnostic_codes(encounters_df):
    """
    Process ICD-9 diagnostic codes into useful categories.
    """
    # Common diabetes-related ICD-9 codes
    diabetes_codes = ['250', '249']  # 250.xx = diabetes, 249.xx = secondary diabetes
    
    # Check if primary diagnosis is diabetes
    encounters_df['primary_diag_diabetes'] = encounters_df['diag_1'].apply(
        lambda x: str(x).startswith(tuple(diabetes_codes)) if pd.notna(x) else False
    )
    
    # Count diabetes-related diagnoses
    diag_cols = ['diag_1', 'diag_2', 'diag_3']
    encounters_df['diabetes_diag_count'] = 0
    
    for col in diag_cols:
        encounters_df['diabetes_diag_count'] += encounters_df[col].apply(
            lambda x: 1 if str(x).startswith(tuple(diabetes_codes)) else 0
        )
    
    # Group diagnoses into major categories
    def categorize_diagnosis(code):
        """Map ICD-9 codes to major disease categories."""
        code = str(code)
        if code.startswith('250'): return 'diabetes'
        elif code.startswith(('390', '459')): return 'circulatory'
        elif code.startswith(('460', '519')): return 'respiratory'
        elif code.startswith(('520', '579')): return 'digestive'
        elif code.startswith(('800', '999')): return 'injury'
        else: return 'other'
    
    encounters_df['primary_diag_category'] = encounters_df['diag_1'].apply(categorize_diagnosis)
    
    return encounters_df


# HINT 5: Creating Readmission Target Variables
# ==============================================
# The 'readmitted' column has three values: <30, >30, NO

def create_readmission_targets(encounters_df):
    """
    Create binary classification targets for readmission prediction.
    """
    # For 30-day readmission (required model)
    encounters_df['readmitted_30days'] = (
        encounters_df['readmitted'] == '<30'
    ).astype(int)
    
    # For any readmission (alternative target)
    encounters_df['readmitted_any'] = (
        encounters_df['readmitted'] != 'NO'
    ).astype(int)
    
    # Check class balance
    print("30-day readmission rate:", encounters_df['readmitted_30days'].mean())
    print("Any readmission rate:", encounters_df['readmitted_any'].mean())
    
    if encounters_df['readmitted_30days'].mean() < 0.1:
        print("WARNING: Severe class imbalance in 30-day readmission!")
        print("Consider: SMOTE, class weights, or focusing on high-risk patients")
    
    return encounters_df


# HINT 6: Processing Medication Data
# ===================================
# 23 medication columns with values: No, Steady, Up, Down

def process_medications(encounters_df):
    """
    Convert medication information into useful features.
    """
    # List all medication columns
    med_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
                'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
                'miglitol', 'troglitazone', 'tolazamide', 'insulin',
                'glyburide-metformin', 'glipizide-metformin']
    
    # Count total medications
    encounters_df['num_medications_prescribed'] = 0
    for med in med_cols:
        if med in encounters_df.columns:
            encounters_df['num_medications_prescribed'] += (
                encounters_df[med] != 'No'
            ).astype(int)
    
    # Check if insulin was prescribed
    if 'insulin' in encounters_df.columns:
        encounters_df['insulin_prescribed'] = (
            encounters_df['insulin'] != 'No'
        ).astype(int)
    
    # Identify medication changes
    encounters_df['med_changes'] = 0
    for med in med_cols:
        if med in encounters_df.columns:
            encounters_df['med_changes'] += encounters_df[med].apply(
                lambda x: 1 if x in ['Up', 'Down'] else 0
            )
    
    return encounters_df


# HINT 7: Feature Engineering for Clinical Models
# ================================================

def engineer_clinical_features(encounters_df):
    """
    Create clinically meaningful features for prediction models.
    """
    # Length of stay categories
    encounters_df['los_category'] = pd.cut(
        encounters_df['time_in_hospital'],
        bins=[0, 2, 5, 10, 100],
        labels=['short', 'medium', 'long', 'very_long']
    )
    
    # Emergency admission flag
    encounters_df['emergency_admission'] = (
        encounters_df['admission_type_id'] == 1
    ).astype(int)
    
    # High healthcare utilization
    encounters_df['high_utilization'] = (
        (encounters_df['number_inpatient'] > 2) |
        (encounters_df['number_emergency'] > 2)
    ).astype(int)
    
    # Age group processing (they're given as ranges like '[70-80)')
    age_mapping = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25,
        '[30-40)': 35, '[40-50)': 45, '[50-60)': 55,
        '[60-70)': 65, '[70-80)': 75, '[80-90)': 85,
        '[90-100)': 95
    }
    encounters_df['age_numeric'] = encounters_df['age'].map(age_mapping)
    
    # Complexity score
    encounters_df['complexity_score'] = (
        encounters_df['num_procedures'] + 
        encounters_df['num_medications'] + 
        encounters_df['number_diagnoses']
    )
    
    return encounters_df


# Example Usage
# =============
if __name__ == "__main__":
    # Load your data
    encounters = pd.read_csv('patient_encounters_2023.csv')
    medications = pd.read_csv('patient_medication_feedback.csv')
    
    # Apply preprocessing
    encounters = handle_missing_values(encounters)
    encounters = process_diagnostic_codes(encounters)
    encounters = create_readmission_targets(encounters)
    encounters = process_medications(encounters)
    encounters = engineer_clinical_features(encounters)
    
    # Check patient linking
    linked_patients = link_patient_data(encounters, medications)
    
    # Prepare retinal data
    train_retinal, val_retinal = prepare_retinal_data()
    
    # Save preprocessed data
    encounters.to_csv('encounters_preprocessed.csv', index=False)
    
    print("\nPreprocessing complete!")
    print(f"Encounters: {len(encounters)}")
    print(f"30-day readmission rate: {encounters['readmitted_30days'].mean():.2%}")
    print(f"Retinal images ready for CNN: {len(train_retinal)}")
    
    # Important checks
    print("\nData Quality Checks:")
    print(f"Missing weight: {encounters['weight_missing'].mean():.1%}")
    print(f"Emergency admissions: {encounters['emergency_admission'].mean():.1%}")
    print(f"Patients on insulin: {encounters['insulin_prescribed'].mean():.1%}")