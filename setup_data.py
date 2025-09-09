#!/usr/bin/env python3
"""
AI Consulting Capstone - Data Setup Verification
Run this after downloading image datasets from Slack.
"""

import os
import sys
import pandas as pd
from pathlib import Path

def print_header():
    """Print welcome header."""
    print("="*60)
    print("🚀 AI CONSULTING CAPSTONE - DATA VERIFICATION")
    print("="*60)
    print()

def check_csv_files():
    """Verify all CSV files are present."""
    print("📊 Checking CSV files...")
    
    required_files = {
        'ecommerce': {
            'product_catalog_2025.csv': 11900,  # Expected rows
            'customer_reviews_export.csv': 19964
        },
        'healthcare': {
            'patient_encounters_2023.csv': 91589,
            'patient_medication_feedback.csv': 3728,
            'clinical_codes_reference.csv': 68,
            'retinal_labels.csv': 3222
        }
    }
    
    all_good = True
    
    for folder, files in required_files.items():
        print(f"\n  {folder.upper()}:")
        for filename, expected_rows in files.items():
            filepath = Path(folder) / filename
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath)
                    actual_rows = len(df)
                    if abs(actual_rows - expected_rows) < 10:  # Allow small variance
                        print(f"    ✅ {filename}: {actual_rows:,} rows")
                    else:
                        print(f"    ⚠️  {filename}: {actual_rows:,} rows (expected ~{expected_rows:,})")
                except Exception as e:
                    print(f"    ❌ {filename}: Error reading file")
                    all_good = False
            else:
                print(f"    ❌ {filename}: FILE MISSING!")
                all_good = False
    
    return all_good

def check_images():
    """Check if image datasets are properly downloaded."""
    print("\n🖼️  Checking image datasets...")
    
    ecom_path = Path('ecommerce/product_images')
    health_path = Path('healthcare/retinal_scan_images')
    
    # Detect which project this team is working on
    ecom_exists = Path('ecommerce').exists()
    health_exists = Path('healthcare').exists()
    
    if ecom_exists and not health_exists:
        print("\n  📊 Project: E-COMMERCE (StyleVision)")
        ecom_images = list(ecom_path.glob('*.jpg')) if ecom_path.exists() else []
        ecom_count = len(ecom_images)
        
        if ecom_count == 0:
            print(f"    ❌ No product images found!")
            print(f"       → Download product_images.zip from Slack #capstone-data")
            print(f"       → Extract to ecommerce/product_images/")
            return False
        elif ecom_count < 12000:
            print(f"    ⚠️  {ecom_count:,} images (expected ~12,114)")
            print(f"       → You might have a partial download")
            return False
        else:
            print(f"    ✅ {ecom_count:,} product images ready!")
            return True
    
    elif health_exists and not ecom_exists:
        print("\n  📊 Project: HEALTHCARE (MedInsight)")
        health_images = list(health_path.glob('*.png')) if health_path.exists() else []
        health_count = len(health_images)
        
        if health_count == 0:
            print(f"    ❌ No retinal images found!")
            print(f"       → Download retinal_images.zip from Slack #capstone-data")
            print(f"       → Extract to healthcare/retinal_scan_images/")
            return False
        elif health_count < 3200:
            print(f"    ⚠️  {health_count:,} images (expected ~3,222)")
            print(f"       → You might have a partial download")
            return False
        else:
            print(f"    ✅ {health_count:,} retinal images ready!")
            return True
    
    elif ecom_exists and health_exists:
        print("\n  ⚠️  BOTH projects detected!")
        print("     You should focus on ONE project only.")
        print("     E-commerce teams: rm -rf healthcare/")
        print("     Healthcare teams: rm -rf ecommerce/")
        return False
    
    else:
        print("\n  ❌ No project data found!")
        print("     Please check your repository setup.")
        return False

def check_preprocessing_scripts():
    """Verify preprocessing hint files exist."""
    print("\n🔧 Checking preprocessing scripts...")
    
    scripts = [
        'ecommerce/data_preprocessing_hints.py',
        'healthcare/clinical_data_preprocessing_hints.py'
    ]
    
    all_present = True
    for script in scripts:
        if Path(script).exists():
            print(f"    ✅ {script}")
        else:
            print(f"    ❌ {script} missing!")
            all_present = False
    
    return all_present

def check_meeting_docs():
    """Verify client meeting documents exist."""
    print("\n📄 Checking client meeting documents...")
    
    docs = [
        'ecommerce_client_first_meeting.md',
        'healthcare_client_first_meeting.md'
    ]
    
    all_present = True
    for doc in docs:
        if Path(doc).exists():
            print(f"    ✅ {doc}")
        else:
            print(f"    ❌ {doc} missing!")
            all_present = False
    
    return all_present

def main():
    """Main verification function."""
    print_header()
    
    # Run all checks
    csv_ok = check_csv_files()
    scripts_ok = check_preprocessing_scripts()
    docs_ok = check_meeting_docs()
    images_ok = check_images()
    
    # Summary
    print("\n" + "="*60)
    print("📋 SETUP SUMMARY")
    print("="*60)
    
    if csv_ok and scripts_ok and docs_ok:
        print("✅ Code and data files: READY")
    else:
        print("❌ Code and data files: INCOMPLETE")
    
    if images_ok:
        print("✅ Image datasets: READY")
    else:
        print("⚠️  Image datasets: NEED DOWNLOAD")
    
    print("\n" + "="*60)
    
    if csv_ok and scripts_ok and docs_ok and images_ok:
        print("🎉 ALL SET! Your consulting firm is ready to compete!")
        print("\nNext steps:")
        print("1. Read both client meeting documents")
        print("2. Start exploring the data")
        print("3. Make your first commit by tonight!")
    elif csv_ok and scripts_ok and docs_ok:
        print("⚠️  Almost ready! Just need to download images from Slack.")
        print("\nTo complete setup:")
        print("1. Go to Slack #capstone-data")
        print("2. Download both zip files")
        print("3. Extract to the correct directories")
        print("4. Run this script again")
    else:
        print("❌ Setup incomplete. Check the errors above.")
        print("\nNeed help? Post in Slack or create a GitHub issue.")
    
    print("="*60)

if __name__ == "__main__":
    main()