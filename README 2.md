[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/u74dDkVQ)
# 🚀 AI Consulting Capstone Competition

## ⚠️ IMPORTANT: Choose Your Project

**Your team is assigned to compete for ONE contract:**
- 🛍️ **StyleVision** (E-Commerce) - $2.5M contract
- 🏥 **MedInsight** (Healthcare) - $3.5M contract

**Delete the files for the project you're NOT working on!**

---

## 📋 Initial Setup Steps

### 1. Identify Your Project
Check your assignment - are you competing for StyleVision or MedInsight?

### 2. Download YOUR Project's Images

#### For E-Commerce Teams (StyleVision):
- Download `product_images.zip` from Slack #capstone-data
- Extract to `ecommerce/product_images/`
- Delete the healthcare folder: `rm -rf healthcare/`

#### For Healthcare Teams (MedInsight):
- Download `retinal_images.zip` from Slack #capstone-data  
- Extract to `healthcare/retinal_scan_images/`
- Delete the e-commerce folder: `rm -rf ecommerce/`

### 3. Verify Setup
```bash
python setup_data.py
```

### 4. Update This README
Replace this content with your team information and project focus.

---

## 📁 Repository Structure

### For E-Commerce Teams:
```
├── ecommerce/                    # Your project data
│   ├── product_catalog_2025.csv
│   ├── customer_reviews_export.csv
│   ├── product_images/          # Download from Slack
│   └── data_preprocessing_hints.py
├── ecommerce_client_first_meeting.md  # Your client doc
└── setup_data.py
```

### For Healthcare Teams:
```
├── healthcare/                   # Your project data
│   ├── patient_encounters_2023.csv
│   ├── patient_medication_feedback.csv
│   ├── retinal_labels.csv
│   ├── retinal_scan_images/    # Download from Slack
│   └── clinical_data_preprocessing_hints.py
├── healthcare_client_first_meeting.md  # Your client doc
└── setup_data.py
```

---

## 🎯 Your Deliverables

### E-Commerce Teams Must Build:
1. Recommendation System (>60% precision@10)
2. Sentiment Analysis (>80% accuracy)
3. Visual Search CNN (>70% relevance)
4. Innovation Model (your choice)

### Healthcare Teams Must Build:
1. 30-Day Readmission (>75% AUC-ROC)
2. Retinopathy CNN (>85% sensitivity)
3. Medication NLP (>70% accuracy)
4. Clinical Innovation (your choice)

---

## 🏆 Competition Rules

- You compete ONLY against teams in your industry
- E-commerce teams compete for the StyleVision contract
- Healthcare teams compete for the MedInsight contract
- One winner per industry
- Evaluated on hidden test data

---

## 📅 Key Dates

- **Today:** Setup and first commit
- **Aug 21:** Client progress meeting
- **Sep 15/17:** Final presentations
- **Sep 18:** Winners announced

---

**Start by reading your client meeting document and understanding your specific requirements!**
