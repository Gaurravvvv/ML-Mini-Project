from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "app" / "static"

RAW_DATA_FILE = RAW_DATA_DIR / "german.data"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "german_credit_cleaned.csv"

# URLs for UCI German Credit Data
UCI_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
FALLBACK_DATA_URL = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/credit.csv"

# Column names
RAW_COLUMNS = [
    "checking_status",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment",
    "installment_rate",
    "personal_status",
    "other_parties",
    "residence_since",
    "property_magnitude",
    "age",
    "other_payment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "own_telephone",
    "foreign_worker",
    "class",
]

NUMERICAL_FEATURES = [
    "duration",
    "credit_amount",
    "installment_rate",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
]

CATEGORICAL_FEATURES = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment",
    "personal_status",
    "other_parties",
    "property_magnitude",
    "other_payment_plans",
    "housing",
    "job",
    "own_telephone",
    "foreign_worker",
]

TARGET_COLUMN = "default"

# Attribute code decoders for raw UCI dataset (A11-A202)
VALUE_MAPPINGS = {
    "checking_status": {
        "A11": "< 0 DM",
        "A12": "0 - 200 DM",
        "A13": ">= 200 DM",
        "A14": "no checking account",
    },
    "credit_history": {
        "A30": "no credits taken / all paid duly",
        "A31": "all credits at this bank paid duly",
        "A32": "existing credits paid duly till now",
        "A33": "delay in paying off in the past",
        "A34": "critical account / other credits elsewhere",
    },
    "purpose": {
        "A40": "car (new)",
        "A41": "car (used)",
        "A42": "furniture/equipment",
        "A43": "radio/television",
        "A44": "domestic appliances",
        "A45": "repairs",
        "A46": "education",
        "A47": "vacation",
        "A48": "retraining",
        "A49": "business",
        "A410": "others",
    },
    "savings_status": {
        "A61": "< 100 DM",
        "A62": "100 - 500 DM",
        "A63": "500 - 1000 DM",
        "A64": ">= 1000 DM",
        "A65": "unknown / no savings account",
    },
    "employment": {
        "A71": "unemployed",
        "A72": "< 1 year",
        "A73": "1 - 4 years",
        "A74": "4 - 7 years",
        "A75": ">= 7 years",
    },
    "personal_status": {
        "A91": "male : divorced/separated",
        "A92": "female : divorced/separated/married",
        "A93": "male : single",
        "A94": "male : married/widowed",
        "A95": "female : single",
    },
    "other_parties": {
        "A101": "none",
        "A102": "co-applicant",
        "A103": "guarantor",
    },
    "property_magnitude": {
        "A121": "real estate",
        "A122": "building society savings / life insurance",
        "A123": "car or other",
        "A124": "unknown / no property",
    },
    "other_payment_plans": {
        "A141": "bank",
        "A142": "stores",
        "A143": "none",
    },
    "housing": {
        "A151": "rent",
        "A152": "own",
        "A153": "for free",
    },
    "job": {
        "A171": "unemployed / unskilled non-resident",
        "A172": "unskilled resident",
        "A173": "skilled employee / official",
        "A174": "management / self-employed / qualified",
    },
    "own_telephone": {
        "A191": "none",
        "A192": "yes",
    },
    "foreign_worker": {
        "A201": "yes",
        "A202": "no",
    },
}

# Features specifically used for K-Means Credit Risk Profiling
KMEANS_FEATURES = [
    "duration",
    "credit_amount",
    "installment_rate",
    "age",
    "existing_credits",
]

# Random State
RANDOM_STATE = 42
TEST_SIZE = 0.20
