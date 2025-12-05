# Support Tickets ETL Pipeline
## Complete Workflow Documentation

---

**Project Name:** support-tickets-etl-pipeline  
**Version:** 2.0  
**Date:** December 5, 2025  

---

## 📁 Project Structure

```
Pipeline-Project/
│
├── etl/                        # ETL Modules Directory
│   ├── config/
│   │   └── config.yaml         # ETL pipeline configuration
│   └── scripts/
│       ├── cleaning.py         # Data cleaning module (advanced imputation)
│       ├── transformation.py   # Data transformation module
│       └── validation.py       # Data validation module
│
├── data/
│   ├── raw/                    # Raw archived data with timestamps
│   ├── processed/              # Cleaned and processed Parquet files
│   └── reports/                # ETL execution and validation reports
│
├── dashboard/
│   └── app.py                  # Streamlit analytics dashboard (3 pages)
│
├── logs/                       # ETL execution logs
├── source/                     # Source CSV files location
├── diagrams/                   # Architecture diagrams
│
├── pipeline.py                 # Main ETL orchestrator
├── demo_for_mentor.py          # Complete verification demo
├── cleanup_data.py             # Data cleanup utility
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── workflow_documentation.md   # This documentation
```

---

## 🔧 Core Files Explanation

### **Main Pipeline Orchestrator**
- **`pipeline.py`** - Central controller that coordinates all ETL processes
  - Loads configuration from YAML file
  - Executes cleaning, transformation, and validation in sequence
  - Handles error logging and data archival
  - Automatically saves processed data with timestamps

### **ETL Scripts Directory (`etl/scripts/`)**

#### **Data Cleaning (`cleaning.py`)**
**Purpose:** Advanced data quality improvement and missing value handling
- **Mode Imputation:** Fills categorical fields with most frequent values
- **Forward Fill:** Uses customer patterns for intelligent data completion
- **Similarity-Based:** Matches text fields using content similarity
- **Regression-Based:** Predicts missing values using business logic relationships
- **Invalid Data Removal:** Eliminates impossible dates and negative IDs
- **Duplicate Detection:** Identifies and removes redundant records

#### **Data Transformation (`transformation.py`)**
**Purpose:** Data structure optimization and feature engineering
- **Date Processing:** Standardizes timestamp formats and creates date ranges
- **Text Normalization:** Cleans and standardizes text fields
- **Category Mapping:** Maps inconsistent categories to standard values
- **Derived Features:** Creates age_days, resolution_days, SLA metrics
- **Data Type Conversion:** Ensures proper data types for analysis
- **Business Logic:** Applies domain-specific transformation rules

#### **Data Validation (`validation.py`)**
**Purpose:** Quality assurance and data integrity verification
- **Schema Validation:** Verifies required columns and data types
- **Range Checking:** Validates date ranges and numeric boundaries
- **Relationship Validation:** Ensures logical data relationships
- **Quality Scoring:** Assigns quality scores based on completeness
- **Anomaly Detection:** Identifies outliers and suspicious patterns
- **Compliance Verification:** Checks data meets business requirements

### **Dashboard Application**
- **`dashboard/app.py`** - Interactive Streamlit analytics platform
  - **3-Page Structure:** Dataset Overview, Ticket Insights, SLA Performance
  - **Real-time Loading:** Automatically detects latest processed files
  - **Interactive Filtering:** Priority, status, date range selections
  - **Professional Visualizations:** Plotly charts and metrics
  - **Mobile Responsive:** Optimized for different screen sizes

### **Configuration Management**
- **`etl/config/config.yaml`** - Centralized configuration file
  - Data paths and file naming conventions
  - Processing parameters and validation rules
  - Imputation strategies and business logic settings
  - Logging levels and output formats

### **Demonstration Tools**
- **`demo_for_mentor.py`** - Professional presentation script
  - Complete verification matrix for all status-priority combinations
  - Exact record counts for academic verification
  - Data quality metrics and processing statistics
  - Professional formatting for mentor demonstrations

---

## Appendix: Sample Data Format

### Input CSV Structure
```csv
Ticket ID,Customer Name,Customer Email,Customer Age,Customer Gender,Product Purchased,Date of Purchase,Ticket Type,Ticket Subject,Ticket Description,Ticket Status,Resolution,Ticket Priority,Ticket Channel,First Response Time,Time to Resolution,Customer Satisfaction Rating
1,John Smith,john.smith@example.com,32,Male,Laptop Pro,2021-03-22,Technical issue,Setup problem,"Having trouble setting up the device",Open,,High,Email,2.5,24.0,4.2
2,Sarah Johnson,sarah.j@example.com,28,Female,Tablet Max,2021-03-23,Billing,Payment issue,"Charge appeared twice on my card",Closed,Refund processed,Medium,Phone,1.0,12.5,4.8
```

### Output Parquet Schema
```
Ticket ID: int64
Customer Name: object
Customer Email: object
Customer Age: float64
Customer Gender: object
Product Purchased: object
Date of Purchase: datetime64[ns]
Ticket Type: object
Ticket Subject: object
Ticket Description: object
Ticket Status: object
Resolution: object
Ticket Priority: object
Ticket Channel: object
First Response Time: float64
Time to Resolution: float64
Customer Satisfaction Rating: float64
purchase_year: int64
purchase_month: int64
purchase_day_of_week: object
resolution_time_category: object
days_since_purchase: int64
priority_score: int64
is_closed: bool
is_open: bool
description_length: int64
description_word_count: float64
subject_length: int64
```
```
pandas==2.2.2
pyyaml==6.0.2
streamlit==1.36.0
pyarrow==16.1.0
numpy==1.26.4
python-dateutil==2.9.0
tqdm==4.66.4
typing_extensions==4.12.2
matplotlib==3.9.0
plotly==5.22.0
```

---

## Appendices

### Sample Data Format

#### Input CSV Structure
```csv
Ticket ID,Customer Name,Customer Email,Customer Age,Customer Gender,Product Purchased,Date of Purchase,Ticket Type,Ticket Subject,Ticket Description,Ticket Status,Resolution,Ticket Priority,Ticket Channel,First Response Time,Time to Resolution,Customer Satisfaction Rating
1,John Smith,john.smith@example.com,32,Male,Laptop Pro,2021-03-22,Technical issue,Setup problem,"Having trouble setting up the device",Open,,High,Email,2.5,24.0,4.2
2,Sarah Johnson,sarah.j@example.com,28,Female,Tablet Max,2021-03-23,Billing,Payment issue,"Charge appeared twice on my card",Closed,Refund processed,Medium,Phone,1.0,12.5,4.8
```

#### Output Parquet Schema
```
Ticket ID: int64
Customer Name: object
Customer Email: object
Customer Age: float64
Customer Gender: object
Product Purchased: object
Date of Purchase: datetime64[ns]
Ticket Type: object
Ticket Subject: object
Ticket Description: object
Ticket Status: object
Resolution: object
Ticket Priority: object
Ticket Channel: object
First Response Time: float64
Time to Resolution: float64
Customer Satisfaction Rating: float64
purchase_year: int64
purchase_month: int64
purchase_day_of_week: object
resolution_time_category: object
days_since_purchase: int64
priority_score: int64
is_closed: bool
is_open: bool
description_length: int64
description_word_count: float64
subject_length: int64
```

