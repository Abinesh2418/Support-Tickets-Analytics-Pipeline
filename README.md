# Customer Support Tickets ETL Pipeline & Analytics Dashboard

A comprehensive modular ETL pipeline for processing customer support tickets data with an integrated analytics dashboard for data insights and visualization.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Setup Instructions](#-setup-instructions)
- [Usage Guide](#-usage-guide)
- [Features](#-features)
- [Configuration](#-configuration)
- [Analytics Dashboard](#-analytics-dashboard)
- [Data Quality Assurance](#-data-quality-assurance)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Project Overview

This project provides an end-to-end ETL (Extract, Transform, Load) pipeline for processing customer support ticket data. It includes advanced data cleaning, transformation, validation, and a multi-page analytics dashboard built with Streamlit for comprehensive data visualization and insights.

**Key Highlights:**
- Modular ETL architecture with separate cleaning, transformation, and validation modules
- Configuration-driven pipeline using YAML files
- Advanced data imputation techniques (mode, forward fill, similarity-based, regression-based)
- Comprehensive data quality scoring and validation
- Interactive 3-page Streamlit dashboard with real-time analytics
- Automated logging and detailed reporting

---

## 🛠️ Tech Stack

### **Backend & ETL**
- **Python 3.10+** - Core programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing and operations
- **PyYAML** - Configuration file management
- **PyArrow** - High-performance Parquet file format

### **Frontend & Visualization**
- **Streamlit** - Interactive web dashboard framework
- **Plotly** - Professional interactive data visualizations
- **Matplotlib** - Additional statistical plotting

### **Data Storage**
- **Parquet** - Columnar storage format for efficient data processing
- **CSV** - Source data format

### **Development Tools**
- **Python Logging** - Comprehensive logging and monitoring
- **JSON** - Configuration and report file handling
- **Path Management** - Modern pathlib for file operations

---

## 🏗️ Project Structure

```
project-root/
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
├── source/                     # Source data files
│   └── SupportTickets_50k.csv
│
├── pipeline.py                 # Main ETL orchestrator
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore patterns
└── README.md                   # Project documentation
```

---

## 📂 Data Requirements

### **Required CSV Format**

The pipeline expects a CSV file in the `source/` directory with the following columns:

| Column Name | Data Type | Description | Required |
|-------------|-----------|-------------|----------|
| `ticket_id` | Integer | Unique identifier for each ticket | ✅ Yes |
| `subject` | String | Brief description of the issue | ✅ Yes |
| `description` | String | Detailed description of the problem | ✅ Yes |
| `priority` | String | Ticket priority (Low, Medium, High, Critical) | ✅ Yes |
| `status` | String | Current status (Open, In Progress, Resolved, Closed, Pending, Escalated) | ✅ Yes |
| `category` | String | Issue category (e.g., Account Management, Technical Support) | ✅ Yes |
| `origin` | String | Source of ticket (Email, Phone, Web, Chat) | ✅ Yes |
| `customer_id` | Integer | Customer identifier | ✅ Yes |
| `assigned_agent` | String | Agent assigned to the ticket | ⚠️ Optional |
| `created_at` | DateTime | Ticket creation timestamp (ISO 8601 format) | ✅ Yes |
| `resolved_at` | DateTime | Ticket resolution timestamp (ISO 8601 format) | ⚠️ Optional |
| `sla_breached` | Boolean | Whether SLA was breached (True/False) | ✅ Yes |
| `product` | String | Related product or service | ⚠️ Optional |
| `channel` | String | Communication channel used | ⚠️ Optional |
| `severity` | String | Issue severity level | ⚠️ Optional |

### **Data Format Guidelines**

**Date/Time Format:**
- Use ISO 8601 format: `YYYY-MM-DDTHH:MM:SSZ`
- Example: `2023-03-03T19:31:03Z`

**Priority Values:**
- Low, Medium, High, Critical

**Status Values:**
- Open, In Progress, Resolved, Closed, Pending, Escalated

**Boolean Values:**
- True, False (case-sensitive)

**Sample CSV Structure:**
```csv
ticket_id,subject,description,priority,status,category,origin,customer_id,assigned_agent,created_at,resolved_at,sla_breached,product,channel,severity
10000,Payment Failed,User submitted a support request...,Low,Closed,Account Management,Web,74254,Agent123,2023-03-03T19:31:03Z,2023-12-09T19:31:03Z,False,Product C,Email,Trivial
```

**Important Notes:**
- Place your source CSV file in the `source/` directory
- Large CSV files are automatically ignored by Git (see `.gitignore`)
- Missing optional fields will be handled by the cleaning module
- The pipeline performs automatic data validation and quality checks

---

## ⚙️ Setup Instructions

### **Step 1: Prerequisites**
Ensure you have the following installed:
- Python 3.10 or higher
- pip (Python package manager)
- Git (for cloning the repository)

### **Step 2: Clone the Repository**
```bash
git clone https://github.com/yourusername/customer-support-etl-pipeline.git
cd customer-support-etl-pipeline
```

### **Step 3: Create Virtual Environment**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# For Windows:
venv\Scripts\activate

# For macOS/Linux:
source venv/bin/activate
```

### **Step 4: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 5: Verify Installation**
```bash
python --version
pip list
```

---

## 🚀 Usage Guide

### **Step 1: Configure the Pipeline**
Edit the configuration file at `etl/config/config.yaml` to customize pipeline behavior:
- Data source paths
- Cleaning rules
- Transformation settings
- Validation thresholds

### **Step 2: Run the ETL Pipeline**
Execute the main pipeline to process your data:
```bash
python pipeline.py
```

**What happens:**
1. Reads source data from `source/SupportTickets_50k.csv`
2. Archives raw data with timestamps to `data/raw/`
3. Applies data cleaning (missing values, duplicates, normalization)
4. Transforms data (feature engineering, encoding, type conversion)
5. Validates data quality and generates quality scores
6. Saves processed data to `data/processed/`
7. Creates execution and validation reports in `data/reports/`
8. Logs all operations to `logs/`

### **Step 3: Launch Analytics Dashboard**
Start the Streamlit dashboard to visualize insights:
```bash
streamlit run dashboard/app.py
```

The dashboard will automatically open in your browser at `http://localhost:8501`

### **Step 4: Explore the Dashboard**
Navigate through the 3 specialized pages:
1. **Dataset Overview & Data Quality** - Summary statistics and quality metrics
2. **Ticket Insights & Operations** - Priority, status, category analysis
3. **SLA & Performance Analytics** - Performance metrics and time series analysis

---

## ✨ Features

### **ETL Pipeline Features**
✅ **Modular Architecture** - Separate cleaning, transformation, and validation modules  
✅ **Configuration-Driven** - Easy customization via YAML configuration  
✅ **Advanced Data Cleaning** - Sophisticated imputation techniques (mode, forward fill, similarity-based)  
✅ **Feature Engineering** - Creates 22+ derived features for analysis  
✅ **Data Validation** - Comprehensive quality scoring (0-100 scale)  
✅ **Raw Data Archival** - Preserves original data with timestamps  
✅ **Comprehensive Logging** - Detailed execution logs with statistics  
✅ **Error Handling** - Graceful failure handling with detailed reporting  
✅ **Report Generation** - JSON reports with execution summaries

### **Dashboard Features**
📊 **Multi-Page Interface** - 3 specialized analytics views  
📈 **Interactive Filtering** - Dynamic date range, priority, status filters  
🎨 **Professional Visualizations** - Plotly interactive charts and graphs  
⚡ **Real-time Data Loading** - Automatically loads latest processed data  
📥 **Export Capabilities** - Download filtered datasets  
📱 **Responsive Design** - Mobile and desktop optimized  
🔄 **Auto-refresh** - Detects and loads newly processed data

---

## 🔧 Configuration

### **ETL Configuration File** (`etl/config/config.yaml`)

```yaml
# Data source and output paths
data:
  source_file: "source/SupportTickets_50k.csv"
  raw_output_dir: "data/raw"
  processed_output_dir: "data/processed"
  reports_output_dir: "data/reports"

# Cleaning module configuration
cleaning:
  remove_invalid_dates: true
  fill_missing_values:
    assigned_agent: "Unassigned"
    product: "Unknown"
  remove_duplicates: true
  normalize_text: true

# Transformation module configuration  
transformation:
  create_derived_features: true
  convert_data_types: true
  encode_categorical: true
  standardize_columns: true

# Validation module configuration
validation:
  required_fields: ["ticket_id", "subject", "priority", "status"]
  missing_value_threshold: 0.3
  date_validation: true
  outlier_detection: true
  quality_score_threshold: 75.0

# Logging configuration
logging:
  level: "INFO"
  file: "logs/etl_pipeline.log"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

---

## 📊 Analytics Dashboard

### **Page 1: Dataset Overview & Data Quality**
- Total records, columns, and data types summary
- Data quality metrics and missing value analysis
- Column distribution and sample data preview
- Quality indicators with visual health checks

### **Page 2: Ticket Insights & Operations**
- **Priority Distribution** - Critical, High, Medium, Low breakdown
- **Status Analysis** - Open, In Progress, Resolved, Closed trends
- **Category Insights** - Product/service category performance
- **Agent Workload** - Assignment distribution and metrics
- **Origin Analysis** - Email, Phone, Web, Chat channel insights

### **Page 3: SLA & Performance Analytics**
- **SLA Breach Analysis** - Compliance rates and patterns
- **Resolution Time Metrics** - Average times by priority/category
- **Time Series Analysis** - Daily, weekly, monthly trends
- **Business Hours Impact** - Performance during/after hours
- **Performance KPIs** - Real-time calculated metrics

### **Interactive Features**
🔍 **Dynamic Filtering** - Filter by date range, priority, status, category  
📥 **Export Data** - Download filtered datasets as CSV  
🔄 **Auto-load Latest** - Automatically detects new processed data  
📱 **Responsive UI** - Optimized for all screen sizes

---

## 🧪 Data Quality Assurance

### **Validation Framework**
- **Required Field Validation** - Ensures critical business columns exist
- **Data Type Consistency** - Validates and converts appropriate types
- **Missing Value Assessment** - Configurable thresholds for tolerance
- **Duplicate Detection** - Multi-level identification and removal
- **Business Logic Validation** - Date consistency and status flow checks
- **Outlier Detection** - Statistical anomaly identification
- **Quality Scoring** - Comprehensive 0-100 score with detailed metrics

### **Data Cleaning Standards**
- Invalid data handling (impossible dates, corrupted records)
- Configurable missing value strategies per column type
- Text normalization and standardization
- Categorical value consistency and encoding
- Robust date format validation

### **Quality Metrics Tracking**
- Data completeness (missing value percentages)
- Data accuracy (business rule validation rates)
- Data consistency (cross-column validation)
- Processing impact (before/after statistics)

---

## 🚧 Troubleshooting

### **Common Issues**

**1. ModuleNotFoundError: 'etl.scripts'**
```bash
# Ensure you're running from the project root directory
cd /path/to/project-root
python pipeline.py
```

**2. FileNotFoundError: Source file not found**
```bash
# Verify source file exists
dir source\SupportTickets_50k.csv
```

**3. Dashboard Shows "No processed data found"**
```bash
# Run ETL pipeline first to generate processed data
python pipeline.py
```

**4. Streamlit Import Error**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### **Log Analysis**
Check execution logs for detailed information:
- **ETL Logs**: `logs/` directory with timestamped files
- **Processing Statistics**: Row counts, column transformations
- **Error Messages**: Detailed descriptions with stack traces
- **Performance Metrics**: Execution times and quality scores

### **Validation Reports**
Review reports in `data/reports/` for:
- ETL pipeline execution summaries (JSON format)
- Data validation reports with quality scores
- Processing statistics and transformation details

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### **Code Standards**
- Follow PEP 8 style guidelines
- Include comprehensive docstrings
- Add unit tests for new features
- Update documentation accordingly

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📬 Contact

For any queries or suggestions, feel free to reach out:

**📧 Email:** abineshbalasubramaniyam@example.com  
**💼 LinkedIn:** [linkedin.com/in/abinesh-b-1b14a1290/](https://linkedin.com/in/abinesh-b-1b14a1290/)  
**🐙 GitHub:** [github.com/Abinesh2418](https://github.com/yourusername)
