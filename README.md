# Customer Support Tickets ETL Pipeline & Analytics Dashboard

A complete modular ETL pipeline for processing customer support tickets data with a separated analytics dashboard for comprehensive data insights and visualization.

## 🏗️ Project Structure

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
├── source/                     # Additional source files
├── diagrams/                   # Architecture diagrams
│
├── pipeline.py                 # Main ETL orchestrator
├── demo_for_mentor.py          # Complete verification demo for presentations
├── cleanup_data.py             # Data cleanup utility
├── requirements.txt            # Python dependencies
└── README.md                   # This documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Required packages: pandas, streamlit, plotly, PyYAML

### Method 1: Complete ETL + Analytics Workflow (Recommended)

1. **Run ETL Pipeline (Command Line)**
   ```bash
   python pipeline.py
   ```
   *Note: Pipeline now uses default config path `etl/config/config.yaml`*

2. **Launch Analytics Dashboard**
   ```bash
   streamlit run dashboard/app.py
   ```

3. **Professional Demonstration (For Presentations)**
   ```bash
   python demo_for_mentor.py
   ```
   *Complete verification demo with all status-priority combinations*

3. **Access Dashboard**: Open http://localhost:8501 in your browser

### Method 2: Development Setup

1. **Create Virtual Environment**
   ```bash
   python -m venv venv
   ```

2. **Activate Virtual Environment**
   ```bash
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install pandas streamlit plotly PyYAML numpy pyarrow
   ```

## 🆕 Recent Improvements (December 2025)

### Advanced Data Processing
- **Sophisticated Imputation Techniques**: Replaced simple "Unknown" filling with:
  - Mode imputation for categorical fields
  - Forward fill with customer pattern analysis
  - Similarity-based imputation for text fields
  - Regression-based imputation using business logic
  - Business rule preservation for specific fields

### Professional Demonstration Tools
- **Comprehensive Verification**: `demo_for_mentor.py` provides complete project demonstration
- **Status-Priority Combinations**: All 18 combinations (6 statuses × 3 priorities) with exact counts
- **Filter Verification**: Professional presentation with exact numbers for mentor validation

### Enhanced Dashboard
- **Improved Date Range**: Full 2020-2026 date range support
- **Professional Error Messaging**: Clean, user-friendly error displays
- **Real-time Data Loading**: Automatic detection of latest processed files

### Streamlined Architecture
- **Configuration-Driven**: Default config path `etl/config/config.yaml`
- **Clean Project Structure**: Removed redundant verification files
- **Production Ready**: Comprehensive logging and error handling

## 📊 Pipeline Execution

### Complete Workflow

```bash
# 1. Execute ETL Pipeline (processes tickets.csv -> cleaned data)
python pipeline.py --config etl/config/config.yaml

# 2. Launch Analytics Dashboard (displays processed data insights)
streamlit run dashboard/app.py
```

### ETL Pipeline Features
- **Modular Architecture**: Separate cleaning, transformation, and validation modules
- **Raw Data Archival**: Timestamps and preserves original data
- **Comprehensive Logging**: Detailed execution logs with statistics
- **Quality Validation**: Data quality scoring and validation reports
- **Error Handling**: Graceful failure handling with detailed reporting

### Dashboard Features
- **3-Page Analytics**: Comprehensive data insights across multiple views
- **Real-time Data Loading**: Automatically loads latest processed data
- **Interactive Filtering**: Dynamic data exploration capabilities
- **Professional Visualizations**: Plotly charts and metrics
- **No ETL Execution**: Pure analytics platform (ETL runs separately)

## 🔧 Configuration

### ETL Configuration (`etl/config/config.yaml`)

The modular pipeline behavior is controlled through the YAML configuration file:

```yaml
# Data source and output paths
data:
  source_file: "tickets.csv"
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

## 📈 Analytics Dashboard

The Streamlit dashboard provides comprehensive analytics in **3 specialized pages**:

### 📊 **Page 1: Dataset Overview & Data Quality**
- **Dataset Summary**: Total records, columns, data types, and file information
- **Data Quality Metrics**: Missing value analysis, data completeness scores
- **Column Analysis**: Distribution of data types, sample data preview
- **Quality Indicators**: Visual indicators of data health and completeness

### 🎫 **Page 2: Ticket Insights & Operations**
- **Priority Distribution**: Visual breakdown of ticket priorities (Critical, High, Medium, Low)
- **Status Analysis**: Current ticket status distribution and trends
- **Category Insights**: Product/service category analysis with filtering
- **Agent Workload**: Assignment distribution and agent performance metrics
- **Origin Analysis**: Ticket source channels (Email, Phone, Web, Chat)

### ⚡ **Page 3: SLA & Performance Analytics**
- **SLA Breach Analysis**: SLA compliance rates and breach patterns
- **Resolution Time Metrics**: Average resolution times by priority and category
- **Time Series Analysis**: Daily, weekly, and monthly performance trends
- **Business Hours Impact**: Performance analysis during business vs after hours
- **Performance KPIs**: Key metrics dashboard with real-time calculations

### 🔍 **Interactive Features**
- **Dynamic Filtering**: Date range, priority, status, and category filters
- **Real-time Updates**: Automatically loads latest processed data
- **Export Capabilities**: Download filtered datasets
- **Responsive Design**: Mobile and desktop optimized
- **Professional Visualizations**: Plotly interactive charts and graphs

## 🔄 Modular Data Processing Pipeline

### 1. Main Orchestrator (`pipeline.py`)
- **Central Coordinator**: Manages entire ETL workflow from root directory
- **Configuration Management**: Loads and validates YAML configuration
- **Raw Data Archival**: Timestamps and preserves original datasets
- **Module Coordination**: Orchestrates cleaning → transformation → validation
- **Comprehensive Logging**: Detailed execution logs with performance metrics
- **Report Generation**: Creates execution reports and validation summaries

### 2. Data Cleaning (`etl/scripts/cleaning.py`)
- **Invalid Data Removal**: Eliminates impossible dates and corrupted records
- **Missing Value Handling**: Configurable strategies for missing data
- **Duplicate Detection**: Identifies and removes duplicate records
- **Text Normalization**: Standardizes text fields and formats
- **Quality Reporting**: Detailed cleaning statistics and impact analysis

### 3. Data Transformation (`etl/scripts/transformation.py`)  
- **Data Type Conversion**: Optimizes column types for performance
- **Feature Engineering**: Creates 22+ derived features including:
  - Date/time components (year, month, day_of_week, hour)
  - Resolution time calculations
  - Business logic flags (weekend, business hours, SLA status)
  - Text analysis features (length, word count)
- **Categorical Encoding**: Label encoding for machine learning readiness
- **Column Standardization**: Consistent naming conventions

### 4. Data Validation (`etl/scripts/validation.py`)
- **Required Field Validation**: Ensures critical columns exist
- **Data Quality Scoring**: Comprehensive quality score (0-100)
- **Business Rule Validation**: Checks date logic, status consistency
- **Outlier Detection**: Identifies anomalous data patterns
- **Detailed Reporting**: JSON validation reports with error details

### 5. Analytics Dashboard (`dashboard/app.py`)
- **Auto Data Loading**: Loads latest processed Parquet files
- **Multi-page Interface**: 3 specialized analytics views
- **Interactive Filtering**: Real-time data exploration
- **Professional Visualizations**: Plotly charts and metrics
- **Export Capabilities**: Download filtered datasets

## 📝 Logging and Monitoring

### Comprehensive Logging System
- **Location**: `logs/` directory with timestamped files
- **Format**: Structured logging with timestamps, module names, and log levels
- **Content**: 
  - ETL pipeline execution progress and statistics
  - Data processing steps with row/column counts
  - Error handling and troubleshooting information
  - Performance metrics and execution times
  - Data quality scores and validation results

### Detailed Reporting
- **ETL Reports**: JSON reports with execution summary (`data/reports/`)
- **Validation Reports**: Detailed data quality analysis and validation results
- **Processing Statistics**: Input/output counts, column transformations
- **Error Documentation**: Comprehensive error logs with resolution suggestions

### Monitoring Capabilities
- **Pipeline Status**: Real-time execution status tracking
- **Data Quality Metrics**: Continuous quality score monitoring
- **Performance Tracking**: Execution time and memory usage
- **Alert System**: Warning logs for data quality issues

## 🧪 Data Quality Assurance

### Advanced Validation Framework
- **Required Field Validation**: Ensures critical business columns exist
- **Data Type Consistency**: Validates and converts appropriate data types
- **Missing Value Assessment**: Configurable thresholds for missing data tolerance
- **Duplicate Detection**: Multi-level duplicate identification and removal
- **Business Logic Validation**: Date consistency, status flow validation
- **Outlier Detection**: Statistical outlier identification
- **Quality Scoring**: Comprehensive 0-100 quality score with detailed metrics

### Data Cleaning Standards
- **Invalid Data Handling**: Removes impossible dates and corrupted records
- **Missing Value Strategies**: Configurable fill strategies per column type
- **Text Normalization**: Standardized text processing and cleanup
- **Categorical Standardization**: Consistent category values and encoding
- **Date Format Validation**: Robust date parsing and validation

### Quality Metrics Tracking
- **Data Completeness**: Missing value percentages per column
- **Data Accuracy**: Business rule validation success rates
- **Data Consistency**: Cross-column validation and relationship checks
- **Processing Impact**: Before/after statistics for all transformations

## 🔒 Best Practices

### Code Quality
- **PEP 8 Compliance**: Clean, readable Python code
- **Type Hints**: Enhanced code documentation
- **Error Handling**: Comprehensive exception management
- **Logging**: Detailed operation tracking
- **Modular Design**: Separation of concerns

### Performance Optimization
- **Efficient Data Loading**: Pandas best practices
- **Memory Management**: Chunked processing for large datasets
- **Caching**: Streamlit data caching for dashboard performance
- **Compression**: Parquet format with Snappy compression

### Security Considerations
- **Path Validation**: Secure file operations
- **Input Sanitization**: Safe data processing
- **Error Disclosure**: Secure error handling

## 🚧 Troubleshooting

### Common Issues

1. **ModuleNotFoundError: 'etl.scripts'**
   ```bash
   # Ensure you're running from the project root directory
   cd d:\September-AI\Pipeline-Project
   python pipeline.py --config etl/config/config.yaml
   ```

2. **FileNotFoundError: tickets.csv**
   ```bash
   # Verify dataset is in project root
   dir tickets.csv
   # Should show the CSV file exists
   ```

3. **Dashboard Shows "No processed data found"**
   ```bash
   # Run ETL pipeline first to generate processed data
   python pipeline.py --config etl/config/config.yaml
   ```

4. **Configuration File Not Found**
   ```bash
   # Check config file path
   dir etl\config\config.yaml
   # Or run with absolute path
   python pipeline.py --config D:\September-AI\Pipeline-Project\etl\config\config.yaml
   ```

5. **Streamlit Import Error**
   ```bash
   # Install required packages
   pip install streamlit plotly pandas PyYAML numpy pyarrow
   ```

### Log Analysis
Check execution logs for detailed information:
- **ETL Logs**: Check `logs/` directory for timestamped execution logs
- **Processing Statistics**: Look for row counts, column transformations
- **Error Messages**: Detailed error descriptions with stack traces
- **Performance Metrics**: Execution times and data quality scores

### Validation Reports
Check `data/reports/` directory for:
- ETL pipeline execution reports (JSON format)
- Data validation reports with quality scores
- Processing statistics and transformation details

## 🔮 Future Enhancements

### Planned Features
- **Database Integration**: PostgreSQL, MySQL, MongoDB support
- **Cloud Storage**: AWS S3, Azure Blob, Google Cloud Storage integration
- **Real-time Processing**: Apache Kafka streaming data pipelines
- **Machine Learning Models**: Automated ticket classification and sentiment analysis
- **Advanced Analytics**: Predictive modeling for resolution times and customer satisfaction
- **API Development**: REST API for dashboard data and ETL triggers
- **Scheduling**: Apache Airflow integration for automated pipeline execution
- **Multi-format Support**: JSON, XML, Excel, and database source connectors

### Scalability Improvements
- **Distributed Processing**: Dask or Ray integration for large datasets
- **Container Support**: Docker containerization with Kubernetes deployment
- **Microservices Architecture**: API-driven modular service components
- **Caching Layer**: Redis integration for dashboard performance
- **Load Balancing**: Multi-instance dashboard deployment
- **Data Versioning**: DVC (Data Version Control) integration
- **Monitoring Dashboard**: Dedicated pipeline monitoring and alerting system

## 📚 Dependencies

### Core ETL Libraries
- **pandas**: Data manipulation, analysis, and CSV/Parquet processing
- **numpy**: Numerical computing and statistical operations
- **PyYAML**: Configuration file parsing and management
- **pyarrow**: High-performance Parquet file format support

### Dashboard Libraries  
- **streamlit**: Interactive web dashboard framework
- **plotly**: Professional interactive data visualizations
- **matplotlib**: Additional statistical plotting capabilities

### Utility Libraries
- **pathlib**: Modern path handling and file operations
- **datetime**: Date/time processing and timezone handling
- **logging**: Comprehensive logging and monitoring
- **json**: Configuration and report file handling

### Installation
```bash
# Core dependencies (minimum required)
pip install pandas numpy PyYAML pyarrow streamlit plotly

# Full installation with all features
pip install pandas numpy PyYAML pyarrow streamlit plotly matplotlib python-dateutil
```

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create feature branch
3. Install development dependencies
4. Make changes with tests
5. Submit pull request

### Code Standards
- Follow PEP 8 style guidelines
- Include comprehensive docstrings
- Add unit tests for new features
- Update documentation

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review log files in `logs/pipeline.log`
3. Open an issue on GitHub
4. Consult the documentation

## 🧹 Data Cleanup (Development & Testing)

### Quick Cleanup (Recommended)
```bash
# Remove processed data and reports (keeps raw data and source)
python cleanup_data.py --quick
```

### Interactive Cleanup  
```bash
# Interactive mode with file type selection
python cleanup_data.py
```

### Complete Cleanup
```bash
# Remove ALL generated files (caution: includes raw archives)
python cleanup_data.py --all
```

### Cleanup Categories
- **Processed Data**: `data/processed/*.parquet` (cleaned and transformed files)
- **Raw Archives**: `data/raw/*.parquet` (timestamped source data copies)  
- **Reports**: `data/reports/*.json` (ETL execution and validation reports)
- **Logs**: `logs/*.log` (pipeline execution logs)
- **Cache Files**: `__pycache__/`, `*.pyc` (Python bytecode cache)

### Cleanup Features
- **Safe Operations**: Confirmation prompts for destructive operations
- **Selective Cleanup**: Choose specific file types to clean
- **Size Reporting**: Shows disk space that will be freed
- **Logging**: All cleanup operations logged for audit trail
- **Recovery Prevention**: Source data (`tickets.csv`) protected by default

---

**Built with ❤️ for modular data engineering excellence**