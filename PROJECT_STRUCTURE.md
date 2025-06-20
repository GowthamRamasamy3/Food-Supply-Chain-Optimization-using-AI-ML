# Project Structure

## Overview
This document describes the complete structure and organization of the Food Supply Optimization Dashboard project.

## Directory Structure
```
food-supply-optimization/
├── README.md                   # Main project documentation
├── QUICK_START.md             # 5-minute setup guide
├── INSTALL.md                 # Detailed installation instructions
├── USAGE.md                   # Comprehensive usage guide
├── DEPLOYMENT.md              # Production deployment guide
├── CONTRIBUTING.md            # Contribution guidelines
├── LICENSE                    # MIT license
├── PROJECT_STRUCTURE.md       # This file
├── pyproject.toml            # Project configuration and dependencies
├── run.py                    # Execution script with dependency checks
├── app.py                    # Main Streamlit application
├── custom_prediction.py      # Custom prediction module with external data
├── report_generator.py       # PDF report generation with date handling
├── test_date_handling.py     # Date processing validation tests
└── generated_reports/        # Output directory for PDF reports (created at runtime)
```

## Core Components

### Main Application (`app.py`)
- **Purpose**: Primary Streamlit dashboard interface
- **Features**: 
  - Multi-page navigation
  - Home overview with sample visualizations
  - Integration with custom prediction module
  - Report generation interface
- **Dependencies**: streamlit, matplotlib, custom modules

### Custom Prediction Module (`custom_prediction.py`)
- **Purpose**: Handles date-specific predictions with external data integration
- **Features**:
  - Real-time weather simulation
  - Holiday detection and impact calculation
  - Climate news integration using web scraping
  - Day-of-week pattern analysis
  - Factor combination and adjustment calculations
- **External APIs**: Uses trafilatura for climate news scraping
- **Dependencies**: requests, trafilatura, numpy, pandas

### Report Generator (`report_generator.py`)
- **Purpose**: Creates comprehensive PDF reports with proper date handling
- **Features**:
  - Date-specific forecast sections
  - Adjustment factor breakdowns
  - Professional PDF layout with page borders
  - Proper date formatting without shifting issues
  - Visual factor impact representations
- **Dependencies**: reportlab, matplotlib

### Testing (`test_date_handling.py`)
- **Purpose**: Validates date processing accuracy
- **Features**:
  - Tests multiple date formats (date, datetime, string)
  - Verifies no date shifting occurs
  - Generates test reports for validation
- **Usage**: Run independently to verify date handling

## Configuration Files

### `pyproject.toml`
- Project metadata and dependencies
- Build system configuration
- Package management settings

### `run.py`
- Automated execution script
- Dependency checking and installation
- Simplified application launch

## Documentation Structure

### User Documentation
- **README.md**: Complete project overview and setup
- **QUICK_START.md**: Minimal setup for immediate use
- **USAGE.md**: Detailed feature explanations and best practices
- **INSTALL.md**: Step-by-step installation procedures

### Developer Documentation
- **CONTRIBUTING.md**: Development guidelines and contribution process
- **DEPLOYMENT.md**: Production deployment options and configurations
- **PROJECT_STRUCTURE.md**: This architectural overview

## Data Flow

### Prediction Process
1. User inputs (center_id, meal_id, date) → `custom_prediction.py`
2. External data fetching (weather, holidays, climate) → Factor calculations
3. Baseline demand retrieval → Adjustment application
4. Results display → Session state storage

### Report Generation
1. Prediction data from session state → `report_generator.py`
2. Date formatting and validation → PDF structure creation
3. Factor visualization → Document assembly
4. File output → Download interface

## Key Design Decisions

### Date Handling
- Preserves exact user-selected dates without timezone shifts
- Supports multiple date formats (date, datetime, string)
- Comprehensive debugging and validation

### External Data Integration
- Modular approach for easy API replacement
- Graceful fallbacks for missing data
- Simulation for demonstration purposes

### Report Layout
- Professional PDF formatting with proper page management
- Visual factor breakdowns for non-technical users
- Consistent styling and branding

### User Interface
- Multi-page Streamlit design for logical flow
- Form-based inputs with validation
- Real-time feedback and progress indicators

## Extension Points

### Easy Customization Areas
- Holiday data sources (replace with real API)
- Weather data integration (add actual weather service)
- Baseline demand calculation (connect to database)
- Report styling and branding
- Additional prediction factors

### Scalability Considerations
- Session state management for multi-user scenarios
- Database integration points
- API rate limiting and caching
- Performance optimization areas

## Dependencies

### Core Runtime
- streamlit (dashboard framework)
- pandas (data manipulation)
- numpy (numerical operations)
- matplotlib (visualization)

### PDF Generation
- reportlab (PDF creation)

### External Data
- requests (HTTP requests)
- trafilatura (web content extraction)

### Development
- Standard Python 3.11+ libraries
- Optional: uv (fast package manager)