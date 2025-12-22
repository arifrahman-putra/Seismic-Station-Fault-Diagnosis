# Seismic Station Deep Learning-based Fault Diagnosis System

A real-time and historical deep learning-driven fault diagnosis system for seismic monitoring stations. 

## 🎯 Overview

This system automatically detects, isolates, and identifies equipment faults and severities in seismometer networks using PPSD (Probabilistic Power Spectral Density) analysis of hourly seismic waveform data and deep learning predictive maintenance models. It processes data in both **historical** and **real-time** modes, enabling continuous monitoring and maintenance planning for seismometer networks.

### Key Features

- **Three-stage fault diagnosis pipeline**: Detection → Isolation → Identification
- **Earthquake-aware processing**: Automatically filters earthquake signals to prevent false fault detection
- **Multi-component diagnosis**: Separate health monitoring for seismometer, digitizer, and transmission systems
- **Dual operation modes**: Historical data processing and real-time monitoring
- **Automated health scoring (fault identification)**: Daily health reports (0-100%) for each component
- **Database-driven**: PostgreSQL integration for persistent storage and progress tracking

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SEISMIC DATA ACQUISITION                      │
│              (ObsPy SDS Client - Hourly Segments)                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   EARTHQUAKE DETECTION & REMOVAL                 │
│         (EQTransformer + GPD - Filter seismic events)            │
└────────────────────────┬────────────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              │   Earthquake?       │
              └──────────┬──────────┘
                         │
         ┌───────────────┼───────────────┐
         │ YES           │ NO            │
         ▼               ▼               │
    Mark as NORMAL   Feature Extraction │
    (Earthquake      (PPSD Analysis -   │
     occurrence)      NHNM Deviation)   │
                         │               │
                         ▼               │
                  ┌──────────────────┐  │
                  │  FAULT DETECTION │  │
                  │ One-Class AE     │  │
                  │ (Binary: 1/0)    │  │
                  └────────┬─────────┘  │
                           │             │
                  ┌────────┴────────┐   │
                  │   Anomaly?      │   │
                  └────────┬────────┘   │
                           │             │
              ┌────────────┼─────────┐  │
              │ NO         │ YES     │  │
              ▼            ▼         │  │
         Mark as NORMAL  FAULT       │  │
                        ISOLATION    │  │
                        (DNN Classifier) │
                        Multi-class   │  │
                        (LM/FO/EI/DS/ │  │
                         TO/LF/UD/    │  │
                         D_DG/AI)     │  │
                            │          │  │
                            ▼          │  │
                  ┌──────────────────┐│  │
                  │ FAULT CLASSIFICATION│ │
                  │ - Seismometer    ││  │
                  │ - Digitizer      ││  │
                  │ - Transmission   ││  │
                  └────────┬─────────┘│  │
                           │          │  │
                           ▼          ▼  ▼
         ┌───────────────────────────────────────────┐
         │      HOURLY DIAGNOSIS STORAGE              │
         │  (seismometer/digitizer/transmission       │
         │   _diagnosis_hourly tables)                │
         └────────────────┬──────────────────────────┘
                          │
                          ▼
         ┌───────────────────────────────────────────┐
         │    DAILY HEALTH SCORE CALCULATION          │
         │  (% of normal hours per 24-hour period)    │
         │  - Seismometer Health Score (0-100%)       │
         │  - Digitizer Health Score (0-100%)         │
         │  - Transmission Health Score (0-100%)      │
         └────────────────┬──────────────────────────┘
                          │
                          ▼
         ┌───────────────────────────────────────────┐
         │      DAILY DIAGNOSIS STORAGE               │
         │  (seismometer/digitizer/transmission       │
         │   _diagnosis_daily tables)                 │
         └───────────────────────────────────────────┘
```


**Note:** This production implementation uses One-Class Autoencoder (1-AE) for fault detection. 
The research paper evaluated multiple models (LOF, 1-SVM, 1-AE, IF); 1-AE was selected for 
deployment due to its deep learning consistency with the DNN fault isolation model.


## 🧬 Fault Classification

The system categorizes faults into three main equipment types:

| Equipment Type | Fault Classes | Description |
|----------------|---------------|-------------|
| **Seismometer** | LM, FO, EI, DS, TO, LF | Hardware sensor faults |
| **Digitizer** | UD, D_DG | Data acquisition system faults |
| **Transmission** | AI (Availability Issues) | Data transmission/processing failures |

### Fault Class Definitions

- **LM**: Locked Mass (seismometer static mass positioning issue)
- **FO**: Free Oscillation (seismometer automatic mass re-centering force issue)
- **EI**: Electrical Issues (internal electrical circuit defects)
- **DS**: Dead Sensor (sensor failure)
- **TO**: Tilted Offset (sensor mass orientation issue)
- **LF**: Low Frequency issue
- **UD**: Undefined digitizer fault issue (under investigation)
- **D_DG**: Digitizer-specific Data Gap fault
- **AI**: Availability Issue (network, storage, processing failures)

**Note:** 
- AI (Availability Issue) is not isolated by the DNN classifier, but rather identified through data processing or extraction failures.
- Fault classes have been refined based on BMKG operational feedback and differ from the research paper's taxonomy (see "Research vs. Production Implementation" section below).

## 📁 Directory Structure

```
seismometer-fault-diagnosis/
│
├── Seismic Station Realtime Fault Diagnosis.py                      # Main execution script
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
│
├── saved_models/                  # Pre-trained ML models (not included in repo)
│   ├── autoencoder_model.h5       # One-Class Autoencoder for fault detection
│   ├── one_ae_scaler.pkl          # Scaler for One-Class AE
│   ├── dnn_classifier_model.h5    # DNN model for fault isolation
│   ├── dnn_scaler.pkl             # Scaler for DNN classifier
│   └── label_encoder.pkl          # Label encoder for fault classes
│
├── Seismic Folder/                # SDS-formatted seismic waveform data (not included)
│   └── [year]/                    # Organized by year/network/station/channel
│       └── [network]/
│           └── [station]/
│               └── [channel].D/
│                   └── [files]
│
└── Inventory Folder/              # Station metadata XML files (not included)
    ├── STATION1.xml               # StationXML format (ObsPy compatible)
    ├── STATION2.xml
    └── ...
```

### Directory Details

#### `saved_models/`
Contains pre-trained machine learning models:
- **One-Class Autoencoder**: one-class anomaly detection (normal vs. fault)
- **DNN Classifier**: Multi-class fault isolation (8 fault types, AI not included)
- **Scalers & Encoders**: Data preprocessing artifacts

> ⚠️ **Note**: Datasets and Model development scripts are NOT included in this repository due to size. Train your own models or contact the maintainer.

#### `Seismic Folder/`
Expected structure follows the **SeisComP Data Structure (SDS)** format:
```
YEAR/NETWORK/STATION/CHANNEL/NETWORK.STATION.LOC.CHANNEL.YEAR.JULDAY
```

Example:
```
2025/XYZ/STAT01/BHZ.D/XYZ.STAT01..BHZ.D.2025.001
```

#### `Inventory Folder/`
Contains **StationXML** files for each seismic station. Required for:
- Instrument response removal
- PPSD calculation
- Metadata association

File naming convention: `{StationID}.xml`

## 🗄️ Database Schema

The system uses **PostgreSQL** with the following tables:

### Equipment Tracking
```sql
equipments (
    equipmentid TEXT PRIMARY KEY,
    stationid TEXT,
    name TEXT,           ==> 'Seismometer', 'Battery', 'Accelerometer'
    status TEXT,         ==> 'active' or 'inactive'
    channel TEXT         ==> 'B' (broadband) or 'S' (short-period)
)
```

**Note:** While the database tracks multiple equipment types (seismometer, accelerometer, battery, etc.), 
this fault diagnosis system specifically processes seismometer data only. The three-component diagnosis 
(seismometer/digitizer/transmission) uses seismometer waveform recordings as the primary data source.

### Hourly Diagnosis Tables (3 tables)
```sql
-- seismometer_diagnosis_hourly
-- digitizer_diagnosis_hourly  
-- transmission_diagnosis_hourly

(
    TimeStamp TIMESTAMP PRIMARY KEY,
    TimeIndex TIMESTAMP,
    StationID TEXT,
    EquipmentID TEXT,
    Channel TEXT,
    [Component]_HealthStatus INTEGER,    ==> 0=Faulty, 1=Normal
    Diagnosis TEXT,                      ==> Fault class or "Normal"
    [Component]_Description TEXT         ==> Faulty component description  
)
```

### Daily Diagnosis Tables (3 tables)
```sql
-- seismometer_diagnosis_daily
-- digitizer_diagnosis_daily
-- transmission_diagnosis_daily

(
    TimeStamp TIMESTAMP PRIMARY KEY,
    DayIndex DATE,
    StationID TEXT,
    EquipmentID TEXT,
    [Component]_HS INTEGER,         ==> Health Score 0-100%
    Prognosis TEXT,
    Equipment_HS INTEGER
)
```

### Progress Tracking
```sql
FaultDiagnosis_log (
    id SERIAL PRIMARY KEY,
    last_processed_time TIMESTAMP,
    mode TEXT,                      ==> 'historical' or 'realtime'
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8+
- PostgreSQL 12+
- CUDA-capable GPU (optional, for faster processing)

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/seismic-station-fault-diagnosis.git
cd seismic-station-fault-diagnosis
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Database Setup

1. Create PostgreSQL database:
```sql
CREATE DATABASE seismometer_monitoring;
```

2. Update database credentials in `script.py` (lines 318-323):
```python
db_params = {
    "dbname": "seismometer_monitoring",
    "user": "your_username",
    "password": "your_password",
    "host": "localhost",
    "port": "5432"
}
```

3. Populate `equipments` table:
```sql
INSERT INTO equipments (equipmentid, stationid, name, status, channel)
VALUES 
    ('EQ001', 'STAT01', 'Seismometer', 'active', 'B'),
    ('EQ002', 'STAT02', 'Seismometer', 'active', 'S');
```

### Step 4: Prepare Data Directories

1. Create directory structure:
```bash
mkdir -p "saved_models" "Seismic Folder" "Inventory Folder"
```

2. Place your trained models in `saved_models/`:
   - `autoencoder_model.h5`
   - `one_ae_scaler.pkl`
   - `dnn_classifier_model.h5`
   - `dnn_scaler.pkl`
   - `label_encoder.pkl`

3. Organize seismic data in SDS format in `Seismic Folder/`

4. Place StationXML files in `Inventory Folder/` (e.g., `STAT01.xml`)

### Step 5: Configuration

Update the network code in `script.py` (line 557):
```python
Network_Code = "YOUR_NETWORK_CODE"  # Replace with your seismic network code
```

## 💻 Usage

### Running the System

```bash
python "Seismic Station Realtime Fault Diagnosis.py"
```

### Operation Modes

#### 1. **Historical Mode**
- Processes data from **2025-01-01 00:00:00** (or last checkpoint) to present
- Automatically switches to real-time when within 2 hours of current time
- Generates hourly and daily diagnoses
- Progress saved in `FaultDiagnosis_log` table

#### 2. **Real-Time Mode**
- Activates when caught up to current time
- Processes previous hour's data at **30 minutes past each hour**
- Example: At 14:30, processes 13:00-14:00 data
- Generates daily reports at **23:00 each day**

### Resume Processing

The system automatically resumes from the last processed timestamp stored in the database. To restart from scratch:

```sql
DELETE FROM FaultDiagnosis_log;
```

### Monitor Progress

Check processing status:
```sql
SELECT * FROM FaultDiagnosis_log ORDER BY updated_at DESC LIMIT 1;
```

View recent diagnoses:
```sql
-- Hourly diagnosis
SELECT * FROM seismometer_diagnosis_hourly 
ORDER BY TimeIndex DESC LIMIT 10;

-- Daily health scores
SELECT * FROM seismometer_diagnosis_daily 
ORDER BY DayIndex DESC LIMIT 10;
```

## 🔬 Technical Details

### Feature Extraction

The system calculates **NHNM (New High Noise Model) deviation** at four critical periods:

| Period (s) | NHNM Value (dB) | Associated Feature |
|------------|-----------------|--------------------|
| 4 | -97.5958 | H_4                |
| 19 | -135.074 | H_19               |
| 100 | -131.5 | H_100              |
| 325 | -126.395 | H_325              |

Formula: `H_period = NHNM_value - PPSD_value`

### Model Pipeline

1. **Earthquake Detection** (threshold: 0.55)
   - EQTransformer (detection probability)
   - GPD (P-wave and S-wave picks)
   - If earthquake detected → Mark as normal (prevent false positives)

2. **Fault Detection** (One-Class Autoencoder)
   - Input: 4 NHNM deviation features
   - Threshold: 0.001714 (95th percentile reconstruction error)
   - Output: Normal (1) or Anomaly (0)

3. **Fault Isolation** (DNN Classifier)
   - Triggered only if One-Class AE detects anomaly
   - Input: 4 NHNM deviation features
   - Output: Fault class (LM/FO/EI/DS/TO/LF/UD/D_DG)
   - Note: AI (Availability Issue) is assigned during data processing failures, not by the DNN

### Daily Health Score Calculation

```
Health Score (%) = (Number of Normal Hours / Total Hours) × 100
```

- **Expected hours per day**: 72 (24 hours × 3 channels)
- Calculated separately for seismometer, digitizer, and transmission
- Stored at end of each day (23:00 hour)

## 🛠️ Customization

### Modify Processing Start Date

Change line in `determine_start_time()`:
```python
return UTCDateTime(2025, 1, 1, 0, 0, 0)  # Change this date
```

### Adjust Real-time Switch Threshold

Modify `is_realtime_ready()` function (default: 2 hours):
```python
return time_diff <= 7200  # Change threshold in seconds
```

### Update Fault Classifications

Modify fault lists (lines 12-14):
```python
seismo_fault_cases = ["LM", "FO", "EI", "DS", "TO", "LF"]
digitizer_fault_cases = ["UD", "D_DG"]
transmission_fault_cases = ["AI"]
```

## 📝 Requirements

### Python Packages

```
pandas
obspy
seisbench
joblib
psycopg2-binary
numpy
tensorflow>=2.8.0  (or keras)
torch>=1.10.0
```

### Hardware Recommendations

- **CPU**: Multi-core processor (8+ cores recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: NVIDIA GPU with CUDA support (optional, speeds up EQTransformer/GPD, DNN and One Class Autoencoder inference)
- **Storage**: Depends on data volume (estimate ~100GB per year per station)

## 🐛 Troubleshooting

### Common Issues

**1. "No stream data" errors**
- Check SDS folder structure matches ObsPy format
- Verify network/station/channel codes are correct
- Ensure seismic data and inventory files exist for the processed time range

**2. "Error connecting to database"**
- Verify PostgreSQL is running: `sudo systemctl status postgresql`
- Check credentials in `db_params`
- Ensure database exists: `psql -l`

**3. Missing StationXML files**
- Download from FDSN services or contact network operator
- File naming must match StationID exactly

**4. Model files not found**
- Verify `saved_models/` directory contains all `.h5` and `.pkl` files
- Check file paths match the code

### Debug Mode

Add verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 Citation

This system implements the methodology described in:

**Putra, A.Y., Lestari, T., Saputro, A.H.** (2025). "Advancing Seismometer 
Reliability: A Machine Learning-based Fault Diagnosis Framework Using Ambient 
Seismic Spectral Features." *Artificial Intelligence in Geosciences* 
(Under Review).

### Research vs. Production Implementation

**Paper (Research Version):**
- Comparative study of fault detection models: LOF (95.6%), 1-SVM (95.0%), 1-AE (94.0%), IF (93.2%)
- Validated on 7 fault classes: UM, LM, DG, ED, DC, IC, UD
- Academic evaluation and methodology validation

**This Repository (Production Version):**
- Full deep learning implementation using One-Class Autoencoder (1-AE) for deployment consistency
- Updated fault taxonomy based on ongoing BMKG operational feedback: LM, FO, EI, DS, TO, LF, UD, D_DG
- Refined fault definitions aligned with BMKG technician terminology
- Real-time deployment capabilities and database integration

The core methodology (PPSD-based feature extraction, two-phase diagnosis) remains consistent 
between research and production versions. Fault classes and model selection have been refined 
based on operational requirements at BMKG.

If you use this implementation in your research, please cite the paper once published.
## 📄 License

MIT license

## 👤 Author

**Arifrahman Yustika Putra**  
Master's Student, Universitas Indonesia  
Research: machine Learning-based Seismometer Fault Diagnosis 

## 🙏 Acknowledgments

- **BMKG** (Indonesian Agency for Meteorology, Climatology, and Geophysics) for seismic data
- **Universitas Indonesia** for research support
- **ObsPy Development Team** for seismic data processing tools (see https://github.com/obspy/obspy.git)
- **SeisBench Team** for earthquake detection models (see https://github.com/seisbench/seisbench.git)

## 📧 Contact

For questions, issues, or collaboration:
Email: arifrahmanputra01@gmail.com
github: 

---

**Last Updated**: December 2025  

