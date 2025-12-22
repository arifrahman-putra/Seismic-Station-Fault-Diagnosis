import pandas as pd
from obspy import UTCDateTime
from obspy.clients.filesystem.sds import Client
from obspy import read, read_inventory
from obspy.core.stream import Stream
from obspy.signal import PPSD
from seisbench.models import EQTransformer, GPD
import joblib
import psycopg2
import time
import numpy as np
from keras.models import load_model
import torch

# Fault classification lists
seismo_fault_cases = ["LM", "FO", "EI", "DS", "TO", "LF"]
digitizer_fault_cases = ["UD", "D_DG"]
transmission_fault_cases = ["AI"]


class DatabaseManager:
    def __init__(self, db_params):
        """Initialize a persistent database connection."""
        self.db_params = db_params
        self.conn = None
        self.connect()
        self.ensure_tables_exist()  # Ensure tables are created at startup

    def connect(self):
        """Establish a connection to the database."""
        try:
            self.conn = psycopg2.connect(**self.db_params)
            self.conn.autocommit = True  # Ensures commits happen automatically
            print("✅ Database connection established.")
        except Exception as e:
            print(f"❌ Error connecting to database: {e}")
            self.conn = None

    def execute_query(self, query, values=None, fetch=False):
        """Execute an SQL query with optional fetching."""
        if not self.conn:  # If the connection is lost, attempt to reconnect
            self.connect()
            if not self.conn:  # If reconnection fails, exit the function
                return None

        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query, values)
                if fetch:
                    return cursor.fetchall()
            return True  # Query executed successfully
        except psycopg2.Error as e:
            print(f"❌ Database error: {e}")
            return False  # Return False on failure

    def ensure_tables_exist(self):
        """Create tables if they do not exist."""
        seismometer_diagnosis_hourly_query = """
        CREATE TABLE IF NOT EXISTS seismometer_diagnosis_hourly (
            TimeStamp TIMESTAMP PRIMARY KEY,
            TimeIndex TIMESTAMP,
            StationID TEXT,
            EquipmentID TEXT,
            Channel TEXT,
            Seismometer_HealthStatus INTEGER,
            Diagnosis TEXT,
            Seismometer_Description TEXT
        );
        """

        seismometer_diagnosis_daily_query = """
        CREATE TABLE IF NOT EXISTS seismometer_diagnosis_daily (
            TimeStamp TIMESTAMP PRIMARY KEY,
            DayIndex DATE,
            StationID TEXT,
            EquipmentID TEXT,
            Seismometer_HS INTEGER,
            Prognosis TEXT,
            Equipment_HS INTEGER
        );
        """

        digitizer_diagnosis_hourly_query = """
        CREATE TABLE IF NOT EXISTS digitizer_diagnosis_hourly (
            TimeStamp TIMESTAMP PRIMARY KEY,
            TimeIndex TIMESTAMP,
            StationID TEXT,
            EquipmentID TEXT,
            Channel TEXT,
            Digitizer_HealthStatus INTEGER,
            Diagnosis TEXT,
            Digitizer_Description TEXT
        );
        """

        digitizer_diagnosis_daily_query = """
        CREATE TABLE IF NOT EXISTS digitizer_diagnosis_daily (
            TimeStamp TIMESTAMP PRIMARY KEY,
            DayIndex DATE,
            StationID TEXT,
            EquipmentID TEXT,
            Digitizer_HS INTEGER,
            Prognosis TEXT,
            Equipment_HS INTEGER
        );
        """

        transmission_diagnosis_hourly_query = """
        CREATE TABLE IF NOT EXISTS transmission_diagnosis_hourly (
            TimeStamp TIMESTAMP PRIMARY KEY,
            TimeIndex TIMESTAMP,
            StationID TEXT,
            EquipmentID TEXT,
            Channel TEXT,
            Transmission_HealthStatus INTEGER,
            Diagnosis TEXT,
            Transmission_Description TEXT
        );
        """

        transmission_diagnosis_daily_query = """
        CREATE TABLE IF NOT EXISTS transmission_diagnosis_daily (
            TimeStamp TIMESTAMP PRIMARY KEY,
            DayIndex DATE,
            StationID TEXT,
            EquipmentID TEXT,
            Transmission_HS INTEGER,
            Prognosis TEXT,
            Equipment_HS INTEGER
        );
        """

        # Add table for tracking processing progress
        progress_tracking_query = """
        CREATE TABLE IF NOT EXISTS FaultDiagnosis_log (
            id SERIAL PRIMARY KEY,
            last_processed_time TIMESTAMP,
            mode TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        # Execute table creation queries
        self.execute_query(seismometer_diagnosis_hourly_query)
        self.execute_query(seismometer_diagnosis_daily_query)
        self.execute_query(digitizer_diagnosis_hourly_query)
        self.execute_query(digitizer_diagnosis_daily_query)
        self.execute_query(transmission_diagnosis_hourly_query)
        self.execute_query(transmission_diagnosis_daily_query)
        self.execute_query(progress_tracking_query)

        print("✅ Tables verified/created successfully.")

    def get_active_seismometers(self):
        """Fetch all active seismometer stations from the Equipments table."""
        query = """
        SELECT equipmentid, stationid, name, status, channel
        FROM equipments 
        WHERE name = 'Seismometer' AND status = 'active';
        """
        result = self.execute_query(query, fetch=True)
        return result if result else []

    def get_last_processed_time(self):
        """Get the last processed time from the progress tracking table."""
        query = """
        SELECT last_processed_time FROM FaultDiagnosis_log
        ORDER BY updated_at DESC LIMIT 1;
        """
        result = self.execute_query(query, fetch=True)
        if result and result[0][0]:
            return UTCDateTime(result[0][0])
        return None

    def update_FaultDiagnosis_log(self, processed_time, mode):
        """Update the processing progress in the database."""
        # Delete old records and insert new one
        delete_query = "DELETE FROM FaultDiagnosis_log;"
        insert_query = """
        INSERT INTO FaultDiagnosis_log (last_processed_time, mode) 
        VALUES (%s, %s);
        """

        self.execute_query(delete_query)
        return self.execute_query(insert_query, (processed_time.datetime, mode))

    def check_latest_record(self, station_id, end_time, channel, table_type="seismometer"):
        """
        Check if the latest record for this station and channel is more recent than the given end_time.
        Returns True if a more recent record exists, False otherwise.
        """
        table_name = f"{table_type}_diagnosis_hourly"
        query = f"""
        SELECT TimeIndex 
        FROM {table_name}
        WHERE StationID = %s AND Channel = %s 
        ORDER BY TimeIndex DESC 
        LIMIT 1
        """
        result = self.execute_query(query, (station_id, channel), fetch=True)

        if not result:
            return False  # No record exists, so we need to process

        latest_time = UTCDateTime(result[0][0])
        return latest_time >= end_time  # Return True if latest record is more recent or equal

    def insert_hourly_diagnosis(self, timestamp, time_index, station_id, equipment_id, channel,
                                seismo_hs, digitizer_hs, transmission_hs, diagnosis,
                                seismo_desc, digitizer_desc, transmission_desc):
        """Insert hourly diagnosis records into all three tables."""

        # Insert seismometer record
        seismo_query = """
        INSERT INTO seismometer_diagnosis_hourly (TimeStamp, TimeIndex, StationID, EquipmentID, Channel, 
                                                  Seismometer_HealthStatus, Diagnosis, Seismometer_Description)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """

        # Insert digitizer record (with slight time delay to avoid primary key conflict)
        digitizer_query = """
        INSERT INTO digitizer_diagnosis_hourly (TimeStamp, TimeIndex, StationID, EquipmentID, Channel, 
                                               Digitizer_HealthStatus, Diagnosis, Digitizer_Description)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """

        # Insert transmission record (with slight time delay to avoid primary key conflict)
        transmission_query = """
        INSERT INTO transmission_diagnosis_hourly (TimeStamp, TimeIndex, StationID, EquipmentID, Channel, 
                                                   Transmission_HealthStatus, Diagnosis, Transmission_Description)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """

        # Execute insertions with small time delays to avoid primary key conflicts
        from datetime import timedelta

        result1 = self.execute_query(seismo_query, (
            timestamp, time_index, station_id, equipment_id, channel,
            seismo_hs, diagnosis, seismo_desc))

        result2 = self.execute_query(digitizer_query, (
            timestamp + timedelta(milliseconds=1), time_index, station_id, equipment_id, channel,
            digitizer_hs, diagnosis, digitizer_desc))

        result3 = self.execute_query(transmission_query, (
            timestamp + timedelta(milliseconds=2), time_index, station_id, equipment_id, channel,
            transmission_hs, diagnosis, transmission_desc))

        return result1 and result2 and result3

    def insert_daily_diagnosis(self, timestamp, day_index, station_id, equipment_id, force_hour_check=True):
        """Insert daily diagnosis records for all three tables using simplified calculation approach."""

        # First check if we're enforcing the hour check
        if force_hour_check:
            current_hour = UTCDateTime().hour
            if current_hour != 23:
                print(f"⚠️ Refusing to insert daily diagnosis entries outside hour 23 (current hour: {current_hour})")
                return False

        # Check for existing entries in any of the daily tables
        check_queries = [
            "SELECT 1 FROM seismometer_diagnosis_daily WHERE DayIndex = %s AND StationID = %s LIMIT 1;",
            "SELECT 1 FROM digitizer_diagnosis_daily WHERE DayIndex = %s AND StationID = %s LIMIT 1;",
            "SELECT 1 FROM transmission_diagnosis_daily WHERE DayIndex = %s AND StationID = %s LIMIT 1;"
        ]

        for query in check_queries:
            if self.execute_query(query, (day_index, station_id), fetch=True):
                print(f"⚠️ Skipping duplicate daily diagnosis entries for {day_index} - {station_id}")
                return False

        # Calculate date range
        from datetime import datetime, timedelta
        if isinstance(day_index, str):
            day_start = datetime.strptime(day_index, '%Y-%m-%d')
        else:
            day_start = datetime.combine(day_index, datetime.min.time())
        day_end = day_start + timedelta(days=1)

        # Calculate health scores for each component
        tables_and_columns = [
            ("seismometer_diagnosis_hourly", "Seismometer_HealthStatus"),
            ("digitizer_diagnosis_hourly", "Digitizer_HealthStatus"),
            ("transmission_diagnosis_hourly", "Transmission_HealthStatus")
        ]

        health_scores = []
        for table_name, health_column in tables_and_columns:
            count_query = f"""
            SELECT 
                COUNT(*) AS total_count,
                SUM(CASE WHEN {health_column} = 1 THEN 1 ELSE 0 END) AS normal_count
            FROM {table_name}
            WHERE StationID = %s AND TimeIndex >= %s AND TimeIndex < %s
            """

            result = self.execute_query(count_query, (station_id, day_start, day_end), fetch=True)

            if not result or result[0][0] == 0:
                print(f"⚠️ No hourly data found for {station_id} on {day_index} in {table_name}")
                health_scores.append(0)
                continue

            total_count, normal_count = result[0]
            health_score = (normal_count / total_count) * 100 if total_count > 0 else 0.0
            health_scores.append(int(health_score))

        # Insert daily records for all three tables
        daily_queries = [
            """INSERT INTO seismometer_diagnosis_daily (TimeStamp, DayIndex, StationID, EquipmentID, 
               Seismometer_HS, Prognosis, Equipment_HS) VALUES (%s, %s, %s, %s, %s, %s, %s);""",
            """INSERT INTO digitizer_diagnosis_daily (TimeStamp, DayIndex, StationID, EquipmentID, 
               Digitizer_HS, Prognosis, Equipment_HS) VALUES (%s, %s, %s, %s, %s, %s, %s);""",
            """INSERT INTO transmission_diagnosis_daily (TimeStamp, DayIndex, StationID, EquipmentID, 
               Transmission_HS, Prognosis, Equipment_HS) VALUES (%s, %s, %s, %s, %s, %s, %s);"""
        ]

        results = []
        for i, query in enumerate(daily_queries):
            result = self.execute_query(query, (
                timestamp, day_index, station_id, equipment_id, health_scores[i], None, None))
            results.append(result)

        if all(results):
            component_names = ["Seismometer", "Digitizer", "Transmission"]
            print(f"✅ Daily reports inserted for {station_id} at {day_index}")
            for i, (name, score) in enumerate(zip(component_names, health_scores)):
                print(f"   {name} Health Score: {score}%")
            return True

        return False

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            print("✅ Database connection closed.")


# Initialize persistent database connection
db_params = {
    "dbname": "YOUR_DB_NAME",
    "user": "YOUR_USERNAME",
    "password": "YOUR_PASSWORD",
    "host": "localhost",
    "port": "5432"
}


db_manager = DatabaseManager(db_params)

# Constants for NLNM and NHNM values
NHNM_4 = -97.5958
NHNM_19 = -135.074
NHNM_100 = -131.5
NHNM_325 = -126.395

# EQ Transformer
EqT_Model = EQTransformer.from_pretrained("original")
if torch.cuda.is_available():
    EqT_Model.cuda()
else:
    EqT_Model.cpu()  # Use CPU instead

# GPD model
GPD_Model = GPD.from_pretrained("stead")
if torch.cuda.is_available():
    GPD_Model.cuda()
else:
    GPD_Model.cpu()  # Use CPU instead

# One Class Autoencoder
One_AE_path = "saved_models/autoencoder_model.h5"
One_AE_model = load_model(One_AE_path )

One_AE_scaler_path = "saved_models/one_ae_scaler.pkl"
One_AE_scaler = joblib.load(One_AE_scaler_path)

def predict_with_One_AE(data):
    # Predict using the One Class Autoencoder model
    scaled_data = One_AE_scaler.transform(data)
    predict_data = One_AE_model.predict(scaled_data)
    recon_error = np.mean(np.square(scaled_data - predict_data), axis=1)
    threshold = 0.001714 # obtained experimentally during model train evaluation (95th percentile of reconstruction train prediction error)

    if recon_error > threshold:
        Y_pred=0
    else:
        Y_pred=1

    return Y_pred


# DNN classifier
DNN_path = "saved_models/dnn_classifier_model.h5"
DNN_model = load_model(DNN_path)

DNN_scaler_path = "saved_models/dnn_scaler.pkl"
DNN_scaler = joblib.load(DNN_scaler_path)

DNN_encoder_path = "saved_models/label_encoder.pkl"
DNN_encoder = joblib.load(DNN_encoder_path)

def predict_with_DNN(data):
    # Predict using the DNN model
    scaled_data = DNN_scaler.transform(data)
    encoded_predictions = DNN_model.predict(scaled_data)
    return DNN_encoder.inverse_transform(encoded_predictions)


client = Client('Seismic Folder')

# Fetch active seismometers from database
active_seismometers = db_manager.get_active_seismometers()
print(f"Found {len(active_seismometers)} active seismometers to process")


def determine_fault_classification(diagnosis):
    """
    Determine which component is faulty based on SVC diagnosis result.
    Returns tuple: (seismo_hs, digitizer_hs, transmission_hs, seismo_desc, digitizer_desc, transmission_desc)
    """
    if diagnosis in seismo_fault_cases:
        # Seismometer fault
        return (0, 1, 1, "Diagnosis Result", "Seismometer Fault", "Seismometer Fault")
    elif diagnosis in digitizer_fault_cases:
        # Digitizer fault
        return (1, 0, 1, "Digitizer Fault", "Diagnosis Result", "Digitizer Fault")
    elif diagnosis in transmission_fault_cases:
        # Transmission fault
        return (1, 1, 0, "Transmission Fault", "Transmission Fault", "Diagnosis Result")
    else:
        # Unknown fault type - treat as transmission fault
        print(f"⚠️ Unknown diagnosis: {diagnosis}, treating as transmission fault")
        return (1, 1, 0, "Transmission Fault", "Transmission Fault", "Diagnosis Result")


def process_hour_data(start_time, end_time, mode="historical"):
    """
    Process seismometer data for a specific hour.

    Args:
        start_time: UTCDateTime object for the start of the hour
        end_time: UTCDateTime object for the end of the hour
        mode: "historical" or "realtime"
    """
    print(
        f"[{mode.upper()}] Processing data for: {start_time.strftime('%Y-%m-%d %H:00:00')} to {end_time.strftime('%Y-%m-%d %H:00:00')}")

    # Process each active seismometer
    for seismometer_record in active_seismometers:
        EquipmentID, StationID, Equipment, Status, ChannelType = seismometer_record

        # Determine channels based on channel type from Equipments table
        if ChannelType == "S":
            Channels = ["SHE", "SHN", "SHZ"]
        elif ChannelType == "B":
            Channels = ["BHE", "BHN", "BHZ"]
        else:
            print(f"Skipping station {StationID} with unsupported channel type: {ChannelType}")
            continue

        print(f"Processing station {StationID} with channels {Channels}")

        for Channel in Channels:
            # For historical mode, skip checking if record exists to allow reprocessing
            if mode == "realtime" and db_manager.check_latest_record(StationID, end_time, Channel, "seismometer"):
                print(f"Skipping {StationID}-{Channel} at {end_time}: Already processed")
                continue

            try:
                Network_Code = "XYZ" # dummy seismic observation network code
                st = client.get_waveforms(Network_Code, StationID, "*", Channel, start_time, end_time)
                st.merge()
                tr = st[0]

                if isinstance(tr.data, np.ma.MaskedArray):
                    tr.data = tr.data.filled(0)
                    print("Filled masked values with zeros")
                else:
                    print("No masked values found - data is already valid")

                st = Stream(traces=[tr])

            except Exception as e:
                st = None

            if st is None or len(st) == 0:
                print(f"No stream data for {StationID}-{Channel} at {end_time}")
                # Invalid data: Normal for seismo/digitizer, Abnormal for transmission
                seismo_hs, digitizer_hs, transmission_hs = 1, 1, 0
                diagnosis = "AI"
                seismo_desc = "Transmission Fault"
                digitizer_desc = "Transmission Fault"
                transmission_desc = "Diagnosis Result"

            else:
                num = st[0].stats.npts
                SR = st[0].stats.sampling_rate
                length_s = int(num / SR)

                EQ_Threshold = 0.55

                try:
                    GPD_picks = GPD_Model.classify(st, P_threshold=EQ_Threshold, S_threshold=EQ_Threshold).picks

                    annotations = EqT_Model.annotate(st)
                    detection_traces = annotations.select(channel="EQTransformer_Detection")
                    eq_prob = detection_traces[0].data

                    EQ_success = 1

                except Exception as e:
                    EQ_success = 0

                if (EQ_success == 1) and ((len(GPD_picks) > 0) and (any(eq_prob > EQ_Threshold))):
                    # Earthquake detected - all components normal
                    seismo_hs, digitizer_hs, transmission_hs = 1, 1, 1
                    diagnosis = "Normal"
                    seismo_desc = "Earthquake occurrence"
                    digitizer_desc = "Earthquake occurrence"
                    transmission_desc = "Earthquake occurrence"

                else:
                    try:
                        # Calculate PSD features
                        inv_path = f"Inventory Folder/{StationID}.xml"
                        inv = read_inventory(inv_path)
                        ppsd = PPSD(st[0].stats, metadata=inv, ppsd_length=length_s)
                        ppsd.add(st)


                        H_4 = (NHNM_4 - ppsd.extract_psd_values(4)[0][0])
                        H_19 = (NHNM_19 - ppsd.extract_psd_values(19)[0][0])
                        H_100 = (NHNM_100 - ppsd.extract_psd_values(100)[0][0])
                        H_325 = (NHNM_325 - ppsd.extract_psd_values(325)[0][0])

                        features_dict = {
                            "H_4": H_4, "H_19": H_19, "H_100": H_100, "H_325": H_325
                        }

                        error_flag = 0

                    except Exception as e:
                        print(f"Error processing {StationID}-{Channel} at {end_time}: {e}")
                        error_flag = 1

                    if error_flag == 0:
                        try:
                            df_features = pd.DataFrame(features_dict, index=[0])
                            One_AE_result = predict_with_One_AE(df_features)

                            if One_AE_result == 1:
                                # One_AE: Normal ==> all components: normal
                                seismo_hs, digitizer_hs, transmission_hs = 1, 1, 1
                                diagnosis = "Normal"
                                seismo_desc = "Diagnosis Result"
                                digitizer_desc = "Diagnosis Result"
                                transmission_desc = "Diagnosis Result"
                            else:
                                # One_AE: Abnormal ==> use DNN to classify fault
                                diagnosis = predict_with_DNN(df_features)
                                seismo_hs, digitizer_hs, transmission_hs, seismo_desc, digitizer_desc, transmission_desc = determine_fault_classification(
                                    diagnosis)

                        except Exception as e:
                            print(f"Error processing {StationID}-{Channel} at {end_time}: {e}")
                            # Processing failure - treat as transmission fault
                            seismo_hs, digitizer_hs, transmission_hs = 1, 1, 0
                            diagnosis = "AI"
                            seismo_desc = "Transmission Fault"
                            digitizer_desc = "Transmission Fault"
                            transmission_desc = "Diagnosis Result"

                    else:
                        # Feature extraction error - treat as transmission fault
                        seismo_hs, digitizer_hs, transmission_hs = 1, 1, 0
                        diagnosis = "AI"
                        seismo_desc = "Transmission Fault"
                        digitizer_desc = "Transmission Fault"
                        transmission_desc = "Diagnosis Result"

            HS_now_time = UTCDateTime().datetime
            HS_TimeIndex = end_time.datetime

            db_manager.insert_hourly_diagnosis(
                HS_now_time, HS_TimeIndex, str(StationID), EquipmentID, str(Channel),
                seismo_hs, digitizer_hs, transmission_hs, str(diagnosis),
                seismo_desc, digitizer_desc, transmission_desc
            )

            print(
                f"{StationID}_{end_time}_{Channel}: Seismo={seismo_hs}, Digitizer={digitizer_hs}, Transmission={transmission_hs}, Diagnosis={diagnosis}")

        print("-" * 80)

        # Calculate daily health report for historical mode or when appropriate for real-time
        process_date = start_time.date
        process_start = UTCDateTime(process_date.year, process_date.month, process_date.day)
        process_end = process_start + 86400  # 24 hours in seconds

        # For historical mode, generate daily report at the end of each day (hour 23)
        # For real-time mode, use the existing logic
        should_generate_daily_report = False

        if mode == "historical":
            # Generate report at the end of each day (hour 23) in historical mode
            if start_time.hour == 23:
                should_generate_daily_report = True
        elif mode == "realtime":
            # Use existing real-time logic (only at hour 23 of current day)
            should_generate_daily_report = True

        if should_generate_daily_report:
            DR_now_time = UTCDateTime().datetime
            DR_DateIndex = process_date
            force_hour_check = (mode == "realtime")

            if db_manager.insert_daily_diagnosis(DR_now_time, DR_DateIndex, StationID, EquipmentID,
                                                 force_hour_check=force_hour_check):
                print(f"✅ Daily reports generated for all three components")
        print("*" * 80)

    # Update processing progress
    db_manager.update_FaultDiagnosis_log(end_time, mode)


def determine_start_time():
    """
    Determine where to start processing based on the last processed time or default to 2025-01-01.
    """
    last_processed = db_manager.get_last_processed_time()

    if last_processed:
        print(f"Resuming from last processed time: {last_processed}")
        # Start from the next hour after the last processed time
        next_hour = UTCDateTime(last_processed.year, last_processed.month, last_processed.day,
                                last_processed.hour) + 3600
        return next_hour
    else:
        print("No previous processing history found. Starting from 2025-01-01 0:00:00 UTC")
        return UTCDateTime(2025, 1, 1, 0, 0, 0)


def is_realtime_ready(current_process_time):
    """
    Determine if we should switch to real-time mode.
    Switch when we're within 2 hours of current time.
    """
    now = UTCDateTime()
    time_diff = now - current_process_time

    # Switch to real-time when we're within 2 hours of current time
    return time_diff <= 7200  # 2 hours in seconds


try:
    # Determine starting point
    start_time = determine_start_time()

    print(f"🚀 Starting seismometer monitoring system...")
    print(f"📅 Start time: {start_time}")
    print(f"🕐 Current time: {UTCDateTime()}")

    current_time = start_time

    while True:
        try:
            now = UTCDateTime()

            # Check if we should switch to real-time mode
            if is_realtime_ready(current_time):
                print("\n" + "=" * 80)
                print("🔄 SWITCHING TO REAL-TIME MODE")
                print("=" * 80 + "\n")

                # Real-time mode processing
                while True:
                    try:
                        # Get the current time
                        current_time = UTCDateTime()

                        # Check if we're past the 30-minute mark of the current hour
                        if current_time.minute < 30:
                            # If we're in the first 30 minutes of the hour, wait until 30-minute mark
                            print(f"Current time: {current_time.strftime('%H:%M:%S')}")
                            print(f"Waiting until 30 minutes past the hour to ensure data availability")
                            sleep_time = (30 - current_time.minute) * 60 - current_time.second
                            if sleep_time > 0:
                                print(f"Sleeping for {sleep_time} seconds...")
                                time.sleep(sleep_time)
                                current_time = UTCDateTime()  # Update current time after sleeping

                        # Process the previous hour's data
                        process_hour = current_time.hour - 1

                        # Handle hour rollover at midnight
                        if process_hour < 0:
                            process_hour = 23
                            # Also need to adjust the date to previous day
                            process_date = UTCDateTime(current_time.year, current_time.month, current_time.day) - 86400
                        else:
                            process_date = UTCDateTime(current_time.year, current_time.month, current_time.day)

                        # Set StartTime to the beginning of the hour we're processing
                        StartTime = UTCDateTime(process_date.year, process_date.month, process_date.day,
                                                process_hour, 0, 0)
                        EndTime = StartTime + 3600  # One hour later

                        print(f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

                        process_hour_data(StartTime, EndTime, mode="realtime")

                        # Wait until the next hour before processing again
                        next_hour = UTCDateTime(current_time.year, current_time.month, current_time.day,
                                                current_time.hour, 0, 0) + 3600
                        sleep_time = max(10, (next_hour - UTCDateTime()) + 10)  # Add 10 seconds buffer
                        print(f"Waiting {sleep_time} seconds until next processing cycle")
                        time.sleep(sleep_time)

                    except KeyboardInterrupt:
                        print("Process interrupted by user. Exiting...")
                        break
                    except Exception as e:
                        print(f"Unexpected error in real-time mode: {e}")
                        time.sleep(60)  # Wait a minute before retrying on unexpected errors

                break  # Exit the historical processing loop

            else:
                # Historical mode processing
                end_time = current_time + 3600  # One hour later

                # Show progress
                total_hours_to_process = (now - start_time) / 3600
                hours_processed = (current_time - start_time) / 3600
                if total_hours_to_process > 0:
                    progress = (hours_processed / total_hours_to_process) * 100
                    print(
                        f"📊 Historical processing progress: {progress:.1f}% ({hours_processed:.0f}/{total_hours_to_process:.0f} hours)")

                process_hour_data(current_time, end_time, mode="historical")

                # Move to next hour
                current_time = end_time

                # Add a small delay to prevent overwhelming the system
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("Process interrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            time.sleep(60)  # Wait a minute before retrying on unexpected errors

except Exception as e:
    print(f"Critical error: {e}")
finally:
    db_manager.close()