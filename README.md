# govee-heat-loss-tracker

Ingest manually exported CSV data from Govee H5106 Air Quality monitors and report on overnight temperature loss (degC per minute).

This Streamlit application reads the CSVs you download from the Govee app (stored under `data/office` and `data/livingroom`), converts Fahrenheit temperatures to Celsius, and summarises how quickly each room cools between midnight and 07:00.

## Features

- 📁 Load office and living room CSV exports directly from disk (no Bluetooth or API calls)
- 🔁 Auto-detect the latest export based on the timestamp embedded in the filename
- 🌡️ Convert Fahrenheit readings to Celsius and focus on overnight data (00:00-07:00)
- 🕐 Report hourly cooling loss (degC per minute) alongside totals for each monitor
- 📉 Visualise temperature traces and inspect the raw filtered readings
- 📱 Responsive web interface built with Streamlit

## Requirements

- Python 3.8 or higher
- Govee H5106 Air Quality monitors (manual CSV exports from the Govee app)
- CSV files placed under `data/office` and `data/livingroom` (filenames include a timestamp such as `aq-monitor-office_export_202511050547.csv`)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/xuxoramos/govee-heat-loss-tracker.git
cd govee-heat-loss-tracker
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Export fresh data for each monitor via the Govee mobile app and place the CSV files into:
   - `data/office`
   - `data/livingroom`

   Filenames should retain the timestamp suffix so the dashboard can sort them chronologically.

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser to the URL shown (typically http://localhost:8501)

3. In the sidebar, choose which CSV to analyse for each monitor (the latest file is pre-selected).

4. Review the overnight cooling metrics, hourly loss table, temperature chart, and filtered raw readings.

## How It Works

1. **File discovery**: The app lists CSV exports found in `data/office` and `data/livingroom`, automatically highlighting the most recent file per monitor.

2. **Parsing**: Column headers are normalised (for example `Timestamp for sample frequency every 1 min min`, `PM2.5(ug/m3)`, `Temperature_Fahrenheit`, `Relative_Humidity`). Timestamps are converted to datetime objects and temperatures are converted to Celsius.

3. **Filtering**: Only measurements between 00:00 and 07:00 are retained for analysis.

4. **Hourly metrics**: For each hourly window (00:00-01:00 through 06:00-07:00), the app computes the average cooling rate (degC per minute) and the total loss for that hour.

5. **Visualisation and review**: Temperature traces, hourly tables, and the filtered raw readings are displayed for quick inspection.

## Understanding the Results

- **Negative change rate**: Indicates cooling (heat loss)
- **Positive change rate**: Indicates warming (heat gain)
- **Zero change rate**: Temperature is stable

## Troubleshooting

- **No overnight samples**: Confirm the CSV actually contains readings between 00:00 and 07:00; some exports may start later in the day.
- **Unexpected column names**: Ensure the CSV headers match the expected layout (`timestamp`, `pm25`, `temp (in F)` or `temp`, `relative humidity`).
- **Duplicate filenames**: Keep the timestamp suffix intact so the app can pick the latest export automatically.

## License

MIT License
