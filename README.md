# govee-heat-loss-tracker

Reads data from Govee Air Quality (AQ) sensors and displays the rate of heat loss (celsius per minute).

This Streamlit application connects to Govee AQ monitors ("aq-monitor-office" and "aq-monitor-livingroom") via the Govee API, collects temperature data over configurable time windows, calculates the average temperature change rate per minute, and visualizes the data with interactive charts.

## Features

- 📊 Real-time temperature data collection from multiple Govee AQ monitors
- 📈 Calculate average temperature drop/rise rate per minute
- 📉 Interactive visualization with 3 lines: Office monitor, Living Room monitor, and Average
- ⏱️ Configurable data collection duration and polling intervals
- 📱 Responsive web interface built with Streamlit

## Requirements

- Python 3.8 or higher
- Govee AQ monitors (aq-monitor-office and aq-monitor-livingroom)
- Govee API key

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

3. Configure your Govee API key:
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your Govee API key
# GOVEE_API_KEY=your_api_key_here
```

Alternatively, you can set the environment variable directly:
```bash
export GOVEE_API_KEY=your_api_key_here
```

Or use Streamlit secrets (create `.streamlit/secrets.toml`):
```toml
GOVEE_API_KEY = "your_api_key_here"
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser to the URL shown (typically http://localhost:8501)

3. Configure the data collection parameters in the sidebar:
   - **Data Collection Duration**: How long to collect data (10-120 minutes)
   - **Polling Interval**: How often to poll the API (30-300 seconds)

4. Click "Start Data Collection" to begin tracking temperature

5. View the results in three tabs:
   - **Office Monitor**: Data from the office AQ monitor
   - **Living Room Monitor**: Data from the living room AQ monitor
   - **Combined Analysis**: Side-by-side comparison with metrics and charts

## How It Works

1. **Data Collection**: The app polls the Govee API at regular intervals to fetch temperature readings from both AQ monitors.

2. **Rate Calculation**: For each monitor, the app calculates the average temperature change rate (°C per minute) over the collection window.

3. **Visualization**: Three lines are plotted:
   - Blue line: Office monitor temperature
   - Green line: Living room monitor temperature
   - Red dashed line: Average of both monitors

4. **Metrics**: The app displays:
   - Temperature change rate per minute for each monitor
   - Temperature change per hour (scaled from per-minute rate)
   - Summary statistics (min, max, average, standard deviation)

## Understanding the Results

- **Negative change rate**: Indicates cooling (heat loss)
- **Positive change rate**: Indicates warming (heat gain)
- **Zero change rate**: Temperature is stable

## API Configuration

The app uses the Govee API with the following configuration:
- **API Endpoint**: https://developer-api.govee.com
- **API Key**: Configured in app.py
- **Monitors**: aq-monitor-office and aq-monitor-livingroom

## Troubleshooting

- **No data collected**: Ensure your Govee monitors are online and properly named in the Govee app
- **API errors**: Check that your API key is valid and has not exceeded rate limits
- **Connection issues**: Verify your internet connection and firewall settings

## License

MIT License
