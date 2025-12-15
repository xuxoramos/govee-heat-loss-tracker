import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Govee API configuration
GOVEE_API_KEY = "b60cedae-5c36-4a21-8da0-e9c6311a052b"
GOVEE_API_BASE = "https://developer-api.govee.com"

# Monitor names
MONITORS = {
    "office": "aq-monitor-office",
    "livingroom": "aq-monitor-livingroom"
}

def get_govee_devices():
    """Fetch list of Govee devices."""
    headers = {
        "Govee-API-Key": GOVEE_API_KEY,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(
            f"{GOVEE_API_BASE}/v1/devices",
            headers=headers
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching devices: {e}")
        return None

def get_device_state(device_id, model):
    """Fetch current state of a Govee device."""
    headers = {
        "Govee-API-Key": GOVEE_API_KEY,
        "Content-Type": "application/json"
    }
    
    params = {
        "device": device_id,
        "model": model
    }
    
    try:
        response = requests.get(
            f"{GOVEE_API_BASE}/v1/devices/state",
            headers=headers,
            params=params
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching device state: {e}")
        return None

def collect_temperature_data(device_id, model, device_name, duration_minutes=60, interval_seconds=60):
    """
    Collect temperature data from a device over a specified duration.
    
    Args:
        device_id: Device ID
        model: Device model
        device_name: Human-readable device name
        duration_minutes: How long to collect data (default 60 minutes)
        interval_seconds: How often to poll (default 60 seconds)
    
    Returns:
        DataFrame with timestamp and temperature columns
    """
    data = []
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=duration_minutes)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    iteration = 0
    total_iterations = duration_minutes * 60 // interval_seconds
    
    while datetime.now() < end_time:
        state = get_device_state(device_id, model)
        
        if state and 'data' in state:
            properties = state['data'].get('properties', [])
            temp_data = next((p for p in properties if p.get('key') == 'temperature'), None)
            
            if temp_data:
                temperature = temp_data.get('value', 0) / 100.0  # Govee returns temp in hundredths
                timestamp = datetime.now()
                data.append({
                    'timestamp': timestamp,
                    'temperature': temperature,
                    'device': device_name
                })
                
                status_text.text(f"Collecting data from {device_name}: {temperature}°C at {timestamp.strftime('%H:%M:%S')}")
        
        iteration += 1
        progress_bar.progress(min(iteration / total_iterations, 1.0))
        
        if datetime.now() < end_time:
            time.sleep(interval_seconds)
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(data)

def calculate_temperature_drop_rate(df):
    """
    Calculate the average temperature drop per minute in 1-hour windows.
    
    Args:
        df: DataFrame with timestamp and temperature columns
    
    Returns:
        Average temperature drop per minute (negative value indicates cooling)
    """
    if len(df) < 2:
        return 0.0
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Calculate time differences in minutes
    df['time_diff_minutes'] = df['timestamp'].diff().dt.total_seconds() / 60.0
    
    # Calculate temperature differences
    df['temp_diff'] = df['temperature'].diff()
    
    # Calculate rate of change (degrees per minute)
    df['temp_rate'] = df['temp_diff'] / df['time_diff_minutes']
    
    # Return average rate (excluding first NaN value)
    return df['temp_rate'].mean()

def plot_temperature_data(office_df, livingroom_df):
    """
    Plot temperature data from both monitors and their average.
    
    Args:
        office_df: DataFrame with office monitor data
        livingroom_df: DataFrame with living room monitor data
    """
    fig = go.Figure()
    
    # Plot office monitor
    if not office_df.empty:
        fig.add_trace(go.Scatter(
            x=office_df['timestamp'],
            y=office_df['temperature'],
            mode='lines+markers',
            name='Office Monitor',
            line=dict(color='blue')
        ))
    
    # Plot living room monitor
    if not livingroom_df.empty:
        fig.add_trace(go.Scatter(
            x=livingroom_df['timestamp'],
            y=livingroom_df['temperature'],
            mode='lines+markers',
            name='Living Room Monitor',
            line=dict(color='green')
        ))
    
    # Calculate and plot average
    if not office_df.empty and not livingroom_df.empty:
        # Merge dataframes on timestamp (or use nearest timestamp)
        combined_df = pd.merge(
            office_df[['timestamp', 'temperature']].rename(columns={'temperature': 'temp_office'}),
            livingroom_df[['timestamp', 'temperature']].rename(columns={'temperature': 'temp_livingroom'}),
            on='timestamp',
            how='outer'
        ).sort_values('timestamp')
        
        # Forward fill missing values
        combined_df = combined_df.fillna(method='ffill').fillna(method='bfill')
        
        # Calculate average
        combined_df['temp_average'] = (combined_df['temp_office'] + combined_df['temp_livingroom']) / 2
        
        fig.add_trace(go.Scatter(
            x=combined_df['timestamp'],
            y=combined_df['temp_average'],
            mode='lines+markers',
            name='Average',
            line=dict(color='red', dash='dash')
        ))
    
    fig.update_layout(
        title='Temperature Readings from Govee AQ Monitors',
        xaxis_title='Time',
        yaxis_title='Temperature (°C)',
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(
        page_title="Govee Heat Loss Tracker",
        page_icon="🌡️",
        layout="wide"
    )
    
    st.title("🌡️ Govee Heat Loss Tracker")
    st.markdown("Track temperature readings from Govee Air Quality Monitors and calculate heat loss rates.")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    duration = st.sidebar.slider(
        "Data Collection Duration (minutes)",
        min_value=10,
        max_value=120,
        value=60,
        step=5
    )
    
    interval = st.sidebar.slider(
        "Polling Interval (seconds)",
        min_value=30,
        max_value=300,
        value=60,
        step=30
    )
    
    # Show device information
    st.header("Connected Devices")
    
    with st.expander("View Device List"):
        devices = get_govee_devices()
        if devices:
            st.json(devices)
    
    # Main collection button
    if st.button("Start Data Collection", type="primary"):
        st.info(f"Collecting data for {duration} minutes with {interval}-second intervals...")
        
        # For demo purposes, we'll use mock device IDs
        # In production, these would be retrieved from the devices list
        office_device_id = "AQ:01:23:45:67:89"  # Placeholder
        office_model = "H5179"  # Placeholder - typical Govee AQ monitor model
        
        livingroom_device_id = "AQ:01:23:45:67:90"  # Placeholder
        livingroom_model = "H5179"  # Placeholder
        
        # Create tabs for each monitor
        tab1, tab2, tab3 = st.tabs(["Office Monitor", "Living Room Monitor", "Combined Analysis"])
        
        # Collect data from office monitor
        with tab1:
            st.subheader("Office Monitor Data Collection")
            office_df = collect_temperature_data(
                office_device_id,
                office_model,
                "Office",
                duration,
                interval
            )
            
            if not office_df.empty:
                st.success(f"Collected {len(office_df)} data points from office monitor")
                st.dataframe(office_df)
            else:
                st.warning("No data collected from office monitor")
        
        # Collect data from living room monitor
        with tab2:
            st.subheader("Living Room Monitor Data Collection")
            livingroom_df = collect_temperature_data(
                livingroom_device_id,
                livingroom_model,
                "Living Room",
                duration,
                interval
            )
            
            if not livingroom_df.empty:
                st.success(f"Collected {len(livingroom_df)} data points from living room monitor")
                st.dataframe(livingroom_df)
            else:
                st.warning("No data collected from living room monitor")
        
        # Combined analysis
        with tab3:
            st.subheader("Temperature Analysis")
            
            # Calculate temperature drop rates
            col1, col2, col3 = st.columns(3)
            
            with col1:
                office_rate = calculate_temperature_drop_rate(office_df) if not office_df.empty else 0.0
                st.metric(
                    "Office - Temp Change Rate",
                    f"{office_rate:.4f} °C/min",
                    delta=f"{office_rate * 60:.2f} °C/hour"
                )
            
            with col2:
                livingroom_rate = calculate_temperature_drop_rate(livingroom_df) if not livingroom_df.empty else 0.0
                st.metric(
                    "Living Room - Temp Change Rate",
                    f"{livingroom_rate:.4f} °C/min",
                    delta=f"{livingroom_rate * 60:.2f} °C/hour"
                )
            
            with col3:
                avg_rate = (office_rate + livingroom_rate) / 2
                st.metric(
                    "Average Temp Change Rate",
                    f"{avg_rate:.4f} °C/min",
                    delta=f"{avg_rate * 60:.2f} °C/hour"
                )
            
            # Plot temperature data
            st.subheader("Temperature Trends")
            plot_temperature_data(office_df, livingroom_df)
            
            # Show summary statistics
            st.subheader("Summary Statistics")
            
            summary_data = []
            
            if not office_df.empty:
                summary_data.append({
                    'Monitor': 'Office',
                    'Min Temp (°C)': office_df['temperature'].min(),
                    'Max Temp (°C)': office_df['temperature'].max(),
                    'Avg Temp (°C)': office_df['temperature'].mean(),
                    'Std Dev': office_df['temperature'].std(),
                    'Change Rate (°C/min)': office_rate
                })
            
            if not livingroom_df.empty:
                summary_data.append({
                    'Monitor': 'Living Room',
                    'Min Temp (°C)': livingroom_df['temperature'].min(),
                    'Max Temp (°C)': livingroom_df['temperature'].max(),
                    'Avg Temp (°C)': livingroom_df['temperature'].mean(),
                    'Std Dev': livingroom_df['temperature'].std(),
                    'Change Rate (°C/min)': livingroom_rate
                })
            
            if summary_data:
                st.table(pd.DataFrame(summary_data))
    
    # Instructions
    st.markdown("---")
    st.markdown("""
    ### Instructions
    1. Ensure your Govee AQ monitors are online and connected
    2. Configure the data collection duration and polling interval in the sidebar
    3. Click "Start Data Collection" to begin tracking temperature
    4. View real-time data collection progress for each monitor
    5. Analyze temperature trends and heat loss rates in the combined analysis tab
    
    **Note:** The application will poll the Govee API at the specified interval to collect temperature data.
    A negative change rate indicates cooling (heat loss), while positive indicates warming.
    """)

if __name__ == "__main__":
    main()
