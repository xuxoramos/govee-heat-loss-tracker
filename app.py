import streamlit as st
import pandas as pd
import numpy as np
import glob
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
from sklearn.linear_model import HuberRegressor

import plotly.graph_objects as go

st.set_page_config(page_title="Temperature Loss Tracker", layout="wide")

@st.cache_data
def load_data():
    # Read livingroom CSV files
    livingroom_files = glob.glob('data/aq-monitor-livingroom*.csv')
    livingroom = pd.concat([pd.read_csv(f) for f in livingroom_files], ignore_index=True)
    
    # Read office CSV files
    office_files = glob.glob('data/aq-monitor-office*.csv')
    office = pd.concat([pd.read_csv(f) for f in office_files], ignore_index=True)
    
    # Standardize column names
    livingroom.columns = ['timestamp', 'pm25', 'temperature_f', 'humidity']
    office.columns = ['timestamp', 'pm25', 'temperature_f', 'humidity']
    
    # Sort by timestamp
    livingroom = livingroom.sort_values(by='timestamp', ascending=True)
    office = office.sort_values(by='timestamp', ascending=True)
    
    # Convert temperature from Fahrenheit to Celsius
    livingroom['temperature_c'] = (livingroom['temperature_f'] - 32) * 5/9
    office['temperature_c'] = (office['temperature_f'] - 32) * 5/9
    
    # Drop Fahrenheit column
    livingroom = livingroom.drop('temperature_f', axis=1)
    office = office.drop('temperature_f', axis=1)
    
    return livingroom, office

def insert_gaps(df):
    last_records = df.groupby(df['datetime'].dt.date).last()
    gaps = last_records.copy()
    gaps['temperature_c'] = np.nan
    gaps['datetime'] = gaps['datetime'] + pd.Timedelta(hours=1)
    df_with_gaps = pd.concat([df, gaps]).sort_values(by='datetime').reset_index(drop=True)
    return df_with_gaps

def add_peak_column(df):
    df['is_peak'] = 0
    for date in df['datetime'].dt.date.unique():
        daily_data = df[df['datetime'].dt.date == date]
        peaks, _ = find_peaks(daily_data['temperature_c'].dropna(), distance=20, prominence=0.3, width=10)
        valleys, _ = find_peaks(-daily_data['temperature_c'].dropna(), distance=20, prominence=0.3, width=10)
        peak_indices = daily_data.iloc[peaks].index
        valley_indices = daily_data.iloc[valleys].index
        df.loc[peak_indices, 'is_peak'] = 1
        df.loc[valley_indices, 'is_peak'] = -1

def calculate_cooling_rates(df):
    df['cooling_rate'] = np.nan
    peak_indices = df[df['is_peak'] == 1].index.tolist()
    valley_indices = df[df['is_peak'] == -1].index.tolist()
    previous_rate = np.nan
    
    for peak_idx in peak_indices:
        next_valleys = [v for v in valley_indices if v > peak_idx]
        
        if len(next_valleys) > 0:
            valley_idx = next_valleys[0]
            segment = df.loc[peak_idx:valley_idx]
            segment_clean = segment[segment['temperature_c'].notna()]
            
            if len(segment_clean) >= 2:
                X = np.arange(len(segment_clean)).reshape(-1, 1)
                y = segment_clean['temperature_c'].values
                huber = HuberRegressor()
                huber.fit(X, y)
                cooling_rate = huber.coef_[0]
                previous_rate = cooling_rate
            else:
                cooling_rate = previous_rate
        else:
            cooling_rate = previous_rate
        
        if len(next_valleys) > 0:
            valley_idx = next_valleys[0]
            df.loc[peak_idx:valley_idx, 'cooling_rate'] = cooling_rate
        else:
            df.loc[peak_idx:, 'cooling_rate'] = cooling_rate
    
    return df

def create_temp_plot(df, room_name):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['datetime'],
        y=df['temperature_c'],
        mode='lines',
        name=f'{room_name} Temperature',
        line=dict(color='blue'),
        showlegend=False,
        connectgaps=False
    ))
    
    peaks_data = df[df['is_peak'] == 1]
    for idx, row in peaks_data.iterrows():
        fig.add_vline(x=row['datetime'], line_dash="dot", line_color="red", opacity=0.5)
    
    valleys_data = df[df['is_peak'] == -1]
    for idx, row in valleys_data.iterrows():
        fig.add_vline(x=row['datetime'], line_dash="dot", line_color="darkgreen", opacity=0.5)
    
    peak_indices = df[df['is_peak'] == 1].index.tolist()
    for i in range(len(peak_indices) - 1):
        start_idx = peak_indices[i]
        end_idx = peak_indices[i + 1]
        segment = df.loc[start_idx:end_idx]
        cooling_rate = segment['cooling_rate'].iloc[0]
        
        if pd.notna(cooling_rate):
            mid_datetime = segment['datetime'].iloc[len(segment)//2]
            mid_temp = segment['temperature_c'].iloc[len(segment)//2]
            fig.add_annotation(
                x=mid_datetime, y=mid_temp, text=f'{cooling_rate:.4f}',
                showarrow=False, font=dict(size=10, color='darkgreen'),
                bgcolor='rgba(255, 255, 255, 0.7)', bordercolor='darkgreen', borderwidth=1
            )
    
    fig.update_layout(
        title=f'{room_name} Temperature Time Series with Cooling Rates',
        xaxis=dict(title='Timestamp', rangeslider=dict(visible=True)),
        yaxis=dict(title='Temperature (°C)'),
        hovermode='x unified',
        height=600
    )
    
    return fig

# Main app
st.title("🌡️ Temperature Loss Tracker")

livingroom, office = load_data()

# Convert timestamp to datetime
livingroom['datetime'] = pd.to_datetime(livingroom['timestamp'])
office['datetime'] = pd.to_datetime(office['timestamp'])

# Filter records
livingroom_hours = livingroom[(livingroom['datetime'].dt.hour >= 0) & (livingroom['datetime'].dt.hour <= 7)]
office_hours = office[(office['datetime'].dt.hour >= 0) & (office['datetime'].dt.hour < 7)]

livingroom_hours_months = livingroom_hours[livingroom_hours['datetime'].dt.month.isin([11, 12])].copy()
office_hours_months = office_hours[office_hours['datetime'].dt.month.isin([11, 12])].copy()

# Process data
livingroom_hours_months = insert_gaps(livingroom_hours_months)
office_hours_months = insert_gaps(office_hours_months)

add_peak_column(livingroom_hours_months)
add_peak_column(office_hours_months)

livingroom_cooling_rates = calculate_cooling_rates(livingroom_hours_months.copy())
office_cooling_rates = calculate_cooling_rates(office_hours_months.copy())

# Calculate daily statistics
daily_stats_office = office_cooling_rates[office_cooling_rates['cooling_rate'].notna()].groupby(
    office_cooling_rates['datetime'].dt.date)['cooling_rate'].agg(['mean', 'median', 'std']).reset_index()
daily_stats_office.columns = ['date', 'mean_cooling_rate_office', 'median_cooling_rate_office', 'std_cooling_rate_office']

daily_stats_livingroom = livingroom_cooling_rates[livingroom_cooling_rates['cooling_rate'].notna()].groupby(
    livingroom_cooling_rates['datetime'].dt.date)['cooling_rate'].agg(['mean', 'median', 'std']).reset_index()
daily_stats_livingroom.columns = ['date', 'mean_cooling_rate_livingroom', 'median_cooling_rate_livingroom', 'std_cooling_rate_livingroom']

daily_stats = pd.merge(daily_stats_office, daily_stats_livingroom, on='date', how='inner')

pitt_min_temp = pd.read_csv('data/pitt_min_temp.csv')
pitt_min_temp['DATE'] = pd.to_datetime(pitt_min_temp['DATE'])
daily_stats['date'] = pd.to_datetime(daily_stats['date'])

daily_stats = pd.merge(daily_stats, pitt_min_temp[['DATE', 'TMIN']], left_on='date', right_on='DATE', how='left')
daily_stats = daily_stats.drop('DATE', axis=1)
daily_stats = daily_stats.rename(columns={'TMIN': 'daily_min_temp'})

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Office", "Living Room", "Daily Statistics", "Data Table"])

with tab1:
    st.header("Office Temperature Time Series")
    st.plotly_chart(create_temp_plot(office_cooling_rates, "Office"), use_container_width=True)

with tab2:
    st.header("Living Room Temperature Time Series")
    st.plotly_chart(create_temp_plot(livingroom_cooling_rates, "Living Room"), use_container_width=True)

with tab3:
    st.header("Daily Cooling Rate Statistics")
    
    # Create comparison plot
    fig = make_subplots(
        rows=2, cols=1, subplot_titles=('Office', 'Living Room'),
        vertical_spacing=0.12, specs=[[{"secondary_y": True}], [{"secondary_y": True}]]
    )
    
    fig.add_trace(go.Scatter(x=daily_stats['date'], y=daily_stats['mean_cooling_rate_office'],
        mode='lines', name='Office Mean', line=dict(color='blue')), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=daily_stats['date'], y=daily_stats['median_cooling_rate_office'],
        mode='lines', name='Office Median', line=dict(color='blue', dash='dash')), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Bar(x=daily_stats['date'], y=daily_stats['daily_min_temp'],
        name='Min Temp', marker=dict(color='red', opacity=0.3)), row=1, col=1, secondary_y=True)
    
    fig.add_trace(go.Scatter(x=daily_stats['date'], y=daily_stats['mean_cooling_rate_livingroom'],
        mode='lines', name='Living Room Mean', line=dict(color='green')), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=daily_stats['date'], y=daily_stats['median_cooling_rate_livingroom'],
        mode='lines', name='Living Room Median', line=dict(color='green', dash='dash')), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Bar(x=daily_stats['date'], y=daily_stats['daily_min_temp'],
        name='Min Temp', marker=dict(color='red', opacity=0.3), showlegend=False), row=2, col=1, secondary_y=True)
    
    fig.update_xaxes(dtick=7*24*60*60*1000, tickformat='%Y-%m-%d')
    fig.update_yaxes(title_text='Cooling Rate (°C/min)', row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text='Min Temp (°C)', row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text='Cooling Rate (°C/min)', row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text='Min Temp (°C)', row=2, col=1, secondary_y=True)
    
    fig.update_layout(height=800, title_text='Daily Cooling Rates by Room', hovermode='x unified')
    
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Daily Statistics Table")
    st.dataframe(daily_stats, use_container_width=True)