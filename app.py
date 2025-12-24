import streamlit as st
import pandas as pd
import numpy as np
import glob
from scipy.signal import find_peaks
from sklearn.linear_model import HuberRegressor
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(page_title="Heat Loss Tracker", layout="wide")

# Title
st.title("🌡️ Heat Loss Tracker")
st.markdown("Analysis of temperature cooling rates in different rooms")

# Sidebar for data loading
st.sidebar.header("Data Configuration")

@st.cache_data
def load_data():
    """Load and preprocess the data"""
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
    
    # Drop the Fahrenheit column
    livingroom = livingroom.drop('temperature_f', axis=1)
    office = office.drop('temperature_f', axis=1)
    
    # Convert timestamp to datetime
    livingroom['datetime'] = pd.to_datetime(livingroom['timestamp'])
    office['datetime'] = pd.to_datetime(office['timestamp'])
    
    # Filter records between midnight and 7 AM
    livingroom_hours = livingroom[(livingroom['datetime'].dt.hour >= 0) & (livingroom['datetime'].dt.hour < 7)]
    office_hours = office[(office['datetime'].dt.hour >= 0) & (office['datetime'].dt.hour < 7)]
    
    # Keep records from October to December only
    livingroom_hours_months = livingroom_hours[livingroom_hours['datetime'].dt.month.isin([11, 12])]
    office_hours_months = office_hours[office_hours['datetime'].dt.month.isin([11, 12])]
    
    return livingroom_hours_months, office_hours_months

def insert_gaps(df):
    """Insert NaN temperature record before the first of every day"""
    first_records = df[df['datetime'].dt.hour == 0].groupby(df['datetime'].dt.date).first()
    gaps = first_records.copy()
    gaps['temperature_c'] = np.nan
    gaps['datetime'] = gaps['datetime'] - pd.Timedelta(minutes=1)
    df_with_gaps = pd.concat([df, gaps]).sort_values(by='datetime').reset_index(drop=True)
    return df_with_gaps

def add_peak_column(df):
    """Detect peaks and add the is_peak column to the dataframe"""
    df['is_peak'] = 0
    for date in df['datetime'].dt.date.unique():
        daily_data = df[df['datetime'].dt.date == date]
        peaks, _ = find_peaks(daily_data['temperature_c'].dropna(), distance=20, prominence=0.3, width=10)
        peak_indices = daily_data.iloc[peaks].index
        df.loc[peak_indices, 'is_peak'] = 1
    return df

def calculate_cooling_rates(df):
    """Calculate cooling rate for segments between peaks"""
    df['cooling_rate'] = np.nan
    peak_indices = df[df['is_peak'] == 1].index.tolist()
    
    for i in range(len(peak_indices) - 1):
        start_idx = peak_indices[i]
        end_idx = peak_indices[i + 1]
        
        segment = df.loc[start_idx:end_idx].copy()
        segment_clean = segment.dropna(subset=['temperature_c'])
        
        if len(segment_clean) >= 2:
            X = np.arange(len(segment_clean)).reshape(-1, 1)
            y = segment_clean['temperature_c'].values
            
            huber = HuberRegressor()
            huber.fit(X, y)
            
            cooling_rate = huber.coef_[0]
            df.loc[start_idx:end_idx, 'cooling_rate'] = cooling_rate
    
    return df

def create_temperature_plot(df, room_name):
    """Create temperature time series plot with cooling rates"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['datetime'],
        y=df['temperature_c'],
        mode='lines',
        name=f'{room_name} Temperature',
        line=dict(color='blue'),
        showlegend=False
    ))
    
    peaks_data = df[df['is_peak'] == 1]
    for idx, row in peaks_data.iterrows():
        fig.add_vline(
            x=row['datetime'],
            line_dash="dot",
            line_color="red",
            opacity=0.5
        )
    
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
                x=mid_datetime,
                y=mid_temp,
                text=f'{cooling_rate:.4f}',
                showarrow=False,
                font=dict(size=10, color='darkgreen'),
                bgcolor='rgba(255, 255, 255, 0.7)',
                bordercolor='darkgreen',
                borderwidth=1
            )
    
    fig.update_layout(
        title=f'{room_name} Temperature Time Series with Cooling Rates',
        xaxis=dict(
            title='Timestamp',
            rangeselector=dict(
                buttons=[
                    dict(count=1, label='1d', step='day', stepmode='backward'),
                    dict(count=7, label='1w', step='day', stepmode='backward'),
                    dict(count=1, label='1m', step='month', stepmode='backward'),
                    dict(step='all')
                ]
            ),
            rangeslider=dict(visible=True)
        ),
        yaxis=dict(title='Temperature (°C)'),
        hovermode='x unified',
        showlegend=False
    )
    
    return fig

def create_daily_stats_plot(daily_stats):
    """Create daily statistics plot with subplots for each room"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Office', 'Living Room'),
        vertical_spacing=0.12,
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]]
    )
    
    # Office traces
    fig.add_trace(
        go.Scatter(
            x=daily_stats['date'],
            y=daily_stats['mean_cooling_rate_office'],
            mode='lines',
            name='Office Mean',
            line=dict(color='blue')
        ),
        row=1, col=1, secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=daily_stats['date'],
            y=daily_stats['median_cooling_rate_office'],
            mode='lines',
            name='Office Median',
            line=dict(color='blue', dash='dash')
        ),
        row=1, col=1, secondary_y=False
    )
    
    fig.add_trace(
        go.Bar(
            x=daily_stats['date'],
            y=daily_stats['daily_min_temp'],
            name='Min Temp',
            marker=dict(color='red', opacity=0.3),
            showlegend=True
        ),
        row=1, col=1, secondary_y=True
    )
    
    # Living room traces
    fig.add_trace(
        go.Scatter(
            x=daily_stats['date'],
            y=daily_stats['mean_cooling_rate_livingroom'],
            mode='lines',
            name='Living Room Mean',
            line=dict(color='green')
        ),
        row=2, col=1, secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=daily_stats['date'],
            y=daily_stats['median_cooling_rate_livingroom'],
            mode='lines',
            name='Living Room Median',
            line=dict(color='green', dash='dash')
        ),
        row=2, col=1, secondary_y=False
    )
    
    fig.add_trace(
        go.Bar(
            x=daily_stats['date'],
            y=daily_stats['daily_min_temp'],
            name='Min Temp',
            marker=dict(color='red', opacity=0.3),
            showlegend=False
        ),
        row=2, col=1, secondary_y=True
    )
    
    # Update axes
    fig.update_xaxes(
        dtick=7*24*60*60*1000,
        tickformat='%Y-%m-%d',
        row=1, col=1
    )
    
    fig.update_xaxes(
        title_text='Date',
        dtick=7*24*60*60*1000,
        tickformat='%Y-%m-%d',
        row=2, col=1
    )
    
    fig.update_yaxes(title_text='Cooling Rate (°C/min)', row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text='Min Temp (°C)', row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text='Cooling Rate (°C/min)', row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text='Min Temp (°C)', row=2, col=1, secondary_y=True)
    
    fig.update_layout(
        height=800,
        title_text='Daily Cooling Rates by Room',
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

# Main app
with st.spinner('Loading data...'):
    livingroom_hours_months, office_hours_months = load_data()

st.sidebar.success(f"✓ Loaded {len(livingroom_hours_months)} livingroom records")
st.sidebar.success(f"✓ Loaded {len(office_hours_months)} office records")

# Process data
with st.spinner('Processing data...'):
    livingroom_hours_months = insert_gaps(livingroom_hours_months)
    office_hours_months = insert_gaps(office_hours_months)
    
    livingroom_hours_months = add_peak_column(livingroom_hours_months)
    office_hours_months = add_peak_column(office_hours_months)
    
    livingroom_cooling_rates = calculate_cooling_rates(livingroom_hours_months)
    office_cooling_rates = calculate_cooling_rates(office_hours_months)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Office Analysis", "🏠 Living Room Analysis", "📈 Daily Statistics", "📋 Data Tables"])

with tab1:
    st.plotly_chart(create_temperature_plot(office_cooling_rates, "Office"), use_container_width=True)

with tab2:
    st.plotly_chart(create_temperature_plot(livingroom_cooling_rates, "Living Room"), use_container_width=True)

with tab3:
    # Calculate daily statistics
    office_daily_stats = office_cooling_rates.groupby(office_cooling_rates['datetime'].dt.date)['cooling_rate'].agg(['mean', 'median', 'std']).reset_index()
    office_daily_stats.columns = ['date', 'mean_cooling_rate', 'median_cooling_rate', 'std_cooling_rate']
    
    livingroom_daily_stats = livingroom_cooling_rates.groupby(livingroom_cooling_rates['datetime'].dt.date)['cooling_rate'].agg(['mean', 'median', 'std']).reset_index()
    livingroom_daily_stats.columns = ['date', 'mean_cooling_rate', 'median_cooling_rate', 'std_cooling_rate']
    
    daily_stats = pd.merge(office_daily_stats, livingroom_daily_stats, on='date', suffixes=('_office', '_livingroom'))
    
    # Load Pittsburgh minimum temperature data
    pitt_min_temp = pd.read_csv('data/pitt_min_temp.csv')
    daily_stats['date'] = pd.to_datetime(daily_stats['date'])
    pitt_min_temp['DATE'] = pd.to_datetime(pitt_min_temp['DATE'])
    
    daily_stats = daily_stats.merge(
        pitt_min_temp[['DATE', 'TMIN']], 
        left_on='date', 
        right_on='DATE', 
        how='left'
    )
    
    daily_stats = daily_stats.rename(columns={'TMIN': 'daily_min_temp'})
    daily_stats = daily_stats.drop('DATE', axis=1)
    
    st.plotly_chart(create_daily_stats_plot(daily_stats), use_container_width=True)

with tab4:
    st.subheader("Daily Statistics Summary")
    st.dataframe(daily_stats, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Office Data Sample")
        st.dataframe(office_cooling_rates.head(20), use_container_width=True)
    
    with col2:
        st.subheader("Living Room Data Sample")
        st.dataframe(livingroom_cooling_rates.head(20), use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This app analyzes temperature cooling rates in different rooms "
    "by detecting heating peaks and calculating the rate of temperature "
    "decrease between peaks during overnight hours (midnight to 7 AM)."
)
