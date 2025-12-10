import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for Streamlit Cloud
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Wind Power Forecast Dashboard",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_forecast_data(file_path):
    """Load all sheets from the forecast Excel file"""
    try:
        # Read all sheets
        historical_data = pd.read_excel(file_path, sheet_name='Historical Data')
        lstm_forecast = pd.read_excel(file_path, sheet_name='LSTM Forecast')
        gru_forecast = pd.read_excel(file_path, sheet_name='GRU Forecast')
        cnn_forecast = pd.read_excel(file_path, sheet_name='1D CNN Forecast')
        combined_forecast = pd.read_excel(file_path, sheet_name='Combined Forecast')
        
        # Convert Time columns to datetime
        historical_data['Time'] = pd.to_datetime(historical_data['Time'])
        lstm_forecast['Time'] = pd.to_datetime(lstm_forecast['Time'])
        gru_forecast['Time'] = pd.to_datetime(gru_forecast['Time'])
        cnn_forecast['Time'] = pd.to_datetime(cnn_forecast['Time'])
        combined_forecast['Time'] = pd.to_datetime(combined_forecast['Time'])
        
        return {
            'historical': historical_data,
            'lstm': lstm_forecast,
            'gru': gru_forecast,
            'cnn': cnn_forecast,
            'combined': combined_forecast
        }
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Main header
st.markdown('<h1 class="main-header">🌬️ Wind Power Forecast Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### Three-Model Forecast Comparison & Analysis")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Settings")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload Forecast Excel File", 
    type=['xlsx'],
    help="Upload the wind_power_forecast_combined.xlsx file"
)

# Use uploaded file or default path
if uploaded_file is not None:
    data = load_forecast_data(uploaded_file)
else:
    # Default path - try both deployment and local paths
    default_paths = [
        'wind_power_forecast_combined.xlsx',  # For deployment
        '/Users/changyichun/Desktop/2025 Fall/Data Visualization /wind_power_forecast_combined.xlsx'  # For local
    ]
    
    data = None
    for default_path in default_paths:
        try:
            data = load_forecast_data(default_path)
            st.sidebar.success("✅ File loaded successfully")
            break
        except:
            continue
    
    if data is None:
        st.warning("⚠️ Please upload the forecast Excel file to continue")
        data = None

if data is not None:
    historical_data = data['historical']
    lstm_forecast = data['lstm']
    gru_forecast = data['gru']
    cnn_forecast = data['cnn']
    combined_forecast = data['combined']
    
    # Sidebar filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Display Options")
    
    show_historical = st.sidebar.checkbox("Show Historical Data", value=True)
    historical_window = st.sidebar.slider("Historical Data Window (hours)", 50, 500, 200, 50)
    
    show_confidence = st.sidebar.checkbox("Show Confidence Levels", value=True)
    show_direction = st.sidebar.checkbox("Show Direction Indicators", value=True)
    
    # Model selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Model Selection")
    show_lstm = st.sidebar.checkbox("LSTM", value=True)
    show_gru = st.sidebar.checkbox("GRU", value=True)
    show_cnn = st.sidebar.checkbox("1D CNN", value=True)
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🎯 Model Comparison", "📊 Detailed Forecast", "📋 Data Tables"])
    
    # TAB 1: Overview
    with tab1:
        st.header("📊 Forecast Summary")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("LSTM Model")
            increase_count = (lstm_forecast['Direction'] == 1).sum()
            total = len(lstm_forecast)
            st.metric("Predicted Increase", f"{increase_count}/{total}", f"{increase_count/total*100:.1f}%")
            st.metric("Avg Confidence", f"{lstm_forecast['Confidence'].mean():.2%}")
            st.metric("Avg Power", f"{lstm_forecast['Predicted Power'].mean():.4f}")
        
        with col2:
            st.subheader("GRU Model")
            increase_count = (gru_forecast['Direction'] == 1).sum()
            total = len(gru_forecast)
            st.metric("Predicted Increase", f"{increase_count}/{total}", f"{increase_count/total*100:.1f}%")
            st.metric("Avg Confidence", f"{gru_forecast['Confidence'].mean():.2%}")
            st.metric("Avg Power", f"{gru_forecast['Predicted Power'].mean():.4f}")
        
        with col3:
            st.subheader("1D CNN Model")
            increase_count = (cnn_forecast['Direction'] == 1).sum()
            total = len(cnn_forecast)
            st.metric("Predicted Increase", f"{increase_count}/{total}", f"{increase_count/total*100:.1f}%")
            st.metric("Avg Confidence", f"{cnn_forecast['Confidence'].mean():.2%}")
            st.metric("Avg Power", f"{cnn_forecast['Predicted Power'].mean():.4f}")
        
        st.markdown("---")
        
        # Power forecast comparison chart
        st.subheader("📈 Power Generation Forecast Comparison")
        
        fig, ax = plt.subplots(figsize=(16, 6))
        
        # Plot historical data if selected
        if show_historical:
            hist_recent = historical_data.tail(historical_window)
            ax.plot(hist_recent['Time'], hist_recent['Power'], 
                   linewidth=2, label='Historical Data', color='black', alpha=0.7)
        
        # Plot forecasts
        colors = {'LSTM': '#1f77b4', 'GRU': '#ff7f0e', '1D CNN': '#2ca02c'}
        
        if show_lstm:
            ax.plot(lstm_forecast['Time'], lstm_forecast['Predicted Power'],
                   linewidth=2, label='LSTM Forecast', color=colors['LSTM'], 
                   linestyle='--', marker='o', markersize=4)
        
        if show_gru:
            ax.plot(gru_forecast['Time'], gru_forecast['Predicted Power'],
                   linewidth=2, label='GRU Forecast', color=colors['GRU'], 
                   linestyle='--', marker='s', markersize=4)
        
        if show_cnn:
            ax.plot(cnn_forecast['Time'], cnn_forecast['Predicted Power'],
                   linewidth=2, label='1D CNN Forecast', color=colors['1D CNN'], 
                   linestyle='--', marker='^', markersize=4)
        
        # Mark forecast start
        if show_historical:
            forecast_start = lstm_forecast['Time'].iloc[0]
            ax.axvline(x=forecast_start, color='red', linestyle=':', 
                      linewidth=2, label='Forecast Start', alpha=0.7)
        
        ax.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Power Generation', fontsize=12, fontweight='bold')
        ax.set_title('Wind Power: Historical Data vs Model Forecasts', fontsize=16, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)
    
    # TAB 2: Model Comparison
    with tab2:
        st.header("🎯 Model Comparison Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Prediction direction comparison
            st.subheader("📊 Direction Prediction Comparison")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            x = np.arange(len(lstm_forecast))
            width = 0.25
            
            if show_lstm:
                ax.bar(x - width, lstm_forecast['Direction'], width, 
                      label='LSTM', alpha=0.8, color=colors['LSTM'])
            if show_gru:
                ax.bar(x, gru_forecast['Direction'], width, 
                      label='GRU', alpha=0.8, color=colors['GRU'])
            if show_cnn:
                ax.bar(x + width, cnn_forecast['Direction'], width, 
                      label='1D CNN', alpha=0.8, color=colors['1D CNN'])
            
            ax.set_xlabel('Forecast Step', fontsize=11, fontweight='bold')
            ax.set_ylabel('Direction', fontsize=11, fontweight='bold')
            ax.set_title('Prediction Direction by Step', fontsize=14, fontweight='bold')
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Decrease/Same', 'Increase'])
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            st.pyplot(fig)
        
        with col2:
            # Confidence comparison
            st.subheader("🎯 Confidence Level Comparison")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            if show_lstm:
                ax.plot(lstm_forecast['Step'], lstm_forecast['Confidence'],
                       marker='o', linewidth=2, label='LSTM', color=colors['LSTM'])
            if show_gru:
                ax.plot(gru_forecast['Step'], gru_forecast['Confidence'],
                       marker='s', linewidth=2, label='GRU', color=colors['GRU'])
            if show_cnn:
                ax.plot(cnn_forecast['Step'], cnn_forecast['Confidence'],
                       marker='^', linewidth=2, label='1D CNN', color=colors['1D CNN'])
            
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=2)
            ax.set_xlabel('Forecast Step', fontsize=11, fontweight='bold')
            ax.set_ylabel('Confidence Level', fontsize=11, fontweight='bold')
            ax.set_title('Model Confidence by Step', fontsize=14, fontweight='bold')
            ax.set_ylim([0, 1])
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            st.pyplot(fig)
        
        st.markdown("---")
        
        # Agreement analysis
        st.subheader("🤝 Model Agreement Analysis")
        
        # Calculate agreement
        agreement_data = []
        for i in range(len(combined_forecast)):
            lstm_dir = 'Increase' if lstm_forecast.iloc[i]['Direction'] == 1 else 'Decrease/Same'
            gru_dir = 'Increase' if gru_forecast.iloc[i]['Direction'] == 1 else 'Decrease/Same'
            cnn_dir = 'Increase' if cnn_forecast.iloc[i]['Direction'] == 1 else 'Decrease/Same'
            
            agreement = sum([lstm_dir == gru_dir, lstm_dir == cnn_dir, gru_dir == cnn_dir])
            all_agree = (lstm_dir == gru_dir == cnn_dir)
            
            agreement_data.append({
                'Step': i + 1,
                'LSTM': lstm_dir,
                'GRU': gru_dir,
                '1D CNN': cnn_dir,
                'Agreement Score': agreement,
                'All Agree': 'Yes' if all_agree else 'No'
            })
        
        agreement_df = pd.DataFrame(agreement_data)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            total_steps = len(agreement_df)
            all_agree_count = (agreement_df['All Agree'] == 'Yes').sum()
            agreement_pct = all_agree_count / total_steps * 100
            
            st.metric("Total Steps", total_steps)
            st.metric("All Models Agree", all_agree_count, f"{agreement_pct:.1f}%")
            
            # Agreement distribution
            fig, ax = plt.subplots(figsize=(6, 4))
            agreement_counts = agreement_df['All Agree'].value_counts()
            ax.pie(agreement_counts, labels=agreement_counts.index, autopct='%1.1f%%',
                  colors=['#ff9999', '#66b3ff'], startangle=90)
            ax.set_title('Model Agreement Distribution', fontsize=12, fontweight='bold')
            st.pyplot(fig)
        
        with col2:
            st.dataframe(
                agreement_df.style.apply(
                    lambda x: ['background-color: lightgreen' if v == 'Yes' else '' 
                              for v in x], 
                    subset=['All Agree']
                ),
                height=400,
                use_container_width=True
            )
    
    # TAB 3: Detailed Forecast
    with tab3:
        st.header("📊 Detailed Forecast Visualization")
        
        # Model selector for detailed view
        selected_model = st.selectbox(
            "Select Model for Detailed View",
            options=['LSTM', 'GRU', '1D CNN', 'All Models']
        )
        
        if selected_model == 'All Models':
            # Show combined view
            st.subheader("Combined Forecast Comparison")
            
            fig, axes = plt.subplots(2, 1, figsize=(16, 10))
            
            # Power values
            ax1 = axes[0]
            ax1.plot(combined_forecast['Step'], combined_forecast['LSTM_Power'],
                    marker='o', linewidth=2, label='LSTM', color=colors['LSTM'])
            ax1.plot(combined_forecast['Step'], combined_forecast['GRU_Power'],
                    marker='s', linewidth=2, label='GRU', color=colors['GRU'])
            ax1.plot(combined_forecast['Step'], combined_forecast['1D CNN_Power'],
                    marker='^', linewidth=2, label='1D CNN', color=colors['1D CNN'])
            ax1.set_xlabel('Forecast Step', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Predicted Power', fontsize=12, fontweight='bold')
            ax1.set_title('Power Predictions Comparison', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Confidence levels
            ax2 = axes[1]
            ax2.plot(combined_forecast['Step'], combined_forecast['LSTM_Confidence'],
                    marker='o', linewidth=2, label='LSTM', color=colors['LSTM'])
            ax2.plot(combined_forecast['Step'], combined_forecast['GRU_Confidence'],
                    marker='s', linewidth=2, label='GRU', color=colors['GRU'])
            ax2.plot(combined_forecast['Step'], combined_forecast['1D CNN_Confidence'],
                    marker='^', linewidth=2, label='1D CNN', color=colors['1D CNN'])
            ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=2)
            ax2.set_xlabel('Forecast Step', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Confidence Level', fontsize=12, fontweight='bold')
            ax2.set_title('Confidence Levels Comparison', fontsize=14, fontweight='bold')
            ax2.set_ylim([0, 1])
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
        else:
            # Show individual model
            if selected_model == 'LSTM':
                forecast_data = lstm_forecast
                color = colors['LSTM']
            elif selected_model == 'GRU':
                forecast_data = gru_forecast
                color = colors['GRU']
            else:
                forecast_data = cnn_forecast
                color = colors['1D CNN']
            
            st.subheader(f"{selected_model} Model Detailed Forecast")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(forecast_data['Step'], forecast_data['Predicted Power'],
                       marker='o', linewidth=2, color=color)
                ax.set_xlabel('Forecast Step', fontsize=11, fontweight='bold')
                ax.set_ylabel('Predicted Power', fontsize=11, fontweight='bold')
                ax.set_title(f'{selected_model} Power Predictions', fontsize=13, fontweight='bold')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(10, 5))
                bars = ax.bar(forecast_data['Step'], forecast_data['Confidence'],
                             color=[color if x == 1 else '#cccccc' 
                                   for x in forecast_data['Direction']],
                             alpha=0.7)
                ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=2)
                ax.set_xlabel('Forecast Step', fontsize=11, fontweight='bold')
                ax.set_ylabel('Confidence', fontsize=11, fontweight='bold')
                ax.set_title(f'{selected_model} Confidence Levels', fontsize=13, fontweight='bold')
                ax.set_ylim([0, 1])
                ax.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                st.pyplot(fig)
            
            # Detailed statistics
            st.markdown("---")
            st.subheader(f"{selected_model} Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean Power", f"{forecast_data['Predicted Power'].mean():.4f}")
            with col2:
                st.metric("Std Power", f"{forecast_data['Predicted Power'].std():.4f}")
            with col3:
                st.metric("Mean Confidence", f"{forecast_data['Confidence'].mean():.2%}")
            with col4:
                increase_pct = (forecast_data['Direction'] == 1).sum() / len(forecast_data) * 100
                st.metric("Increase %", f"{increase_pct:.1f}%")
    
    # TAB 4: Data Tables
    with tab4:
        st.header("📋 Forecast Data Tables")
        
        # Table selector
        table_view = st.selectbox(
            "Select Data View",
            options=['Combined Forecast', 'LSTM Forecast', 'GRU Forecast', 
                    '1D CNN Forecast', 'Historical Data']
        )
        
        if table_view == 'Combined Forecast':
            st.dataframe(
                combined_forecast.style.background_gradient(
                    cmap='RdYlGn',
                    subset=['LSTM_Confidence', 'GRU_Confidence', '1D CNN_Confidence']
                ),
                use_container_width=True,
                height=600
            )
            
            # Download button
            csv = combined_forecast.to_csv(index=False)
            st.download_button(
                label="📥 Download Combined Forecast (CSV)",
                data=csv,
                file_name=f"combined_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
        elif table_view == 'LSTM Forecast':
            st.dataframe(
                lstm_forecast.style.background_gradient(cmap='RdYlGn', subset=['Confidence']),
                use_container_width=True,
                height=600
            )
        elif table_view == 'GRU Forecast':
            st.dataframe(
                gru_forecast.style.background_gradient(cmap='RdYlGn', subset=['Confidence']),
                use_container_width=True,
                height=600
            )
        elif table_view == '1D CNN Forecast':
            st.dataframe(
                cnn_forecast.style.background_gradient(cmap='RdYlGn', subset=['Confidence']),
                use_container_width=True,
                height=600
            )
        else:
            st.dataframe(
                historical_data.tail(500),
                use_container_width=True,
                height=600
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Wind Power Forecast Dashboard | Built with Streamlit 🎈</p>
        <p>Models: LSTM, GRU, 1D CNN | Three-Model Comparison Analysis</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("👆 Please upload the forecast Excel file to view the dashboard")
    
    st.markdown("### 📌 Instructions:")
    st.markdown("""
    1. Upload your `wind_power_forecast_combined.xlsx` file using the sidebar
    2. The file should contain the following sheets:
       - Historical Data
       - LSTM Forecast
       - GRU Forecast
       - 1D CNN Forecast
       - Combined Forecast
    3. Explore the different tabs for various visualizations and analyses
    """)
