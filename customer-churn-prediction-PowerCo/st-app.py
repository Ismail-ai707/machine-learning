import streamlit as st
# import requests
from streamlit_lottie import st_lottie
import json
import io

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from datetime import datetime

import altair as alt
import time
import zipfile

# Page title
st.set_page_config(page_title='ML Project - Churn Prediction', page_icon='🚀')
st.title('🚀 Customer Churn Prediction for PowerCo')

# Function to load Lottie animation from a JSON file
def load_lottiefile(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)
    
# Load the Lottie animation from the file
lottie_animation = load_lottiefile('DSProject-animation.json')

st_lottie(lottie_animation, speed=1, height=500, key="animation")

with st.expander('**About this project**'):
  st.markdown('**Project Description**')
  st.info('PowerCo is a major gas and electricity utility that supplies to small and medium sized enterprises. As a data scientist I am willing to help PowerCo diagnose why it’s customers are churning by analysing data and making effective predictions. This client states that price sensitivity might be the reason why customers leave. In the following analysis, we will verify this hypothesis in order to provide PowerCo with suitable answers')

  st.markdown('**What can this app do?**')
  st.info('This app')

  st.markdown('**How to use the app?**')
  st.warning('To engage with the app, go to the sidebar and 1. Select a data set and 2. Adjust the model parameters by adjusting the various slider widgets. As a result, this would initiate the ML model building process, display the model results as well as allowing users to download the generated models and accompanying data.')
  st.markdown('**Data sets**')
  st.code('''- Client data set : customer historical data such as usage, sign up date, forecasted usage etc.
- Price data set : historical pricing data such as variable and fixed pricing data etc.
  ''', language='markdown')

#   st.file_uploader('Please upload your file here')
  with open('./Data_Description.pdf', 'rb') as f:
      st.download_button('Download Data Description', f, file_name='Data_Description.pdf', mime='application/pdf')

  st.markdown('**Libraries used**')
  st.code('''- Pandas for data wrangling
- Scikit-learn for building and evaluating a machine learning model
- Seaborn and matplotlib for data visualization
- Streamlit for user interface
  ''', language='markdown')

# My GitHub and LinkedIn profile URLs
github_url = "https://github.com/Ismail-ai707"
linkedin_url = "https://www.linkedin.com/in/rhazi-ismail"

# Create the markdown content with your name and links
markdown_content = f"""
<p style='text-align: right; font-size: 18px;'>Project by Ismail Rhazi</p>
<p style='text-align: right;'>
    Connect with me on <a href="{linkedin_url}" target="_blank">LinkedIn</a>
</p>
"""
# Display the markdown content
st.markdown(markdown_content, unsafe_allow_html=True)

''' - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - '''

# Define some functions that will be used by this app

def annotate_stacked_bars(ax, pad=0.99, colour="white", textsize=8):
    """
    Add value annotations to the bars
    """
    annotations = []

    # Iterate over the plotted rectangles/bars
    for p in ax.patches:
        # Calculate annotation
        value = str(round(p.get_height(), 1))
        # If value is 0 do not annotate
        if value == '0.0':
            continue
        annotation = ax.annotate(
            value,
            ((p.get_x() + p.get_width() / 2) * pad - 0.05, (p.get_y() + p.get_height() / 2) * pad),
            color=colour,
            size=textsize
        )
        annotations.append(annotation)

    return annotations

def plot_stacked_bars(dataframe, title_, size_=(18, 10), rot_=0, legend_="upper right"):
    """
    Plot stacked bars with annotations
    """
    fig, ax = plt.subplots(figsize=size_)

    dataframe.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        rot=rot_,
        title=title_
    )

    # Annotate bars
    annotations = annotate_stacked_bars(ax, textsize=18)
    # Rename legend
    ax.legend(["Retention", "Churn"], loc=legend_)
    # Labels
    ax.set_ylabel("Clients %")

    # Display the Matplotlib figure in Streamlit
    st.pyplot(fig)

# def display_data_types_info(dataframe):
#     # Obtenir les types de données pour chaque variable
#     data_types_info = dataframe.dtypes.reset_index()
#     data_types_info.columns = ['Variable', 'Type']

#     # Afficher les informations dans Streamlit
#     #st.write("#### Informations sur les types de données:")
#     st.table(data_types_info)

def plot_distribution(dataframe, column, bins_=50):
    """
    Plot variable distirbution in a stacked histogram of churned or retained company
    """
    # Create a temporal dataframe with the data to be plot
    temp = pd.DataFrame({"Retention": dataframe[dataframe["churn"]==0][column],
                         "Churn":dataframe[dataframe["churn"]==1][column]})

    # Plot the histogram
    fig, ax = plt.subplots(figsize=(8, 10))
    temp[["Retention","Churn"]].plot(kind='hist', bins=bins_, ax=ax, stacked=True)
    # X-axis label
    ax.set_xlabel(column)
    # Change the x-axis to plain style
    ax.ticklabel_format(style='plain', axis='x')
    st.pyplot(fig)

# A function that i'm using to plot variables density for each group previously defined
def density_plot(data):
    """
    Add function description here
    """
    plt.figure(figsize=(15, 10))
    sns.set_style('whitegrid')

    for i, var in enumerate(data.columns[:-1], 1):
        plt.subplot(3, 3, i)
        sns.histplot(data=data, x=var, hue='churn', kde=True, element='step', stat='density', common_norm=False)
        plt.title(f'Distribution of {var} by Churn')
        if var=='cons_last_month':
            plt.xticks(rotation=40)

    plt.tight_layout()
    plt.show()
    st.pyplot(plt)

def group_boxenplot(dataframe):
    """
    Add function description here
    """
    columns_to_plot = [col for col in dataframe.columns[:-1] if col != 'forecast_discount_energy']
    fig, axs = plt.subplots(nrows=len(columns_to_plot), figsize=(18, 20))
    for i, c in enumerate(columns_to_plot):
        sns.boxenplot(x=dataframe[c], ax=axs[i])
        axs[i].set_xlabel(c, fontsize=12)
        
        # Calculate and set the quantiles as ticks
        quantiles = dataframe[c].quantile([0.25, 0.5, 0.75, 0.95]).values
        axs[i].set_xticks(quantiles)
        axs[i].set_xticklabels([f'{q:.2f}' for q in quantiles], fontsize=10)

        # Add gridlines for the quantiles
        for q in quantiles:
            axs[i].axvline(q, color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()
    st.pyplot(plt)

def heatmap_plot(correlation_matrix):
    """
    Add function description here
    """
    plt.figure(figsize=(25, 15))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix')

    plt.tight_layout()
    plt.show()
    st.pyplot(plt)

# Variable price vs forecasted price during off-peak and peak periods according to reference date
def lineplot_ref_comp(dataframe, y1, y2):
    """
    Add function description here
    """
    plt.figure(figsize=(15,5))
    sns.lineplot(data=dataframe, x='price_date', y=y1, label=y1)
    sns.lineplot(data=dataframe, x='price_date', y=y2, label=y2)
    st.pyplot(plt)

def lintplot_year_comp(dataframe, y1, y2):
    """
    Add function description here
    """
    plt.figure(figsize=(15,5))
    sns.lineplot(data=dataframe, x=dataframe['date_activ'].dt.year, y=y1, label=y1)
    sns.lineplot(data=dataframe, x=dataframe['date_activ'].dt.year, y=y2, label=y2)
    st.pyplot(plt)

def convert_months(reference_date, df, column):
    """
    Input a column with timedeltas and return months
    """
    time_delta = reference_date - df[column]
    months = (time_delta.dt.days / (365.25 / 12)).astype(int)
    return months

def feature_importances_plot(feature_importances):
    """
    This function plots feature importances from a dataframe
    The dataframe contains training data and importances calculated from an ensemble-based model
    """
    plt.figure(figsize=(15, 20))
    plt.title('Feature Importance')
    plt.barh(range(len(feature_importances)), feature_importances['importance'], color='cadetblue', align='center')
    plt.yticks(range(len(feature_importances)), feature_importances['features'])
    plt.xlabel('Importance')

    sns.set_style('whitegrid')
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)

def get_dataframe_info(df):
    """
    Capture the output of pd.DataFrame.info() as a string.

    Parameters:
    df (pd.DataFrame): The DataFrame for which to capture the info.

    Returns:
    str: A string containing the output of df.info().
    """
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    return info_str
''' - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - '''
# Sidebar for accepting input parameters

st.sidebar.title("Menu")
with st.sidebar:
    st.markdown('<p style="font-size:20px; font-weight:bold;">1. Input Data</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:20px; font-weight:bold;">2. Exploratory Data Analyis</p>', unsafe_allow_html=True)
    # eda_button = st.sidebar.button('Launch EDA')
    st.markdown('<p style="font-size:20px; font-weight:bold;">3. Feature Engineering</p>', unsafe_allow_html=True)
    # feature_eng_button = st.sidebar.button('See Engineered Features')
    
    st.markdown('<p style="font-size:20px; font-weight:bold;">4. Predict Churn using Machine Learning</p>', unsafe_allow_html=True)
    st.header('4.1. Split Data')
    parameter_split_size = st.slider('Data split ratio (% of Test Set)', 10, 50, 25, 5)

    st.subheader('4.1.1. Detailed Parameters')
    with st.expander('Have fun with model´s parameters'):
        n_estimators = st.slider('Number of estimators', 1, 1000, 100, step=10)
        random_state = st.slider('Random state for the algorithm', 0, 1000, 42, step=10)
        bootstrap = st.select_slider('Bootstramp samples', options=[True, False])
        ccp_alpha = st.select_slider('Use Cost Complexity Pruning', options=[0.0, 0.001, 0.01, 0.1, 1.0])
        class_weight = st.select_slider('Class weights', options=[None, 'balanced'])
        criterion = st.select_slider('Quality of split', options=['gini', 'entropy'])
        max_depth = st.select_slider('Max depth of the tree', options=[None, 1, 3, 5, 10])
        max_features = st.select_slider('Number of features to consider for split', options=[None, 'sqrt', 'log2'])
        max_leaf_nodes = st.select_slider('Max leaf nodes', options=[None, 5, 10, 100])
        max_samples = st.select_slider('Max bootstramp samples', options=[None, 0.5, 10, 100])
        min_impurity_decrease = st.select_slider('Min impurity to split node', options=[0.0, 0.01, 0.1])
        min_samples_leaf = st.select_slider('Min samples to be at leaf node', options=[1, 5, 10])
        min_samples_split = st.select_slider('Min samples to split an internal node', options=[2, 5, 10])
        min_weight_fraction_leaf = st.select_slider('Min weighted fraction to be at leaf node', options=[0.0, 0.01, 0.05, 0.5])
        n_jobs = st.select_slider('Number of jobs to run in parallel', options=[-1, None, 1])
        oob_score = st.select_slider('Use of out-of-bag samples', options=[False, True])
        verbose = st.select_slider('Control verbosity when fitting and predicting', options=[0, 1, 2, 3])
        warm_start = st.select_slider('Re-use last fit', options=[True, False])

# By default show the user how to load data
client_df = None
price_df = None

# -----------------------------Load Data to start --------------------------
st.markdown("<h1 style='font-size:32px;'>Load the data</h1>", unsafe_allow_html=True)
# Cache the function that loads data
@st.cache_data
def load_sample_data(file_path):
    return pd.read_csv(file_path)

@st.cache_data
def load_uploaded_data(uploaded_file):
    return pd.read_csv(uploaded_file, index_col=False)

with st.expander('Import Data to start'):
    # Load data
    st.markdown('**1.1 Download & Explore Sample Data**')        
    with open('./client_data.csv', 'rb') as client_file:
        st.download_button(
            label="Download Client example CSV",
            data=client_file,
            file_name='client_data.csv',
            mime='text/csv',
        )
    with open('./price_data.csv', 'rb') as price_file:
        st.download_button(
            label="Download Price example CSV",
            data=price_file,
            file_name='price_data.csv',
            mime='text/csv',
        )
    # Select example data
    st.markdown('**1.2.1 Use Sample data**')
    example_data = st.toggle('Load example data')
    if example_data:
        client_df = load_sample_data('./client_data.csv')
        price_df = load_sample_data('./price_data.csv')

    st.markdown('**1.2.2 Use Custom Data**')
    client_data_file = st.file_uploader("Upload Client CSV data file", type=["csv"])
    if client_data_file is not None:
        client_df = load_uploaded_data(client_data_file)

    price_data_file = st.file_uploader("Upload Price CSV data file", type=["csv"])
    if price_data_file is not None:
        price_df = load_uploaded_data(price_data_file)

''' - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - '''
# Function to perform preprocessing
@st.cache_data
def perform_feature_engineering(client_df, price_df):
    # Convert date columns to datetime
    client_df['date_activ'] = pd.to_datetime(client_df['date_activ'])
    client_df['date_end'] = pd.to_datetime(client_df['date_end'])
    client_df['date_modif_prod'] = pd.to_datetime(client_df['date_modif_prod'])
    client_df['date_renewal'] = pd.to_datetime(client_df['date_renewal'])
    
    price_df['price_date'] = pd.to_datetime(price_df['price_date'])
    
    # Example data wrangling steps
    client_info = get_dataframe_info(client_df)
    price_info = get_dataframe_info(price_df)
    
    return client_df, price_df, client_info, price_info

# Check if both dataframes are loaded before proceeding with data wrangling
if client_df is not None and price_df is not None:
    client_df, price_df, client_info, price_info = perform_feature_engineering(client_df, price_df)
    
    # Optional: Display DataFrames for verification
    st.write('Client DataFrame:')
    st.write(client_df.head())
    st.write('Price DataFrame:')
    st.write(price_df.head())

    with st.expander('Preprocessing'):
        st.write('Client data info after converting data types :')
        st.text(client_info)
        st.write('Price data info after converting data types :')
        st.text(price_info)
        st.write('Client data description :')
        st.write(client_df.describe())
        st.write('Price data description :')
        st.write(price_df.describe())
        st.write('''client_df observations:
- Consumption (energy & gas) present high variability.
- The minimum of many metrics, including consumption, is 0. This suggests that there are some periods with no consumption or activity for some customers.
- Forcasted consumption metrics are much lower on average compared to actual consumption, which might suggest under-forecasting or changes in consumption patterns.
- The average churn rate is around 9.7% which is relatively low churn rate but with significant variation that we should pay attention to.''')
        st.write('''price_df observations:
- Off-peak prices (for both variable and fix prices) have the highest mean and median.
- price_mid_peak_var, price_peak_fix and price_mid_peak_fix have a median of zero. This suggests that at least half of the observations are zero which indicates a skewness towards zero.''')
        
    consumption = client_df[['cons_12m', 'cons_gas_12m', 'cons_last_month', 'pow_max', 'imp_cons', 'churn']]
    forecast = client_df[['forecast_cons_12m', 'forecast_cons_year', 'forecast_discount_energy', 'forecast_meter_rent_12m', 'forecast_price_energy_off_peak', 'churn']]
    margin = client_df[['margin_gross_pow_ele', 'margin_net_pow_ele', 'net_margin', 'churn']]
    clients_details = client_df[['id', 'channel_sales', 'nb_prod_act', 'num_years_antig', 'origin_up', 'churn']]

    mean_year = price_df.groupby(['id']).mean().reset_index()
    mean_6m = price_df[price_df['price_date'] > '2015-06-01'].groupby(['id']).mean().reset_index()

    ## Mean price by year and by 6 months
    mean_year = mean_year.rename(
        index=str,
    columns={
                        "price_off_peak_var": "mean_year_price_off_peak_var",
                        "price_peak_var": "mean_year_price_peak_var",
                        "price_mid_peak_var": "mean_year_price_mid_peak_var",
                        "price_off_peak_fix": "mean_year_price_off_peak_fix",
                        "price_peak_fix": "mean_year_price_peak_fix",
                        "price_mid_peak_fix": "mean_year_price_mid_peak_fix"
                    }
                )

    mean_year["mean_year_price_off_peak"] = mean_year["mean_year_price_off_peak_var"] + mean_year["mean_year_price_off_peak_fix"]
    mean_year["mean_year_price_peak"] = mean_year["mean_year_price_peak_var"] + mean_year["mean_year_price_peak_fix"]
    mean_year["mean_year_price_mid_peak"] = mean_year["mean_year_price_mid_peak_var"] + mean_year["mean_year_price_mid_peak_fix"]


    mean_6m = mean_6m.rename(
                    index=str,
                    columns={
                        "price_off_peak_var": "mean_6m_price_off_peak_var",
                        "price_peak_var": "mean_6m_price_peak_var",
                        "price_mid_peak_var": "mean_6m_price_mid_peak_var",
                        "price_off_peak_fix": "mean_6m_price_off_peak_fix",
                        "price_peak_fix": "mean_6m_price_peak_fix",
                        "price_mid_peak_fix": "mean_6m_price_mid_peak_fix"
                    }
                )

    mean_6m["mean_6m_price_off_peak"] = mean_6m["mean_6m_price_off_peak_var"] + mean_6m["mean_6m_price_off_peak_fix"]
    mean_6m["mean_6m_price_peak"] = mean_6m["mean_6m_price_peak_var"] + mean_6m["mean_6m_price_peak_fix"]
    mean_6m["mean_6m_price_mid_peak"] = mean_6m["mean_6m_price_mid_peak_var"] + mean_6m["mean_6m_price_mid_peak_fix"]

    price_features = pd.merge(mean_year, mean_6m, on='id')
    price_features.drop(columns=['price_date_x', 'price_date_y'], inplace=True)
    merged_df = pd.merge(client_df, price_features, on='id')

    ## Off-peak prices by companies and month
    monthly_mean_price = price_df.groupby(['id', 'price_date']).agg({'price_off_peak_var': 'mean', 'price_off_peak_fix': 'mean'}).reset_index()

    jan_prices_off_peak = monthly_mean_price.sort_values('price_date').groupby('id').first().reset_index()
    dec_prices = monthly_mean_price.sort_values('price_date').groupby('id').last().reset_index()

    # Difference between off-peak price in december and in preceeding january
    diff_off_peak = pd.merge(dec_prices, jan_prices_off_peak, on='id', suffixes=('_dec', '_jan'))
    diff_off_peak['offpeak_price_diff_dec_jan_energy'] = diff_off_peak['price_off_peak_var_dec'] - diff_off_peak['price_off_peak_var_jan']
    diff_off_peak['offpeak_price_diff_dec_jan_power'] = diff_off_peak['price_off_peak_fix_dec'] - diff_off_peak['price_off_peak_fix_jan']
    diff_off_peak = diff_off_peak[['id', 'offpeak_price_diff_dec_jan_energy', 'offpeak_price_diff_dec_jan_power']]

    # Peak prices by companies and month
    monthly_mean_price = price_df.groupby(['id', 'price_date']).agg({'price_peak_var': 'mean', 'price_peak_fix': 'mean'}).reset_index()

    jan_prices_peak = monthly_mean_price.sort_values('price_date').groupby('id').first().reset_index()
    dec_prices = monthly_mean_price.sort_values('price_date').groupby('id').last().reset_index()

    # Difference between peak price in december and in preceeding january
    diff_peak = pd.merge(dec_prices, jan_prices_peak, on='id', suffixes=('_dec', '_jan'))
    diff_peak['peak_price_diff_dec_jan_energy'] = diff_peak['price_peak_var_dec'] - diff_peak['price_peak_var_jan']
    diff_peak['peak_price_diff_dec_jan_power'] = diff_peak['price_peak_fix_dec'] - diff_peak['price_peak_fix_jan']
    diff_peak = diff_peak[['id', 'peak_price_diff_dec_jan_energy', 'peak_price_diff_dec_jan_power']]

    # Mid-peak prices by companies and month
    monthly_mean_price = price_df.groupby(['id', 'price_date']).agg({'price_mid_peak_var': 'mean', 'price_mid_peak_fix': 'mean'}).reset_index()

    jan_prices_mid_peak = monthly_mean_price.sort_values('price_date').groupby('id').first().reset_index()
    dec_prices = monthly_mean_price.sort_values('price_date').groupby('id').last().reset_index()

    # Difference between mid-peak price in december and in preceeding january
    diff_mid_peak = pd.merge(dec_prices, jan_prices_mid_peak, on='id', suffixes=('_dec', '_jan'))
    diff_mid_peak['midPeak_price_diff_dec_jan_energy'] = diff_mid_peak['price_mid_peak_var_dec'] - diff_mid_peak['price_mid_peak_var_jan']
    diff_mid_peak['midPeak_price_diff_dec_jan_power'] = diff_mid_peak['price_mid_peak_fix_dec'] - diff_mid_peak['price_mid_peak_fix_jan']
    diff_mid_peak = diff_mid_peak[['id', 'midPeak_price_diff_dec_jan_energy', 'midPeak_price_diff_dec_jan_power']]

    # Create reference date
    reference_date = datetime(2016, 1, 1)

    ## Create monthly columns
    merged_df['months_activ'] = convert_months(reference_date, merged_df, 'date_activ')
    merged_df['months_end'] = convert_months(reference_date, merged_df, 'date_end')
    merged_df['months_modif_prod'] = convert_months(reference_date, merged_df, 'date_modif_prod')
    merged_df['months_renewal'] = convert_months(reference_date, merged_df, 'date_renewal')

    # Dealing with has_gas
    ## To resolve the Downcasting behavior in `replace` i had to add the set_option and inter_objects() lines
    pd.set_option('future.no_silent_downcasting', True)
    merged_df['has_gas'] = merged_df['has_gas'].replace(['t', 'f'], [1, 0])
    merged_df['has_gas'] = merged_df['has_gas'].infer_objects()
    # merged_df.to_csv('./cleaned_data_after_eda.csv')

    # Dealing with channel sales
    df = pd.get_dummies(merged_df, columns=['channel_sales'], prefix='channel')
    df = df.drop(columns=['channel_sddiedcslfslkckwlfkdpoeeailfpeds', 'channel_epumfxlbckeskwekxbiuasklxalciiuu', 'channel_fixdbufsefwooaasfcxdxadsiekoceaa'])

    dummy_columns = df.filter(like='channel_').columns
    df[dummy_columns] = df[dummy_columns].astype(int)

    # Dealing with origin_up
    df = pd.get_dummies(df, columns=['origin_up'], prefix='origin_up')
    df = df.drop(columns=['origin_up_MISSING', 'origin_up_usapbepcfoloekilkwsdiboslwaxobdp', 'origin_up_ewxeelcelemmiwuafmddpobolfuxioce'])

    # Drop 'price_date_x', 'price_date_y' columns only if they exist in the DataFrame
    columns_to_drop = ['price_date_x', 'price_date_y']
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=existing_columns_to_drop)

    dummy_columns = df.filter(like='origin_').columns
    df[dummy_columns] = df[dummy_columns].astype(int)

    # Dealing with highly skewed data
    df["cons_12m"] = np.log10(df["cons_12m"] + 1)
    df["cons_gas_12m"] = np.log10(df["cons_gas_12m"] + 1)
    df["cons_last_month"] = np.log10(df["cons_last_month"] + 1)
    df["forecast_cons_12m"] = np.log10(df["forecast_cons_12m"] + 1)
    df["forecast_cons_year"] = np.log10(df["forecast_cons_year"] + 1)
    df["forecast_meter_rent_12m"] = np.log10(df["forecast_meter_rent_12m"] + 1)
    df["imp_cons"] = np.log10(df["imp_cons"] + 1)

    st.markdown("<h1 style='font-size:32px;'>Exploratory Data Analysis</h1>", unsafe_allow_html=True)
    ## churn rate by company id
    churn = merged_df[['id', 'churn']]
    churn.columns = ['Companies', 'churn']

    churn_total = churn.groupby(churn['churn']).count()
    churn_percentage = churn_total / churn_total.sum() * 100

    ## churn and retention percentages by sales channel
    grouped_channels = merged_df.groupby(['channel_sales', 'churn']).size().unstack(fill_value=0)
    grouped_channels = grouped_channels.div(grouped_channels.sum(axis=1), axis=0) * 100
    grouped_channels = grouped_channels.sort_values(by=[1], ascending=False)
    grouped_channels.columns = ['Retention', 'Churn']

    ## Correlation matrix
    numerical_data = merged_df.select_dtypes(['float64', 'int64'])
    correlation_matrix = numerical_data.corr()
    with st.expander('Explore some data here'):
        st.write('First, let´s plot data by group : consumption data, forecasted data and margin data.')
        st.markdown(
            """
            <div style="display: flex; justify-content: center;">
                <h3 style='font-size:16px;'>Consumption data densities</h3>
            </div>
            """, 
            unsafe_allow_html=True
        )
        density_plot(consumption)
        st.markdown(
            """
            <div style="display: flex; justify-content: center;">
                <h3 style='font-size:16px;'>Forecasted data densities</h3>
            </div>
            """, 
            unsafe_allow_html=True
        )
        density_plot(forecast)
        st.markdown(
            """
            <div style="display: flex; justify-content: center;">
                <h3 style='font-size:16px;'>Margin data densities</h3>
            </div>
            """, 
            unsafe_allow_html=True
        )
        density_plot(margin)
        st.write('It is important to notice that most of the variables are highly positively skewed. This skewness suggests the presence of outliers and must be corrected before building any machine learning model. We will deal with this in Feature Engineering section.')
        st.write('One of the best ways to visualize outliers is to use boxplots, let´s see what they will show us. To do this, we are using boxenplot function instead of boxplot, boxenplot can show us additional quantiles by creating a “pyramid” effect and is useful for visualizing deeper distribution layers in data with long tails (which is the case for many of our features).\n')

        st.markdown(
            """
            <div style="display: flex; justify-content: center;">
                <h3 style='font-size:16px;'>Consumption data quantiles</h3>
            </div>
            """, 
            unsafe_allow_html=True
        )
        group_boxenplot(consumption)      
        st.markdown(
            """
            <div style="display: flex; justify-content: center;">
                <h3 style='font-size:16px;'>Forecasted data quantiles</h3>
            </div>
            """, 
            unsafe_allow_html=True
        )
        group_boxenplot(forecast)
        st.markdown(
            """
            <div style="display: flex; justify-content: center;">
                <h3 style='font-size:16px;'>Margin data quantiles</h3>
            </div>
            """, 
            unsafe_allow_html=True
        )
        group_boxenplot(margin)
        st.write('In red, vertical lines are added at 25%, 50%, 75% and 95% quantiles. Using these plots we identified outiliers that we should be careful about before developing our predictive model.')
        
        st.markdown(
            """
            <div style="display: flex; justify-content: center;">
                <h3 style='font-size:17px;'>Churn Rate</h3>
            </div>
            """, 
            unsafe_allow_html=True
        )
        plot_stacked_bars(churn_percentage.transpose(), "Churning status", (4, 4), legend_="lower right")
        st.write("The average churn rate is around 9.7% which is a relatively low churn rate, but with significant variation that we should pay attention to (have a look on descriptive statistics in the preprocessing section).")
        temp_df = client_df.merge(price_df, on='id')

        st.markdown(
            """
            <div style="display: flex; justify-content: center;">
                <h3 style='font-size:15px;'>Comparing variable price to forecasted price during off-peak and peak periods</h3>
            </div>
            """, 
            unsafe_allow_html=True
        )
        lineplot_ref_comp(dataframe=temp_df, y1='price_off_peak_var', y2='forecast_price_energy_off_peak')
        lineplot_ref_comp(dataframe=temp_df, y1='price_peak_var', y2='forecast_price_energy_peak')

        st.markdown('''
        During both off-peak and peak periods, the price of energy is greater than forecasted price. Here are some assumptions we might have :

        * Forecast Inaccuracy: The forecasting model may need adjustments to predict suitable prices for customers.
        * Customer Impact: Higher prices than expected might affect customer satisfaction and could increase churn.
        * Price Sensitivity: We need to analyze patterns such as consumption to understand price sensitivity.
        ''')
        st.markdown(
            """
            <div style="display: flex; justify-content: center;">
                <h3 style='font-size:15px;'>Comparing 12m consumption to forecasted 12m consumption during reference date</h3>
            </div>
            """, 
            unsafe_allow_html=True
        )
        lineplot_ref_comp(temp_df, 'cons_12m', 'forecast_cons_12m')
        st.markdown('It seems like the consumption isn´t impacted by the variability of energy price. We will validate this using machine learning.')
        st.markdown(
            """
            <div style="display: flex; justify-content: center;">
                <h3 style='font-size:15px;'>Price variability over the years by activation dates</h3>
            </div>
            """, 
            unsafe_allow_html=True
        )
        lintplot_year_comp(temp_df, 'price_off_peak_var', 'forecast_price_energy_off_peak')
        lintplot_year_comp(temp_df, 'price_peak_var', 'forecast_price_energy_peak')
        st.write('We can realize that in 2007 PowerCo proposed the lowest prices.')
        st.write('And in 2013 predictions seem to be diverging from real price data.')
    
    st.markdown("<h1 style='font-size:32px;'>Feature Engineering</h1>", unsafe_allow_html=True)
    with st.expander('Some features to better explore data'):
        st.write('To get a better understanding of price variability, we defined mean price by year and by every 6 months based on variable and fix prices for each period (off-peak, peak and mid-peak periods)')
        st.write(price_features.head(10))
        st.write('Then, we combined both client data and price data using common IDs for both datasets. This allowed us to get a merged dataset for which you can find a sample bellow.')
        st.write(merged_df.sample(10))
        st.write('Once we had client and price data together, we used monthly prices to calculate the price difference between december and preceeding january for the three periods.')
        st.write('Off-peak price difference between Dec and pre-Jan : ')
        st.write(diff_off_peak.head(10))
        st.write('Peak price difference between Dec and pre-Jan : ')
        st.write(diff_peak.head(10))
        st.write('Mid-peak price difference between Dec and pre-Jan : ')
        st.write(diff_mid_peak.head(10))

        st.write('In order to give client date more meaning, we focused on monthly dates for contract activation, modification of product, contract renewal and contract end.')
        st.write(merged_df[['months_activ', 'months_end', 'months_modif_prod', 'months_renewal']].head(10))

        st.write('And of course, we handeled categorical data by creating dummies that will be used in our machine learning model.')
        st.write(df.sample(10))
        st.write('''In the previous section, we noted that some variables were highly skewed. It’s important to address skewness because certain predictive models have inherent assumptions about the distribution of input features.
    These models, known as parametric models, generally assume that all variables are both independent and normally distributed.
    While skewness isn’t always negative, addressing highly skewed variables is beneficial for the reasons mentioned and because it can enhance the speed at which predictive models converge to their optimal solutions.
    There are several methods to handle skewed variables, such as applying transformations like the square root, cube root, or logarithm to continuous numerical columns. 
    In our case we applied log10n transformation for positively skewed features.
    ''')
        st.write('Let′s have a look at some data that was previsouly highly skewed.')
        fig, axs = plt.subplots(3,1, figsize=(18, 20))
        # Plot histograms
        sns.histplot((df["cons_12m"].dropna()), kde=True, ax=axs[0], color='#5F9EA0')
        sns.histplot((df[df["has_gas"]==1]["cons_gas_12m"].dropna()), kde=True, ax=axs[1], color='#5F9EA0')
        sns.histplot((df["cons_last_month"].dropna()), kde=True, ax=axs[2], color='#5F9EA0')
        plt.tight_layout()
        plt.show()
        st.pyplot(plt)

        st.write('As you can see, because of the log10 transformation we now have resolved the skewness issue. Now we are ready to build our model.')

    # Modeling and evaluation
    st.markdown("<h1 style='font-size:32px;'>Model Development and Evaluation</h1>", unsafe_allow_html=True)
    with st.expander('Predicting Customer Churn'):
    # Cache the function that prepares the data
        @st.cache_data
        def prepare_data(df):
            df_copy = df.copy()
            model_df = df_copy.drop(columns=['date_activ', 'date_end', 'date_modif_prod', 'date_renewal'], axis=1)
            y = model_df['churn']
            X = model_df.drop(columns=['id', 'churn'])
            return X, y, model_df

        # Cache the model training process
        @st.cache_data
        def train_model(X_train, y_train, n_estimators, random_state, bootstrap, ccp_alpha, class_weight, 
                        criterion, max_depth, max_features, max_leaf_nodes, max_samples, 
                        min_impurity_decrease, min_samples_leaf, min_samples_split, 
                        min_weight_fraction_leaf, n_jobs, oob_score, verbose, warm_start):
            
            model = RandomForestClassifier(
                n_estimators=n_estimators, 
                random_state=random_state, 
                bootstrap=bootstrap,
                ccp_alpha=ccp_alpha,
                class_weight=class_weight,
                criterion=criterion,
                max_depth=max_depth,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                max_samples=max_samples,
                min_impurity_decrease=min_impurity_decrease,
                min_samples_leaf=min_samples_leaf,
                min_samples_split=min_samples_split,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                n_jobs=n_jobs,
                oob_score=oob_score,
                verbose=verbose,
                warm_start=warm_start
            )

            model.fit(X_train, y_train)
            return model

        # Assume df is your loaded DataFrame
        X, y, model_df = prepare_data(df)

        st.write('Before splitting our data and performing machine learning, we removed datetime data from the dataset because we already engineered the necessary monthly data.')
        st.write(model_df.sample(10))
        st.write('The next step is to use that data to predict churn. Our target is churn, while the features for prediction are the remaining features from the previous dataset.')

        # Display the target and features
        st.write('Target :')
        st.write(y.head(10))
        st.write('Features :')
        st.write(X.head(10))

        st.write('Please note that we will be using a random forest classifier, so scaling data isn’t required.')
        st.write('👉 Now, go to the sidebar and choose your test/train split ratio. By default, this ratio is set to 25% for the test set and 75% for the training set.')
        st.write('You can also adjust the model’s parameters to compare results with the default parameter settings.')
        st.sidebar.write('Once you finish setting the parameters, run the model using the button below.')

        if st.sidebar.button('Train Model'):
            # Split data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=parameter_split_size/100, random_state=random_state)
            
            # Train and cache the model
            model = train_model(
                X_train, y_train, 
                n_estimators=n_estimators, 
                random_state=random_state, 
                bootstrap=bootstrap,
                ccp_alpha=ccp_alpha,
                class_weight=class_weight,
                criterion=criterion,
                max_depth=max_depth,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                max_samples=max_samples,
                min_impurity_decrease=min_impurity_decrease,
                min_samples_leaf=min_samples_leaf,
                min_samples_split=min_samples_split,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                n_jobs=n_jobs,
                oob_score=oob_score,
                verbose=verbose,
                warm_start=warm_start
            )

    # Model is now trained, you can proceed with evaluation or predictions    
        # df_copy = df.copy()
        # model_df = df_copy.drop(columns=['date_activ', 'date_end', 'date_modif_prod', 'date_renewal'], axis=1)
        # st.write('Before splitting our data and perform machine learning we removed datetime data from the dataset because we already engineered monthly data that we need.')
        # st.write(model_df.sample(10))
        # st.write('The second step is to use that data to predict churn, our target is churn while the features we will use for prediction are the rest of the features from the prebious dataset.')
        # # Separate target variable from independent variables
        # st.write('Target :')
        # y = model_df['churn']
        # st.write(y.head(10))
        # st.write('Features :')
        # X = model_df.drop(columns=['id', 'churn'])
        # st.write(X.head(10))

        # st.write('Please note that we will be using a random forest classifier so scaling data isn′t required.')
        # st.write('👉 Now, go to the sidebar and choose your test/train split ratio, by default this ratio is set to 25% for test set - 75% for training set.')
        # st.write('You can also play with the model´s parameters to compare results to default parameters settings.')
        # st.sidebar.write('Once you finish parameters settings, run the model using the button below.')
        
        # if st.sidebar.button('Train Model'):
        #     # Split data into trainind and test sets (we're using a 75-25 split here)
        #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=parameter_split_size/100, random_state=random_state)
        #     # st.write(X_train.shape)
        #     # st.write(X_test.shape)
        #     # st.write(y_train.shape)
        #     # st.write(y_test.shape)

        #     ## Model training
        #     model = RandomForestClassifier(n_estimators=n_estimators, 
        #                                 random_state=random_state, 
        #                                 bootstrap=bootstrap,
        #                                 ccp_alpha=ccp_alpha,
        #                                 class_weight=class_weight,
        #                                 criterion=criterion,
        #                                 max_depth=max_depth,
        #                                 max_features=max_features,
        #                                 max_leaf_nodes=max_leaf_nodes,
        #                                 max_samples=max_samples,
        #                                 min_impurity_decrease = min_impurity_decrease,
        #                                 min_samples_leaf=min_samples_leaf,
        #                                 min_samples_split = min_samples_split,
        #                                 min_weight_fraction_leaf = min_weight_fraction_leaf,
        #                                 n_jobs = n_jobs,
        #                                 oob_score = oob_score,
        #                                 verbose = verbose,
        #                                 warm_start = warm_start)

        #     model.fit(X_train, y_train)
            
            st.write("Summary of parameters used :")
            params = model.get_params()
            st.markdown(f'```json\n{params}\n```')

            # Generate predictions on test set
            y_pred = model.predict(X_test)

            # Model Evaluation on test set
            accuracy = accuracy_score(y_pred, y_test)
            st.write('Model accuracy on test set :')
            st.write(accuracy)

            # Get a classification report on test set
            test_classification_report = classification_report(y_pred, y_test)
            st.write('Classification report on test set :')
            # st.text(test_classification_report)
            st.markdown(f'```plaintext\n{test_classification_report}\n```')
            st.markdown(
                """
                <div style="display: flex; justify-content: left;">
                    <h3 style='font-size:16px;'>Now let´s see how well the model is generalizing predictions</h3>
                </div>
                """, 
                unsafe_allow_html=True
            )
            # Model Generalization on base data
            predictions = model.predict(X)
            
            # Model evaluation on base data
            general_accuracy = accuracy_score(predictions, y.values)
            st.write('Model accuracy on initial target data :')
            st.write(general_accuracy)

            # Get a classification report on base data
            base_classification_report = classification_report(predictions, y.values)
            st.write('Classification report on initial target data :')
            st.markdown(f'```plaintext\n{base_classification_report}\n```')

            st.write('Now that our model is performing well on general data, we use an ensemble-based method called feature_importances_ that will allow us show how much each feature contributes to the prediction')
            # Show how much each feature contributes to the prediction using built-in function in ensemble methods
            @st.cache_data
            def compute_feature_importances(_model, X_train):
                return pd.DataFrame({
                    'features': X_train.columns,
                    'importance': _model.feature_importances_
                }).sort_values(by='importance', ascending=True).reset_index(drop=True)

            # After model evaluation
            st.write('Now that our model is performing well on general data, we use an ensemble-based method called feature_importances_ that will allow us to show how much each feature contributes to the prediction')

            feature_importances = compute_feature_importances(model, X_train)
            st.write(feature_importances)
            st.write('Below is a plot showing the degree of importance by feature.')
            feature_importances_plot(feature_importances)

            st.markdown('''
            Some of the most 'important' features for our predictive model are :
            - cons_12m
            - net_margin
            - forecast_meter_rent_12m
            - forecast_cons_12m 
            - margin_net_pow_ele 
            - margin_gross_pow_ele
                    ''')
            
            st.write('According to the graph, our price sensitivity features are not the main reason why some customers churn.')
            st.write('In the next section we will study discount in order to understand how it could retain PowerCo´s clients while maintaining profit.')
            proba_predictions = model.predict_proba(X_test)
            probabilities = proba_predictions[:, 1]

            X_test = X_test.reset_index()
            X_test.drop(columns='index', inplace=True)

            X_test['churn'] = proba_predictions.tolist()
            X_test['churn_probability'] = probabilities.tolist()
else:
    st.warning('👆 Please load both Client and Price data to proceed or click ** Load example data **.')
