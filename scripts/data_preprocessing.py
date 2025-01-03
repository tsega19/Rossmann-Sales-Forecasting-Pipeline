import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append(os.path.abspath('../scripts'))
from utils.logger import configure_logger

logger = configure_logger('data_preprocessing', '../logs/data_preprocessing.log')

def clean_data(data):
    """Perform enhanced data cleaning and visualizations."""
    try:
        logger.info("Starting data cleaning...")

        # Handle missing values only for numerical columns
        for column in data.select_dtypes(include=[np.number]).columns:
            missing_count = data[column].isnull().sum()
            if missing_count > 0:
                data[column].fillna(data[column].median(), inplace=True)
                logger.info(f"Filled {missing_count} missing values in {column} with median.")

        # Handle missing values for categorical columns
        for column in data.select_dtypes(include=['object']).columns:
            missing_count = data[column].isnull().sum()
            if missing_count > 0:
                data[column].fillna(data[column].mode()[0], inplace=True)
                logger.info(f"Filled {missing_count} missing values in {column} with mode.")

        # Handle outliers only for numerical columns
        for column in data.select_dtypes(include=[np.number]).columns:
            if column not in ['Date']:
                q1 = data[column].quantile(0.25)
                q3 = data[column].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
                logger.info(f"Detected {len(outliers)} outliers in {column}.")
                data[column] = np.clip(data[column], lower_bound, upper_bound)

        logger.info("Data cleaning completed.")
        return data
    except Exception as e:
        logger.error(f"Error during data cleaning: {e}")
        raise

def visualize_data(data):
    """Visualize distribution, correlation, and outliers."""
    try:
        logger.info("Starting data visualization...")

        # Distribution of numerical columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        for column in numerical_cols:
            plt.figure(figsize=(10, 6))
            sns.histplot(data[column], kde=True, bins=30)
            plt.title(f"Distribution of {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.show()
            logger.info(f"Displayed distribution of {column}.")

        # Correlation heatmap of numerical features
        corr_matrix = data[numerical_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
        plt.title("Correlation Heatmap")
        plt.show()
        logger.info("Displayed correlation heatmap.")
        # Boxplots to visualize outliers in numerical columns
        for column in numerical_cols:
            plt.figure(figsize=(8, 6))
            sns.boxplot(x=data[column])
            plt.title(f"Boxplot of {column}")
            plt.show()

        logger.info("Data visualization completed.")
    except Exception as e:
        logger.error(f"Error during data visualization: {e}")
        raise
def check_promotions_distribution(data):
    """Check the distribution of promotions in both training and test sets."""
    try:
        logger.info("Checking distribution of promotions between training and test sets...")
        promo_distribution = data.groupby('Promo').size()
        logger.info(f"Promo distribution:\n{promo_distribution}")
    except Exception as e:
        logger.error(f"Error checking promotions distribution: {e}")
        raise
    plt.figure(figsize=(8, 6))
    sns.barplot(x=promo_distribution.index, y=promo_distribution.values)
    plt.title('Promotions Distribution')
    plt.xlabel('Promo')
    plt.ylabel('Frequency')
    plt.show()
    logger.info("Promotions distribution plot displayed.")

def analyze_sales_behavior(data):
    """Analyze sales behavior before, during, and after holidays."""
    try:
        logger.info("Analyzing sales behavior around holidays...")
        # Assuming 'StateHoliday' column contains holiday indicators
        holidays = ['a', 'b', 'c']  # Public holiday, Easter, Christmas
        holiday_sales = data[data['StateHoliday'].isin(holidays)]
        pre_holiday_sales = data[~data['StateHoliday'].isin(holidays)]
        
        # Calculate average sales for different periods
        avg_sales_pre = pre_holiday_sales['Sales'].mean()
        avg_sales_holiday = holiday_sales['Sales'].mean()
        
        logger.info(f"Average sales before holiday: {avg_sales_pre}")
        logger.info(f"Average sales during holiday: {avg_sales_holiday}")
    except Exception as e:
        logger.error(f"Error analyzing sales behavior: {e}")
        raise
    plt.figure(figsize=(8, 6))
    sns.barplot(x=['Pre-Holiday', 'Holiday'], y=[avg_sales_pre, avg_sales_holiday])
    plt.title('Sales Behavior Around Holidays')
    plt.xlabel('Period')
    plt.ylabel('Average Sales')
    plt.show()
    logger.info("Sales behavior around holidays displayed.")

def seasonal_purchase_behavior(data):
    """Find out seasonal purchase behaviors such as Christmas, Easter, etc."""
    try:
        logger.info("Checking seasonal purchase behavior...")
        # Extract month from the date or based on the 'StateHoliday' column
        seasonal_sales = data.groupby('StateHoliday')['Sales'].mean()
        logger.info(f"Seasonal sales by holiday:\n{seasonal_sales}")
    except Exception as e:
        logger.error(f"Error checking seasonal purchase behavior: {e}")
        raise
    # plot
    plt.figure(figsize=(8, 6))
    sns.barplot(x=seasonal_sales.index, y=seasonal_sales.values)
    plt.title('Seasonal Sales by Holiday')
    plt.xlabel('Holiday')
    plt.ylabel('Average Sales')
    plt.show()
    logger.info("Seasonal purchase behavior displayed.")

def sales_and_customers_correlation(data):
    """Analyze the correlation between sales and the number of customers."""
    try:
        logger.info("Analyzing correlation between sales and number of customers...")
        correlation = data['Sales'].corr(data['Customers'])
        logger.info(f"Correlation between sales and customers: {correlation}")
    except Exception as e:
        logger.error(f"Error analyzing sales and customer correlation: {e}")
        raise
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Customers', y='Sales', data=data, color='blue')
    plt.title('Sales vs. Customers')
    plt.xlabel('Number of Customers')
    plt.ylabel('Sales')
    plt.show()
    logger.info("Sales and customers correlation plot displayed.")
def promo_effect_on_sales(data):
    """Analyze how promotions affect sales."""
    try:
        logger.info("Analyzing the effect of promotions on sales...")
        promo_sales = data.groupby('Promo')['Sales'].mean()
        logger.info(f"Average sales with and without promotions:\n{promo_sales}")
    except Exception as e:
        logger.error(f"Error analyzing promo effect on sales: {e}")
        raise
    plt.figure(figsize=(8, 6))
    sns.barplot(x=promo_sales.index, y=promo_sales.values)
    plt.title('Promo Effect on Sales')
    plt.xlabel('Promo')
    plt.ylabel('Average Sales')
    plt.show()
    logger.info("Promo effect on sales plot displayed.")

def analyze_customer_behavior(data):
    """Analyze customer behavior during store opening and closing times."""
    try:
        logger.info("Analyzing customer behavior during store operating hours...")
        # Assuming 'Open' column is available to determine if store was open
        open_stores = data[data['Open'] == 1]
        closed_stores = data[data['Open'] == 0]
        
        # Compare sales for open and closed stores
        avg_sales_open = open_stores['Sales'].mean()
        avg_sales_closed = closed_stores['Sales'].mean()
        
        logger.info(f"Average sales for open stores: {avg_sales_open}")
        logger.info(f"Average sales for closed stores: {avg_sales_closed}")
    except Exception as e:
        logger.error(f"Error analyzing customer behavior: {e}")
        raise
    plt.figure(figsize=(8, 6))
    sns.barplot(x=['Open', 'Closed'], y=[avg_sales_open, avg_sales_closed])
    plt.title('Sales for Open and Closed Stores')
    plt.xlabel('Store Status')
    plt.ylabel('Average Sales')
    plt.show()
    logger.info("Customer behavior analysis displayed.")
def store_open_weekdays(data):
    """Check which stores are open on all weekdays and their sales on weekends."""
    try:
        logger.info("Checking store operations on weekdays and sales on weekends...")
        weekday_stores = data[data['Open'] == 1]
        weekday_sales = weekday_stores.groupby('Store')['Sales'].mean()
        
        # Identify stores open on all weekdays (Monday to Friday)
        stores_open_weekdays = weekday_stores[weekday_stores['Open'] == 1]
        weekend_sales = data[data['Open'] == 1].groupby('Store')['Sales'].mean()
        
        logger.info(f"Stores open weekdays and weekend sales:\n{weekend_sales}")
    except Exception as e:
        logger.error(f"Error checking store operations on weekdays: {e}")
        raise
    plt.figure(figsize=(8, 6))
    sns.barplot(x=weekend_sales.index, y=weekend_sales.values)
    plt.title('Weekend Sales for Stores Open on Weekdays')
    plt.xlabel('Store ID')
    plt.ylabel('Average Sales')
    plt.show()
    logger.info("Store operations on weekdays and weekend sales displayed.")