import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# from scripts.utils.logger import configure_logger

# logger = configure_logger('feature_engineering', 'logs/feature_engineering.log')

def create_features(data):
    """Enhanced feature engineering with graphical representation."""
    try:
        # logger.info("Starting feature engineering...")
        
        # Extract date-related features
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            data['Year'] = data['Date'].dt.year
            data['Month'] = data['Date'].dt.month
            data['Day'] = data['Date'].dt.day
            data['WeekOfYear'] = data['Date'].dt.isocalendar().week
            data['DayOfWeek'] = data['Date'].dt.dayofweek
            data['IsWeekend'] = data['DayOfWeek'].isin([5, 6]).astype(int)
            # logger.info("Date-related features extracted.")

        # Seasonal Features
        data['IsChristmas'] = ((data['Month'] == 12) & (data['Day'] > 20)).astype(int)
        data['IsEaster'] = ((data['Month'] == 4) & (data['Day'] < 10)).astype(int)
        
        # Promo effectiveness
        if 'Promo' in data.columns and 'Customers' in data.columns:
            data['PromoImpact'] = data['Promo'] * data['Customers']
            # logger.info("Promo impact feature created.")
        
        # Distance-related feature interactions
        if 'CompetitionDistance' in data.columns:
            data['CompetitionProximity'] = data['CompetitionDistance'].apply(
                lambda x: 1 / x if x > 0 else 0
            )
            # logger.info("Competition proximity feature created.")

        # Sales transformations
        if 'Sales' in data.columns:
            data['LogSales'] = data['Sales'].apply(lambda x: np.log1p(x) if x > 0 else 0)
            # logger.info("Log-transformed 'Sales' column.")

        # logger.info("Feature engineering completed.")
        return data
    except Exception as e:
        # logger.error(f"Error during feature engineering: {e}")
        raise

def plot_feature_distributions(data):
    """Plots the distribution of key features."""
    features_to_plot = ['Sales', 'LogSales', 'PromoImpact', 'CompetitionProximity', 'IsChristmas', 'IsEaster']
    
    for feature in features_to_plot:
        if feature in data.columns:
            plt.figure(figsize=(8, 6))
            sns.histplot(data[feature], kde=True, color='blue')
            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            plt.show()