import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns

# Set the title of the Streamlit app
st.title('Sales Forecasting Model')

# File uploader to allow the user to upload a CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

# Check if a file has been uploaded
if uploaded_file:
    # Read the uploaded CSV file into a pandas DataFrame and handle any potential encoding issues
    df = pd.read_csv(uploaded_file, encoding='unicode_escape')
    
    # Clean the column names by stripping any extra spaces and converting them to lowercase
    df.columns = df.columns.str.strip().str.lower()

    # Display a dropdown to select a state from the unique values in the 'state' column
    selected_state = st.selectbox("Select a state:", df['state'].unique())
    
    # Filter the dataset based on the selected state
    filtered_df = df[df['state'] == selected_state]

    # Group data by product category and sum the orders for each category
    product_category_orders = filtered_df.groupby('product_category')['orders'].sum().reset_index()

    # Create a bar plot showing the sum of orders by product category for the selected state
    plt.figure(figsize=(10, 6))
    sns.barplot(x='product_category', y='orders', data=product_category_orders, palette='summer')
    plt.xticks(rotation=90)  # Rotate x-axis labels for readability
    plt.title(f'Sum of Orders by Product Category in {selected_state}')
    st.pyplot(plt)  # Display the plot in the Streamlit app
    plt.clf()  # Clear the plot to avoid overlap

    # Display the cleaned column names of the filtered DataFrame
    st.write("Cleaned column names in the dataset:")
    st.write(filtered_df.columns)

    # Define columns to be dropped (irrelevant for forecasting)
    columns_to_drop = ['user_id', 'cust_name', 'product_id', 'gender', 'age group', 'age', 
                       'marital_status', 'state', 'zone', 'occupation', 
                       'status', 'unnamed1']
    
    # Drop unnecessary columns if they exist in the filtered DataFrame
    df_cleaned = filtered_df.drop(columns=[col for col in columns_to_drop if col in filtered_df.columns])

    # Check if 'orders' and 'amount' columns are available in the cleaned DataFrame
    if 'orders' in df_cleaned.columns and 'amount' in df_cleaned.columns:
        # Remove rows with missing values in 'orders' and 'amount'
        df_cleaned = df_cleaned.dropna(subset=['orders', 'amount'])
        
        # Create a 'ds' column for dates and 'y' column for the values Prophet needs to model
        df_cleaned['ds'] = pd.date_range(start='2020-01-01', periods=len(df_cleaned), freq='D')
        df_cleaned['y'] = df_cleaned['amount']

        # Initialize and fit the Prophet model
        model = Prophet()
        model.fit(df_cleaned[['ds', 'y']])
        
        # Create a future DataFrame to predict sales for the next 2 days
        future = model.make_future_dataframe(periods=2)
        forecast = model.predict(future)
        
        # Round and extract the predicted sales and their upper/lower bounds
        forecast['predicted_sales'] = forecast['yhat'].round(0)
        forecast['predicted_sales_lower_bound'] = forecast['yhat_lower'].round(0)
        forecast['predicted_sales_upper_bound'] = forecast['yhat_upper'].round(0)

        # Display the forecasted sales data for the next 2 days
        st.write("Forecasted Data", forecast[['ds', 'predicted_sales', 'predicted_sales_lower_bound', 'predicted_sales_upper_bound']].tail())

        # Plot the overall sales trend using the Prophet model
        st.write("Overall Sales Trend:")
        if 'trend' in forecast.columns:
            fig1 = model.plot(forecast)  # Plot the forecast
            st.pyplot(fig1)  # Display the plot in the Streamlit app
            fig2 = model.plot_components(forecast)  # Plot the forecast components (trend, seasonality, etc.)
            st.pyplot(fig2)
        else:
            st.error("Forecast does not contain the expected trend data.")

        # Get unique product categories from the filtered DataFrame
        products = filtered_df['product_category'].unique()
        
        # Initialize an empty dictionary to store the forecasts for each product category
        product_forecasts = {}

        # Loop through each product category and create a forecast for each
        for product in products:
            product_data = df_cleaned[df_cleaned['product_category'] == product]
            if not product_data.empty:
                # Group by 'ds' (date) and sum the 'amount' for each day
                product_data = product_data.groupby('ds')['amount'].sum().reset_index()
                product_data['y'] = product_data['amount']
                model = Prophet()
                model.fit(product_data[['ds', 'y']])
                
                # Forecast for the next 30 days
                future = model.make_future_dataframe(periods=30)
                forecast = model.predict(future)
                
                # Store the forecast for each product category
                product_forecasts[product] = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        
        # Display the forecast for each product category using a line chart
        for product, forecast in product_forecasts.items():
            st.write(f"Forecast for {product}:")
            st.line_chart(forecast.set_index('ds')['yhat'])

        # Find the product category with the highest total forecasted sales
        max_product = max(product_forecasts, key=lambda p: product_forecasts[p]['yhat'].sum())
        st.write(f"The product to keep for maximum profit is: {max_product}")

        # Compare actual sales data with the forecasted data
        actual_data = df_cleaned[['ds', 'y']].rename(columns={'y': 'actual_sales'})

        # Initialize and fit a Prophet model for the entire dataset
        model = Prophet()
        model.fit(df_cleaned[['ds', 'y']])
        
        # Create a future DataFrame for the next 30 days and predict sales
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        # Rename 'yhat' (forecasted values) to 'forecasted_sales' for clarity
        forecast = forecast[['ds', 'yhat']].rename(columns={'yhat': 'forecasted_sales'})

        # Merge the actual sales data with the forecasted sales
        comparison_df = pd.merge(actual_data, forecast, on='ds', how='outer')

        # Forward-fill missing values in the comparison DataFrame
        comparison_df = comparison_df.fillna(method='ffill')

        # Display a line chart comparing actual sales and forecasted sales
        st.line_chart(comparison_df.set_index('ds')[['actual_sales', 'forecasted_sales']])

    else:
        # Show an error message if the required 'orders' and 'amount' columns are missing
        st.error("The dataset does not have the required 'Orders' and 'Amount' columns.")
