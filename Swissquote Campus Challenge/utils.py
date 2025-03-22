import math  # Import math library for mathematical functions
import numpy as np  # Import numpy for efficient array operations
import pandas as pd  # Import pandas for data handling and analysis
import statsmodels.api as sm  # Import statsmodels for statistical modeling
import matplotlib.pyplot as plt

class TotalReturnIndexCalculator:

    def monthly_simple_returns(self, total_price_list):
        # Initialize list to store monthly simple returns
        monthly_simple_returns_list = []
        
        # Loop through price list to calculate simple returns for each month
        for i in range(1, len(total_price_list)):
            monthly_simple_return = (total_price_list.iloc[i] / total_price_list.iloc[i - 1]) - 1
            monthly_simple_returns_list.append(monthly_simple_return)
        
        # Return the list of calculated monthly simple returns
        return monthly_simple_returns_list

    def monthly_log_returns(self, total_price_list):
        # Initialize list to store monthly log returns
        monthly_log_returns_list = []
        
        # Loop through price list to calculate log returns for each month
        for i in range(1, len(total_price_list)):
            monthly_log_return = math.log(total_price_list.iloc[i] / total_price_list.iloc[i - 1])
            monthly_log_returns_list.append(monthly_log_return)
        
        # Return the list of calculated monthly log returns
        return monthly_log_returns_list

    def annual_log_returns(self, total_price_list, months_per_year=12):
        # Calculate monthly log returns
        monthly_log_returns_list = self.monthly_log_returns(total_price_list)
        
        # Initialize list to store annual log returns
        annual_log_returns_list = []
        
        # Loop through monthly returns in year-long segments
        for i in range(0, len(monthly_log_returns_list), months_per_year):
            # Sum up log returns for each year
            annual_log_return = sum(monthly_log_returns_list[i:i + months_per_year])
            annual_log_returns_list.append(annual_log_return)
        
        # Return the list of calculated annual log returns
        return annual_log_returns_list

    def annual_simple_returns(self, total_price_list):
        # Calculate annual log returns
        annual_log_returns_list = self.annual_log_returns(total_price_list)
        
        # Convert annual log returns to simple returns
        annual_simple_returns_list = [math.exp(annual_log_return) - 1 for annual_log_return in annual_log_returns_list]
        
        # Return the list of annual simple returns
        return annual_simple_returns_list

class Calculator:
    def mean(self, returns_list):
        # Check for empty list to avoid division by zero
        if len(returns_list) == 0:
            return 0  
        
        # Calculate and return the mean of the returns list
        return sum(returns_list) / len(returns_list)

    def geometric_mean(self, returns_list):
        # Check for empty list to avoid division by zero
        if len(returns_list) == 0:
            return 0  
        
        # Convert each return to log(1 + return) for geometric mean calculation
        log_list = [math.log(1 + r) for r in returns_list]
        
        # Calculate the arithmetic mean of the logarithmic values
        log_mean = self.mean(log_list)
        
        # Convert the logarithmic mean back to the original scale and subtract 1
        return math.exp(log_mean) - 1

    def variance(self, returns_list):
        # Check for empty list to avoid division by zero
        if len(returns_list) == 0:
            return 0  
        
        # Calculate mean return for variance computation
        mean_return = self.mean(returns_list)
        
        # Calculate squared differences from the mean
        squared_diff_list = [(r - mean_return) ** 2 for r in returns_list]
        
        # Calculate and return the variance
        return sum(squared_diff_list) / len(returns_list)
    
    def standard_deviation(self, returns_list):
        # Check for empty list to avoid division by zero
        if len(returns_list) == 0:
            return 0  
        
        # Calculate the variance of the returns list
        variance_value = self.variance(returns_list)
        
        # Return the square root of the variance (standard deviation)
        return math.sqrt(variance_value)

    def sharpe_ratio(self, excess_returns_list):
        # Check for empty list to avoid division by zero
        if len(excess_returns_list) == 0:
            return 0  
        
        # Calculate the mean return
        mean_return = self.mean(excess_returns_list)
        
        # Calculate the standard deviation of returns
        std_dev = self.standard_deviation(excess_returns_list)
        
        # Calculate and return the Sharpe Ratio
        if std_dev == 0:
            return 0  # Avoid division by zero in case of zero standard deviation
        return mean_return / std_dev

class Capm:
    def __init__(self):
        # Initialize a Calculator instance for statistical calculations
        self.calculator = Calculator()

    def excess_returns_list(self, returns_list, risk_free_rate):
        # Calculate and return the excess returns by subtracting the risk-free rate
        excess_returns_list = [r - risk_free_rate for r in returns_list]
        return excess_returns_list


    def regression(self, returns_asset, returns_market, risk_free_rate):
        # Calculate excess returns for both the asset and the market
        x = self.excess_returns_list(returns_market, risk_free_rate)  
        # Compute the excess returns for the market (returns above the risk-free rate).
        y = self.excess_returns_list(returns_asset, risk_free_rate)  
        # Compute the excess returns for the asset (returns above the risk-free rate).
        
        # Add a constant to the independent variable to account for the intercept in the regression
        X = sm.add_constant(x)  
        
        # Create and fit the linear regression model
        model = sm.OLS(y, X)  
        # Create an Ordinary Least Squares (OLS) regression model:
        # - y (dependent variable) represents the market's excess returns.
        # - X (independent variable) includes the asset's excess returns and a constant for the intercept.
        results = model.fit()  
        # Fit the model to the data and calculate regression coefficients and statistics.
        
        # Retrieve beta coefficient and its p-value
        beta = results.params[1]  # The beta coefficient (sensitivity of the asset to market returns).
        p_beta = results.pvalues[1]  # The p-value for the beta coefficient, indicating its statistical significance.
    
        # Retrieve alpha coefficient and its p-value
        alpha = results.params[0]  
        # The alpha coefficient represents the excess return of the asset that cannot be explained by the market.
        p_alpha = results.pvalues[0]  
        # The p-value for the alpha coefficient, indicating its statistical significance.
        
        # Calculate the expected return using the Capital Asset Pricing Model (CAPM)
        capm_expected_return = risk_free_rate + beta * (self.calculator.mean(returns_market) - risk_free_rate)
        # CAPM formula:
        # Expected Return = Risk-Free Rate + Beta * (Market Return - Risk-Free Rate).
        
        # Adjust the expected return by incorporating alpha
        alpha_adjusted_expected_return = capm_expected_return + alpha
        # Alpha-adjusted expected return accounts for any returns beyond what CAPM predicts.
        
        # Return a dictionary with the regression results
        return {
            "beta": beta,  # The beta coefficient (sensitivity of the asset to market movements).
            "p_beta": p_beta,  # The p-value for beta (indicates the significance of beta in the regression).
            "capm_expected_return": capm_expected_return,  # The expected return based on CAPM.
            "alpha": alpha,  # The alpha coefficient (excess return not explained by market movements).
            "p_alpha": p_alpha, # The p-value for alpha (indicates the significance of alpha in the regression).
            "alpha_adjusted_expected_return": alpha_adjusted_expected_return  # Expected return with alpha adjustment.
        }


    def draw_sml(self, rf, mean_market_return, betas, expected_returns, assets):
        # Generate a range of beta values for the SML
        sml_betas = np.linspace(0, 2, 100)
            
        # Calculate the expected returns on the SML
        # Formula: Expected Return = Risk-Free Rate + Beta * (Market Return - Risk-Free Rate)
        sml_returns = rf + sml_betas * (mean_market_return - rf)
        
        # Plot the Security Market Line
        plt.figure(figsize=(10, 6))  # Set the figure size to 10x6 inches
        plt.plot(sml_betas, sml_returns, label="Security Market Line (SML)", color="blue")  # Plot the SML in blue
        
        # Scatter plot for the assets
        plt.scatter(betas, expected_returns, color='red', label="Assets")  # Represent each asset as a red dot
        
        # Annotate each asset with its name
        for i, asset in enumerate(assets):
            plt.text(betas[i], expected_returns[i], asset, fontsize=9)  # Add labels for each asset near their points
        
        # Add titles, axis labels, and a legend
        plt.title("Security Market Line (CAPM)")  # Set the title of the chart
        plt.xlabel("Beta (β)")  # Label for the x-axis
        plt.ylabel("Expected Return")  # Label for the y-axis
        plt.axhline(y=rf, color='gray', linestyle='--', label=f"Risk-Free Rate ({100 * rf:.2f} %)")  # Add the risk-free rate with four decimal places
        plt.legend()  # Display the legend
        plt.grid()  # Add a grid for better visualization
        plt.xlim(0, 2)  # Limit the x-axis to a beta range of 0 to 2
        plt.show()  # Display the final plot

