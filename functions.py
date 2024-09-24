import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd


def plot_pdf(pdf, title):
    """Plot the probability density function."""
    x_values = np.linspace(80, 120,(120-80)*4 )
    y_values = pdf.pdf(x_values)
    
    plt.plot(x_values, y_values, label=title)
    plt.title(title)
    plt.xlabel("Day of the year")
    plt.ylabel("Density")
    plt.legend()
    plt.grid()
    plt.show()  

def generate_initial_pdf(mu, std, n_realizations=100):
    """Generate 100 realizations of the initial PDF."""
    return np.random.normal(loc=mu, scale=std, size=n_realizations)

def plot_pdf(pdf, title):
    """Plot the probability density function."""
    x_values = np.linspace(80, 120, (120-80)*4)
    y_values = pdf.pdf(x_values)
    
    plt.plot(x_values, y_values, label=title)
    plt.title(title)
    plt.xlabel("Day of the year")
    plt.ylabel("Density")
    plt.legend()
    plt.grid()
    plt.show()  

def generate_initial_pdf(mu, std, n_realizations=100):
    """Generate realizations from the initial PDF."""
    return np.random.normal(loc=mu, scale=std, size=n_realizations)

def update_betting_file(bet_date, bet_amount, csv_file, ledger_file):
    """Read the CSV file, update PDF based on the last row, and append a new row."""
    # Read the CSV file (tab-separated) with the odds and statistics
    df = pd.read_csv(csv_file, sep=';')
    

    # Read the global ledger (tab-separated) that tracks historical bets
    ledger_df = pd.read_csv(ledger_file, sep=';')

    last_row = df.iloc[-1]
    mu = last_row['mu']
    std = last_row['std']

    # Generate realizations for the initial PDF
    initial_realizations = generate_initial_pdf(mu, std)

    # Extract the "Predicted Day of Breakup" from the global ledger
    ledger_df['predicted date'] =pd.to_datetime(ledger_df['predicted date'], format='%Y-%m-%d').dt.dayofyear

    # Combine the realizations with the actual bets from the global ledger
    combined_data = np.concatenate([initial_realizations, ledger_df['predicted date'].values])

    # Update global mu and std based on combined data (realizations + bets)
    global_mu = np.mean(combined_data)
    global_std = np.std(combined_data)

    #print(get_odds(csv_file, bet_date, params=(global_mu, global_std)))
    # Calculate the return using the updated PDF
    
    odds=get_odds(bet_date=bet_date, params=(global_mu, global_std))

    print(f" A bet of{bet_amount} psor coins will be placed on {bet_date} with odds of {odds:.4f}")
    print("Introduce ' your name and group' to confirm the bet")
# Create a new row with updated values
    name = input('name>> ')
    group =input('group>> ')
    new_row_global = pd.DataFrame.from_dict([{
        'name': name,
        'group': group,
        'bet amount': bet_amount,
        'predicted date': pd.to_datetime(bet_date, format='%Y-%m-%d'),
        'current date': pd.to_datetime('today'),  
    }])
    # Append the new row to the DataFrame
    ledger_df = pd.concat([ ledger_df,new_row_global], ignore_index=True)

    # Write the updated DataFrame back to the CSV file

    new_row = pd.DataFrame.from_dict([{
        'date': pd.to_datetime('today'),
        'mu': np.round(global_mu,3),
        'std': np.round(global_std,3),
    }])
    df=pd.concat([df, new_row], ignore_index=True)
    df.to_csv(csv_file, sep=';', index=False)
    ledger_df.to_csv(ledger_file, sep=';', index=False)

    print(f"Bet succesfully added to the ledger {bet_amount:.2f} on date {bet_date}")

def get_odds(csv_file=None, bet_date=None, params=None):
    """Calculate the odds for a given date based on the latest mu and std from the odds file.
    If no date is passed, it assumes it is just checking the latest odds; 
    if a date is passed, it will calculate the odds for that date.
    Params is a tuple with the mu and std values to use; if not passed, it will use the latest values from the csv file.
    """
    
    if bet_date is None:
        bet_date = input("Introduce date (YYYY-MM-DD) to check odds: ")

    if params is not None:
        mu, std = params
    else: 
        df = pd.read_csv(csv_file, sep=';')
        last_row = df.iloc[-1]
        mu, std = last_row['mu'], last_row['std']

    bet_date = pd.to_datetime(bet_date, format='%Y-%m-%d').dayofyear
    # Create a normal distribution using the latest mu and std
    updated_cdf = stats.norm(loc=mu, scale=std)

    # Calculate the probability density for the given bet date
    prob = updated_cdf.cdf(bet_date + 1) - updated_cdf.cdf(bet_date)
    
    # Set a minimum probability density to avoid division by zero
    min_prob = 1e-10  # Small constant to prevent division by zero
    prob = max(prob, min_prob)
    
    # Inverse relationship: higher odds for lower likelihood
    odds = 10 ** (1 - 2 * prob) * 0.3 + 0.2 

    # Cap odds between 0.99 and 10
    odds = max(0.99, min(odds, 10))
    
    print(f"Odds: {odds:.4f}")
   
    return odds
