import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm


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

def update_betting_file(bet_date, bet_amount, csv_file, ledger_file):
    """Read the CSV file, update PDF based on the last row, and append a new row."""
    df = pd.read_csv(csv_file, sep=';')
    ledger_df = pd.read_csv(ledger_file, sep=';')

    last_row = df.iloc[-1]
    mu = last_row['mu']
    std = last_row['std']

    # Generate realizations for the initial PDF # we could prepopulate this with the ledger data to make it faster
    initial_realizations = generate_initial_pdf(mu, std)
    ledger_df_days =pd.to_datetime(ledger_df['predicted date'], format='%Y-%m-%d').dt.dayofyear
    combined_data = np.concatenate([initial_realizations, ledger_df_days.values])

    global_mu = np.mean(combined_data)
    global_std = np.std(combined_data)

    #print(get_odds(csv_file, bet_date, params=(global_mu, global_std)))
    
    odds=get_odds(bet_date=bet_date, params=(global_mu, global_std))

    print(f" A bet of {bet_amount} psor coins will be placed on {bet_date} with odds of {odds:.4f}")
    print("Introduce ' your name and group' to confirm the bet")

    name = input('name>> ')
    group =input('group>> ')
    new_row_global = pd.DataFrame.from_dict([{
        'name': name,
        'group': group,
        'bet amount': bet_amount,
        'predicted date': bet_date,
        'current date': pd.to_datetime('today'),  
    }])
    
    ledger_df = pd.concat([ ledger_df,new_row_global], ignore_index=True)

    new_row = pd.DataFrame.from_dict([{
        'date': pd.to_datetime('today'),
        'mu': np.round(global_mu,3),
        'std': np.round(global_std,3),
    }])
    df=pd.concat([df, new_row], ignore_index=True)
    df.to_csv(csv_file, sep=';', index=False)
    ledger_df.to_csv(ledger_file, sep=';', index=False)

    print(f"Bet succesfully added to the ledger {bet_amount} on date {bet_date}")

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
    #
    updated_cdf = stats.norm(loc=mu, scale=std)
    prob = updated_cdf.cdf(bet_date + 1) - updated_cdf.cdf(bet_date)
    

    min_prob = 1e-10  # prevent division by zero
    prob = max(prob, min_prob)
    
    # not very weel thought out, but it works
    odds = 10 ** (1 - 2 * prob) * 0.3 + 0.2 
    odds = max(0.99, min(odds, 10))
    
    print(f"Odds: {odds:.4f}")
    return odds

def plot_pdf(day_of_year, title):
    """Plot the PDF with histogram, normal distribution, and KDE"""
    
    mu, std = norm.fit(day_of_year)
    x_values = np.linspace(day_of_year.min(), day_of_year.max(), 500)
    y_norm = norm.pdf(x_values, mu, std)

    # uncomment to use moments estimator, for the amount of data it is basically the same
    # mu_2, std_2 = norm.fit(day_of_year,method='mm')
    # y_norm_2 = norm.pdf(x_values, mu_2, std_2)
    plt.figure(figsize=(20, 5))
    plt.hist(day_of_year, bins='auto', density=True, alpha=0.5, color='gray', label='Histogram')

    plt.plot(x_values, y_norm, label="Normal-MLE", color='blue')
    # plt.plot(x_values, y_norm_2, label="Normal-Moments", color='purple') 
    day_of_year.plot.density(label="KDE", color='green')

    
    plt.title(title)
    plt.xlabel("Day of Year")
    plt.ylabel("Density")
    plt.legend()
    plt.grid()
    plt.show()

def plot_mu_std_vs_n(day_of_year, title):
    """Plot mu and std vs n using two y-axes

     extend to mu/std vs time if we  want to extract the time the data was pushed
    """
    mu_values = []
    std_values = []
    n_values = []

    N = len(day_of_year)
    n = len(day_of_year) // 10  # chunk size

    # Loop to estimate mu, std progressively
    for i in range(n, N + 1, n):
        subset = day_of_year[:i]
        mu, std = norm.fit(subset)
        # could use other estimtors like moments 
        mu_values.append(mu)
        std_values.append(std)
        n_values.append(i)

    
    fig, ax1 = plt.subplots(figsize=(20, 5))

    ax1.plot(n_values, mu_values, label="Mu (mean)", color='red')
    ax1.set_xlabel("Sample size (n)")
    ax1.set_ylabel("Mean (day of year)", color='red')
    ax1.tick_params(axis='y', labelcolor='red')

  
    ax2 = ax1.twinx()
    ax2.plot(n_values, std_values, label="Std (standard deviation)", color='orange')
    ax2.set_ylabel("Std (days)", color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Add legends and title
    ax1.set_title(f"{title}")
    ax1.grid()
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.show()