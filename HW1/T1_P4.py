import csv
import numpy as np
import matplotlib.pyplot as plt

csv_filename = 'data/year-sunspots-republicans.csv'
years  = []
republican_counts = []
sunspot_counts = []

with open(csv_filename, 'r') as csv_fh:
    # Parse as a CSV file.
    reader = csv.reader(csv_fh)

    # Skip the header line.
    next(reader, None)

    # Loop over the file.
    for row in reader:

        # Store the data.
        years.append(float(row[0]))
        sunspot_counts.append(float(row[1]))
        republican_counts.append(float(row[2]))

# Turn the data into numpy arrays.
years  = np.array(years)
republican_counts = np.array(republican_counts)
sunspot_counts = np.array(sunspot_counts)
last_year = 1985

# Basis functions
# Based on the letter input for part ('a','b','c','d'), output numpy arrays for the bases.
# The shape of arrays you return should be: (a) 24x6, (b) 24x12, (c) 24x6, (c) 24x26
# xx is the input of years (or any variable you want to turn into the appropriate basis).
# is_years is a Boolean variable which indicates whether or not the input variable is
# years; if so, is_years should be True, and if the input varible is sunspots, is_years
# should be false
def make_basis(xx,part='a',is_years=True):
#DO NOT CHANGE LINES 65-69
    if part == 'a' and is_years:
        xx = (xx - np.array([1960]*len(xx)))/40
        
    if part == 'a' and not is_years:
        xx = xx/20
    
    if part == 'a':
        phi_xx = np.array([[1, *[x ** j for j in range(1,6)]] for x in xx])
    
    if part == 'b':
        phi_xx = np.array([[1, *[np.exp(-((x - uj) ** 2) / 25.) for uj in range(1960, 2011, 5)]] for x in xx])
    
    if part == 'c':
        phi_xx = np.array([[1, *[np.cos(x / j) for j in range(1,6)]] for x in xx])
    
    if part == 'd':
        phi_xx = np.array([[1, *[np.cos(x / j) for j in range(1,26)]] for x in xx])

    return phi_xx

# Nothing fancy for outputs.
Y = republican_counts

# Find the regression weights using the Moore-Penrose pseudoinverse.
def find_weights(X,Y):
    w = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, Y))
    return w

def find_squared_error(X, Y, w):
    assert len(X) == len(Y), "Invalid arguments X, Y"
    return sum([(Y[i] - np.dot(X[i,:],w)) ** 2 for i in range(len(X))])

# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_years!!!!!
grid_years = np.linspace(1960, 2005, 200)

# Plot and report sum of squared error for each basis (years vs republicans)
for part in ['a', 'b', 'c', 'd']:
    plt.figure(4)

    # Plot years vs. republican_counts
    plt.plot(years, Y, 'o')

    # Make basis for training data and find best weights
    phiX = make_basis(years, part)
    w = find_weights(phiX, Y)
    
    # Calculate squared error
    err = find_squared_error(phiX, Y, w)
    
    # Plot a regression with weights
    grid_phiX = make_basis(grid_years, part)
    grid_Yhat = np.dot(w.T, grid_phiX.T)
    plt.plot(grid_years, grid_Yhat)

    plt.title(f'Least squares regression with basis function ({part})\n (squared error: {round(err,5)})')
    plt.xlabel("years")
    plt.ylabel("republicans")
    plt.savefig(f'4-1-{part}.png')
    plt.show()

# Only consider data before 1985
sunspot_counts = list(map(lambda i : i[1],
                      filter(lambda j : j[0] < last_year, zip(years, sunspot_counts))))
republican_counts = list(map(lambda i : i[1],
                         filter(lambda j : j[0] < last_year, zip(years, republican_counts))))
X = np.array(sunspot_counts)
Y = np.array(republican_counts)

grid_sunspots = np.linspace(0,160,200)

# Plot and report sum of squared error for each basis (sunspots vs republicans)
for part in ['a', 'c', 'd']:
    plt.figure(5)

    # Plot sunspots against republicans
    plt.plot(X, Y, 'o')

    # Make basis for training data and find best weights
    phiX = make_basis(X, part=part, is_years=False)
    w = find_weights(phiX, Y)
    
    # Calculate squared error
    err = find_squared_error(phiX, Y, w)
    
    # Plot a regression with weights
    grid_phiX = make_basis(grid_sunspots, part, is_years=False)
    grid_Yhat = np.dot(w.T, grid_phiX.T)
    plt.plot(grid_sunspots, grid_Yhat)

    plt.title(f'Least squares regression with basis function ({part})\n (squared error: {round(err,5)})')
    plt.xlabel("sunspots")
    plt.ylabel("republicans")
    plt.savefig(f'4-2-{part}.png')
    plt.show()
