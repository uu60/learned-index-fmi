import os
import pandas as pd
import matplotlib.pyplot as plt

# Directory where your CSV files are located
directory = 'record/'  # Replace with your actual directory


# Function to sample one point per 0.05 interval
def sample_points(df, interval=0.05):
    # Round epsilon values to the nearest interval and drop duplicates
    df['epsilon_rounded'] = (df['epsilon'] / interval).round() * interval
    sampled_df = df.drop_duplicates(subset='epsilon_rounded')
    return sampled_df


# Create a plot for each CSV file in the directory
plt.figure(figsize=(10, 6))

for i, filename in enumerate(os.listdir(directory)):
    if filename.endswith('.csv'):
        filepath = os.path.join(directory, filename)

        # Load the CSV file
        df = pd.read_csv(filepath)

        # Sample points at 0.05 intervals
        sampled_df = sample_points(df)

        # Plot each file's data
        plt.plot(sampled_df['epsilon'], sampled_df['max_error'], marker='o', linestyle='-')

# Set plot labels and title
plt.xlabel('Epsilon')
plt.ylabel('Max Error')
plt.title('Max Error vs Epsilon for All CSV Files (Log Scale)')
plt.grid(True)
plt.yscale('log')  # Set the y-axis to log scale

# Define x-axis ticks every 0.05
# plt.xticks(np.arange(0, 1.05, 0.05))

plt.legend(loc='best')
plt.show()