#%%
import pandas as pd
import pandas as pd
import numpy as np
import re
import sys

# Read the CSV file
df = pd.read_csv(r'C:\Users\sauma\OneDrive\Desktop\EV_Market Analysis India\Dataset\EV_India vehicles.csv')
# Display the first few rows of the dataset
print(df.head())
# Display basic information about the dataset
print(df.info())
# Read the CSV file
df = pd.read_csv(r'C:\Users\sauma\OneDrive\Desktop\EV_Market Analysis India\Dataset\EV_India vehicles.csv')
# Display the first few rows of the dataset
print("First few rows of the original dataset:")
print(df.head().to_string())
# Display basic information about the dataset
print("\nInfo about the original dataset:")
print(df.info())
#%%
# Data Preprocessing Steps
# 1. Handle 'BootSpace' column
def clean_boot_space(value):
    if pd.isna(value) or value == 'na':
        return np.nan
    # Extract the first number if multiple values are present
    try:
        return float(str(value).split()[0])
    except ValueError:
        return np.nan

df['BootSpace'] = df['BootSpace'].apply(clean_boot_space)

# Fill missing values with median instead of mean (less affected by outliers)
df['BootSpace'].fillna(df['BootSpace'].median(), inplace=True)

# 2. Convert PriceRange to numeric
def extract_price(price_range):
    if isinstance(price_range, str):
        # Remove the Rupee symbol and any whitespace
        price_range = price_range.replace('\u20b9', '').strip()
        # Extract all numbers from the string
        numbers = re.findall(r'\d+\.?\d*', price_range)
        if numbers:
            if 'L' in price_range:
                return float(numbers[-1]) * 100000
            elif 'Cr' in price_range:
                return float(numbers[-1]) * 10000000
            else:
                return float(numbers[-1])
    return np.nan

df['Price'] = df['PriceRange'].apply(extract_price)

# 3. Convert Range to numeric
df['Range'] = df['Range'].str.extract('(\d+)').astype(float)

# 4. Handle 'Capacity' column
df['Capacity'] = df['Capacity'].str.extract('(\d+)').astype(float)

# 5. Encode categorical variables
categorical_columns = ['Style', 'Transmission', 'VehicleType']
df_encoded = pd.get_dummies(df, columns=categorical_columns)

# 6. Remove unnecessary columns
columns_to_drop = ['Car', 'PriceRange', 'BaseModel', 'TopModel']
df_cleaned = df_encoded.drop(columns=columns_to_drop)

# Display the first few rows of the preprocessed dataset
print("\nFirst few rows of the preprocessed dataset:")
print(df_cleaned.head().to_string())

# Display basic information about the preprocessed dataset
print("\nInfo about the preprocessed dataset:")
print(df_cleaned.info())

# Display summary statistics of the numeric columns
print("\nSummary statistics of numeric columns:")
print(df_cleaned.describe().to_string())

# Check for any remaining issues
print("\nColumns with null values:")
print(df_cleaned.isnull().sum())

print("\nUnique values in categorical columns:")
for col in df_cleaned.select_dtypes(include=['object']).columns:
    print(f"{col}: {df_cleaned[col].unique()}")

# Print the datatypes of each column
print("\nDatatypes of each column:")
print(df_cleaned.dtypes)

# Print unique values in the PriceRange column (before preprocessing)
print("\nUnique values in the original PriceRange column:")
print(df['PriceRange'].unique().tolist())

# Print the extracted Price values
print("\nExtracted Price values:")
print(df_cleaned['Price'].head(10).to_string())
#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Assuming df_cleaned is our preprocessed dataframe from the previous step

# 1. Select relevant features for segmentation
features_for_segmentation = ['Price', 'Range', 'BootSpace', 'Capacity']

# Create a new dataframe with only the selected features
df_segmentation = df_cleaned[features_for_segmentation].copy()

# 2. Handle any remaining missing values
df_segmentation.fillna(df_segmentation.median(), inplace=True)

# 3. Normalize the features
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_segmentation), columns=df_segmentation.columns)

# 4. Perform K-means clustering
# We'll use 3 segments, but you can adjust this number based on your needs
kmeans = KMeans(n_clusters=3, random_state=42)
df_cleaned['Segment'] = kmeans.fit_predict(df_scaled)

# 5. Analyze the segments
segment_analysis = df_cleaned.groupby('Segment').agg({
    'Price': ['mean', 'min', 'max'],
    'Range': ['mean', 'min', 'max'],
    'BootSpace': ['mean', 'min', 'max'],
    'Capacity': ['mean', 'min', 'max']
})

# 6. Create meaningful segment labels
def label_segment(row):
    if row['Price']['mean'] <= segment_analysis['Price']['mean'].min():
        return 'Budget'
    elif row['Price']['mean'] >= segment_analysis['Price']['mean'].max():
        return 'Luxury'
    else:
        return 'Mid-range'

segment_analysis['Segment_Label'] = segment_analysis.apply(label_segment, axis=1)

# Map the labels back to the main dataframe
segment_label_map = segment_analysis['Segment_Label'].to_dict()
df_cleaned['Segment_Label'] = df_cleaned['Segment'].map(segment_label_map)

# 7. Display the results
print("Segment Analysis:")
print(segment_analysis)

print("\nSample of segmented data:")
print(df_cleaned[['Price', 'Range', 'BootSpace', 'Capacity', 'Segment', 'Segment_Label']].head(10))

# 8. Calculate the percentage of vehicles in each segment
segment_distribution = df_cleaned['Segment_Label'].value_counts(normalize=True) * 100
print("\nSegment Distribution:")
print(segment_distribution)

# 9. Identify the most common vehicle type in each segment
# First, find the vehicle type columns (they should start with 'VehicleType_')
vehicle_type_columns = [col for col in df_cleaned.columns if col.startswith('VehicleType_')]

# Function to get the most common vehicle type
def get_most_common_vehicle_type(group):
    vehicle_type_sums = group[vehicle_type_columns].sum()
    return vehicle_type_sums.idxmax().replace('VehicleType_', '')

common_vehicle_type = df_cleaned.groupby('Segment_Label').apply(get_most_common_vehicle_type)
print("\nMost common vehicle type in each segment:")
print(common_vehicle_type)

# 10. Calculate average range for each segment
avg_range_by_segment = df_cleaned.groupby('Segment_Label')['Range'].mean().sort_values(ascending=False)
print("\nAverage range by segment:")
print(avg_range_by_segment)

# 11. Print column names (for debugging)
print("\nAll column names in the dataset:")
print(df_cleaned.columns.tolist())

# 12. Print unique values in Segment_Label (for debugging)
print("\nUnique values in Segment_Label:")
print(df_cleaned['Segment_Label'].unique())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Assuming df_cleaned is your preprocessed dataframe

# Select relevant features for segmentation
features_for_segmentation = ['Price', 'Range', 'BootSpace', 'Capacity']
df_segmentation = df_cleaned[features_for_segmentation].copy()

# Handle any remaining missing values
df_segmentation.fillna(df_segmentation.median(), inplace=True)

# Normalize the features
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_segmentation), columns=df_segmentation.columns)

# Elbow Method
inertias = []
silhouette_scores = []
K = range(2, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(df_scaled, kmeans.labels_))

# Plot the Elbow Curve
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(K, inertias, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')

plt.subplot(1, 2, 2)
plt.plot(K, silhouette_scores, 'rx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal k')

plt.tight_layout()
plt.savefig('ev_elbow_method.png')
plt.close()

print("Elbow method plot saved as 'ev_elbow_method.png'")

# Choose the optimal k (you may need to adjust this based on the plot)
optimal_k = 3  # Example: let's say we chose 3 based on the elbow curve

# Perform K-means clustering with the optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df_cleaned['Segment'] = kmeans.fit_predict(df_scaled)

# Analyze the segments
segment_analysis = df_cleaned.groupby('Segment').agg({
    'Price': ['mean', 'min', 'max'],
    'Range': ['mean', 'min', 'max'],
    'BootSpace': ['mean', 'min', 'max'],
    'Capacity': ['mean', 'min', 'max']
})

# Create meaningful segment labels
def label_segment(row):
    if row['Price']['mean'] <= segment_analysis['Price']['mean'].min():
        return 'Budget'
    elif row['Price']['mean'] >= segment_analysis['Price']['mean'].max():
        return 'Luxury'
    else:
        return 'Mid-range'

segment_analysis['Segment_Label'] = segment_analysis.apply(label_segment, axis=1)

# Map the labels back to the main dataframe
segment_label_map = segment_analysis['Segment_Label'].to_dict()
df_cleaned['Segment_Label'] = df_cleaned['Segment'].map(segment_label_map)

# Display the results
print("Segment Analysis:")
print(segment_analysis)

print("\nSample of segmented data:")
print(df_cleaned[['Price', 'Range', 'BootSpace', 'Capacity', 'Segment', 'Segment_Label']].head(10))

# Calculate the percentage of vehicles in each segment
segment_distribution = df_cleaned['Segment_Label'].value_counts(normalize=True) * 100
print("\nSegment Distribution:")
print(segment_distribution)

# Visualize the clusters
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df_cleaned['Range'], df_cleaned['Price'], c=df_cleaned['Segment'], cmap='viridis')
plt.xlabel('Range (km)')
plt.ylabel('Price (in lakhs)')
plt.title(f'K-means Clustering of EV Market (k={optimal_k})')
plt.colorbar(scatter, label='Segment')
for i, label in enumerate(segment_analysis.index):
    segment_label = segment_analysis.loc[i, 'Segment_Label']
    plt.annotate(segment_label, 
                 (df_cleaned[df_cleaned['Segment'] == i]['Range'].mean(),
                  df_cleaned[df_cleaned['Segment'] == i]['Price'].mean()),
                 textcoords="offset points",
                 xytext=(0,10),
                 ha='center')
plt.savefig('ev_kmeans_clusters.png')
plt.close()

print("K-means clustering plot saved as 'ev_kmeans_clusters.png'")
# Additional analysis as in the original code
vehicle_type_columns = [col for col in df_cleaned.columns if col.startswith('VehicleType_')]
def get_most_common_vehicle_type(group):
    vehicle_type_sums = group[vehicle_type_columns].sum()
    return vehicle_type_sums.idxmax().replace('VehicleType_', '')

common_vehicle_type = df_cleaned.groupby('Segment_Label').apply(get_most_common_vehicle_type)
print("\nMost common vehicle type in each segment:")
print(common_vehicle_type)

avg_range_by_segment = df_cleaned.groupby('Segment_Label')['Range'].mean().sort_values(ascending=False)
print("\nAverage range by segment:")
print(avg_range_by_segment)
#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Assuming df_cleaned is our segmented dataframe from the previous step

# Set the style for all plots
plt.style.use('ggplot')  # Using a built-in style that's similar to seaborn

# Function to save figures
def save_figure(fig, filename):
    fig.savefig(filename)
    plt.close(fig)
# 3. Scatter Plot of Range vs Price, colored by Segment
fig, ax = plt.subplots(figsize=(12, 8))
segments = df_cleaned['Segment_Label'].unique()
colors = plt.cm.rainbow(np.linspace(0, 1, len(segments)))
for segment, color in zip(segments, colors):
    mask = df_cleaned['Segment_Label'] == segment
    ax.scatter(df_cleaned.loc[mask, 'Range'], df_cleaned.loc[mask, 'Price'], 
               c=[color], label=segment, alpha=0.6)
ax.set_xlabel('Range (km)')
ax.set_ylabel('Price (in lakhs)')
ax.set_title('Range vs Price by Segment')
ax.legend()
save_figure(fig, 'ev_range_vs_price.png')

# 4. Bar Plot of Average Range by Segment
fig, ax = plt.subplots(figsize=(10, 6))
avg_range = df_cleaned.groupby('Segment_Label')['Range'].mean().sort_values(ascending=False)
ax.bar(avg_range.index, avg_range.values)
ax.set_title('Average Range by Segment')
ax.set_xlabel('Segment')
ax.set_ylabel('Average Range (km)')
save_figure(fig, 'ev_average_range.png')

# 5. Heatmap of Feature Correlations
fig, ax = plt.subplots(figsize=(10, 8))
corr = df_cleaned[['Price', 'Range', 'BootSpace', 'Capacity']].corr()
im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(np.arange(len(corr.columns)))
ax.set_yticks(np.arange(len(corr.columns)))
ax.set_xticklabels(corr.columns)
ax.set_yticklabels(corr.columns)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black")
ax.set_title('Correlation Heatmap of EV Features')
fig.colorbar(im)
fig.tight_layout()
save_figure(fig, 'ev_correlation_heatmap.png')

# 7. Grouped Bar Chart of Vehicle Types by Segment
vehicle_type_columns = [col for col in df_cleaned.columns if col.startswith('VehicleType_')]
vehicle_type_data = df_cleaned.groupby('Segment_Label')[vehicle_type_columns].sum()
vehicle_type_data = vehicle_type_data.div(vehicle_type_data.sum(axis=1), axis=0)
vehicle_type_data.columns = [col.replace('VehicleType_', '') for col in vehicle_type_data.columns]

fig, ax = plt.subplots(figsize=(12, 6))
vehicle_type_data.plot(kind='bar', stacked=True, ax=ax)
ax.set_title('Vehicle Type Distribution by Segment')
ax.set_xlabel('Segment')
ax.set_ylabel('Proportion')
ax.legend(title='Vehicle Type', bbox_to_anchor=(1.05, 1), loc='upper left')
fig.tight_layout()
save_figure(fig, 'ev_vehicle_type_distribution.png')
print("All plots have been saved as PNG files.")
#%%
# Profiling and Describing Potential Segments

# Function to profile each segment
def segment_profile(df, segment_label, numeric_columns, vehicle_type_columns):
    """
    Creates a profile for a specific segment including statistical summaries of numeric columns 
    and vehicle type distribution.
    """
    # Filter the data for the given segment
    segment_data = df[df['Segment_Label'] == segment_label]
    
    # Summary statistics of numeric columns
    print(f"\n=== {segment_label} Segment Profile ===")
    print("\nNumeric Feature Summary:")
    print(segment_data[numeric_columns].describe().to_string())
    
    # Vehicle type distribution
    print("\nVehicle Type Distribution:")
    vehicle_type_distribution = segment_data[vehicle_type_columns].sum()
    vehicle_type_distribution = vehicle_type_distribution / vehicle_type_distribution.sum()
    print(vehicle_type_distribution.to_string())
    
    # Key insights
    print("\nKey Insights:")
    print(f"- Average Price: ₹{segment_data['Price'].mean():.2f}")
    print(f"- Average Range: {segment_data['Range'].mean():.2f} km")
    print(f"- Average Boot Space: {segment_data['BootSpace'].mean():.2f} liters")
    print(f"- Average Capacity: {segment_data['Capacity'].mean():.2f} kWh")
    print(f"- Most common vehicle type: {vehicle_type_distribution.idxmax()}")

# List of numeric columns and vehicle type columns
numeric_columns = ['Price', 'Range', 'BootSpace', 'Capacity']
vehicle_type_columns = [col for col in df_cleaned.columns if col.startswith('VehicleType_')]

# Loop through each segment and profile it
for segment in df_cleaned['Segment_Label'].unique():
    segment_profile(df_cleaned, segment, numeric_columns, vehicle_type_columns)

# Summary of segment comparison
print("\n=== Segment Comparison ===")
segment_summary = df_cleaned.groupby('Segment_Label')[numeric_columns].mean()
print(segment_summary)

# Visualizing segment comparison

# 1. Average price by segment
fig, ax = plt.subplots(figsize=(8, 6))
segment_summary['Price'].plot(kind='bar', color='skyblue', ax=ax)
ax.set_title('Average Price by Segment')
ax.set_xlabel('Segment')
ax.set_ylabel('Average Price (₹)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('ev_avg_price_by_segment.png')
plt.close()

# 2. Average range by segment
fig, ax = plt.subplots(figsize=(8, 6))
segment_summary['Range'].plot(kind='bar', color='lightgreen', ax=ax)
ax.set_title('Average Range by Segment')
ax.set_xlabel('Segment')
ax.set_ylabel('Average Range (km)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('ev_avg_range_by_segment.png')
plt.close()

# 3. Average boot space by segment
fig, ax = plt.subplots(figsize=(8, 6))
segment_summary['BootSpace'].plot(kind='bar', color='lightcoral', ax=ax)
ax.set_title('Average Boot Space by Segment')
ax.set_xlabel('Segment')
ax.set_ylabel('Average Boot Space (liters)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('ev_avg_bootspace_by_segment.png')
plt.close()

# 4. Average capacity by segment
fig, ax = plt.subplots(figsize=(8, 6))
segment_summary['Capacity'].plot(kind='bar', color='gold', ax=ax)
ax.set_title('Average Capacity by Segment')
ax.set_xlabel('Segment')
ax.set_ylabel('Average Capacity (kWh)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('ev_avg_capacity_by_segment.png')
plt.close()

print("\nSegment profiles and comparisons have been generated. Key figures saved as PNG images.")

#%%
# Selection of Target Segment

# Criteria-based Selection of Target Segment
# Example: Targeting customers who are price-sensitive but also need a reasonable range

def select_target_segment(segment_summary, price_weight=0.6, range_weight=0.4):
    """
    Selects the target segment based on a weighted score that balances price and range.
    You can adjust the weights to prioritize either factor.
    """
    # Normalize the features (lower is better for price, higher is better for range)
    segment_summary['Normalized_Price'] = (segment_summary['Price'] - segment_summary['Price'].min()) / (segment_summary['Price'].max() - segment_summary['Price'].min())
    segment_summary['Normalized_Range'] = (segment_summary['Range'] - segment_summary['Range'].min()) / (segment_summary['Range'].max() - segment_summary['Range'].min())

    # Calculate the weighted score
    segment_summary['Weighted_Score'] = price_weight * (1 - segment_summary['Normalized_Price']) + range_weight * segment_summary['Normalized_Range']
    
    # Select the segment with the highest weighted score
    target_segment = segment_summary['Weighted_Score'].idxmax()
    
    return target_segment

# Assuming 'segment_summary' contains average metrics per segment (from previous analysis)
target_segment = select_target_segment(segment_summary)

print(f"Selected Target Segment: {target_segment}")

# Additional Details about the Selected Target Segment
print("\nDetailed Profile of the Target Segment:")
segment_profile(df_cleaned, target_segment, numeric_columns, vehicle_type_columns)

# Visualization of the Target Segment
fig, ax = plt.subplots(figsize=(12, 6))
segment_summary['Weighted_Score'].plot(kind='bar', color='dodgerblue', ax=ax)
ax.set_title('Segment Scores for Target Selection')
ax.set_xlabel('Segment')
ax.set_ylabel('Weighted Score')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('target_segment_score.png')
plt.close()

print("Target segment score plot saved as 'target_segment_score.png'")
#%%
import matplotlib.pyplot as plt

# Define your marketing mix strategies
marketing_mix = {
    'Product Features': [
        'High battery capacity',
        'Extended range',
        'Smart connectivity features',
        'Stylish design'
    ],
    'Pricing Strategy': [
        'Competitive pricing',
        'Introductory discounts',
        'Loyalty programs'
    ],
    'Distribution Channels': [
        'Online sales',
        'Local dealerships',
        'EV-specialized stores'
    ],
    'Promotional Tactics': [
        'Social media advertising',
        'Content marketing',
        'Influencer partnerships'
    ]
}

# Create subplots for each element of the marketing mix
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
fig.suptitle('Customized Marketing Mix for Target Segment', fontsize=16)

# Product Features
axs[0, 0].barh(marketing_mix['Product Features'], range(len(marketing_mix['Product Features'])), color='skyblue')
axs[0, 0].set_title('Product Features')
axs[0, 0].set_xlabel('Importance')
axs[0, 0].invert_yaxis()  # Reverse the order

# Pricing Strategy
axs[0, 1].barh(marketing_mix['Pricing Strategy'], range(len(marketing_mix['Pricing Strategy'])), color='lightgreen')
axs[0, 1].set_title('Pricing Strategy')
axs[0, 1].set_xlabel('Importance')
axs[0, 1].invert_yaxis()

# Distribution Channels
axs[1, 0].barh(marketing_mix['Distribution Channels'], range(len(marketing_mix['Distribution Channels'])), color='salmon')
axs[1, 0].set_title('Distribution Channels')
axs[1, 0].set_xlabel('Importance')
axs[1, 0].invert_yaxis()

# Promotional Tactics
axs[1, 1].barh(marketing_mix['Promotional Tactics'], range(len(marketing_mix['Promotional Tactics'])), color='gold')
axs[1, 1].set_title('Promotional Tactics')
axs[1, 1].set_xlabel('Importance')
axs[1, 1].invert_yaxis()

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('ev_marketing_mix.png')
plt.show()

print("Marketing mix visualization saved as 'ev_marketing_mix.png'")

# %%
