import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode

np.random.seed(42) # For reproducability
population_size= 10_000

# Skewed toward high efficiency (80% high efficiency, 20% low efficiency)
high_efficiency=np.random.uniform(0.6,1.0 + 0.01,int(0.8*population_size))
low_efficiency=np.random.uniform(0.1,0.6,int(0.2*population_size))
population_high_skew=np.concatenate([low_efficiency,high_efficiency])

low_efficiency=np.random.uniform(0.1,0.6,int(0.8*population_size))
high_efficiency=np.random.uniform(0.6,1.0 + 0.01,int(0.2*population_size))
population_low_skew=np.concatenate([low_efficiency,high_efficiency])

high_efficiency = np.random.uniform(0.6, 1 + 0.01, int(0.5 * population_size))
low_efficiency = np.random.uniform(0.1, 0.6, int(0.5 * population_size))
population_balanced = np.concatenate([low_efficiency,high_efficiency])

sample_size=60
no_of_samples=1000

def sample_means_distribution(population,sample_size,no_of_samples):
    sample_means=[np.mean(np.random.choice(population,sample_size,replace=False)) for _ in range(no_of_samples)]
    return sample_means

high_skew_sample_means= sample_means_distribution(population_high_skew,sample_size,no_of_samples)
low_skew_sample_means=sample_means_distribution(population_low_skew,sample_size,no_of_samples)
balanced_skew_sample_means=sample_means_distribution(population_balanced,sample_size,no_of_samples)

datasets= [
    (population_high_skew,"High Efficiency Skew(Population)"),
    (high_skew_sample_means, "High Efficiency Skew(Sample Means)"),
    (population_low_skew, "Low Efficiency Skew(Population)"),
    (low_skew_sample_means, "Low Efficiency Skew(Sample Means)"),
    (population_balanced, "Balanced Efficiency(Population)"),
    (balanced_skew_sample_means, "Balanced Efficiency(Sample Means)"),
]

# plt.figure(figsize=(10,8))
# for i , (population_data,title) in enumerate(datasets):
#     plt.subplot(3,2,i+1)
#     sns.histplot(population_data,bins=30,kde=True,color='C' + str(i % 3))
#     plt.title(title)
#     plt.xlabel("Efficiency")
#     plt.ylabel("Frequency")

# plt.tight_layout()
# plt.show()

results={}
for i , (population_data,name) in enumerate(datasets):
    mean = np.mean(population_data)
    median = np.median(population_data)
    counts, bins = np.histogram(population_data, bins=30)
    mode_bin = bins[np.argmax(counts)]
    mode_value = (mode_bin + bins[np.argmax(counts) + 1]) / 2
    std_dev = np.std(population_data)
    results[name] = {
        "Mean": mean,
        "Median": median,
        "Mode": mode_value,
        "Standard Deviation": std_dev,
    }


for name, stats in results.items():
    print(f"Statistics for {name}:")
    for stat_name, value in stats.items():
        print(f"  {stat_name}: {value:.3f}")
    print()

m_arr = np.array([1,1,1,0,2,2,2,2,2,2,20])
print(mode(m_arr).mode)


# Statistics for High Efficiency Skew(Population):
#   Mean: 0.711
#   Median: 0.751
#   Mode: 0.631
#   Standard Deviation: 0.221

# Statistics for High Efficiency Skew(Sample Means):
#   Mean: 0.711
#   Median: 0.713
#   Mode: 0.719
#   Standard Deviation: 0.029

# Statistics for Low Efficiency Skew(Population):
#   Mean: 0.443
#   Median: 0.413
#   Mode: 0.540
#   Standard Deviation: 0.231

# Statistics for Low Efficiency Skew(Sample Means):
#   Mean: 0.443
#   Median: 0.443
#   Mode: 0.440
#   Standard Deviation: 0.031

# Statistics for Balanced Efficiency(Population):
#   Mean: 0.577
#   Median: 0.600
#   Mode: 0.995
#   Standard Deviation: 0.266

# Statistics for Balanced Efficiency(Sample Means):
#   Mean: 0.575
#   Median: 0.576
#   Mode: 0.582
#   Standard Deviation: 0.033

