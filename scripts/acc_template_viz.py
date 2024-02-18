# Sample data
from matplotlib import pyplot as plt

num_templates = [2, 8, 32]
probe_accuracies = {
    "Probe k=random_words": [0.7, 0.75, 0.8, 0.85],
    # 'Probe k=4': [0.65, 0.7, 0.78, 0.82],
    "Probe k=2*random_words": [0.6, 0.68, 0.76, 0.84],
    "Probe k=3*random_words": [0.55, 0.63, 0.71, 0.79],
}

# Line styles for different probes
line_styles = ["solid", "dashed", "dotted", "dashdot"]

# Plotting
plt.figure(figsize=(8, 6))
for i, (probe, accuracies) in enumerate(probe_accuracies.items()):
    plt.plot(
        num_templates, accuracies, marker="o", linestyle=line_styles[i], label=probe
    )

plt.xlabel("Number of random words")
plt.ylabel("Accuracy")
plt.title("Probe Accuracy")
plt.legend()
plt.grid(True)

# Save the plot as a PDF file
plt.savefig("probe_accuracy_vs_templates_named.pdf")

plt.show()
