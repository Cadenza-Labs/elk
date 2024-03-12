# Sample data
from matplotlib import pyplot as plt

num_templates = [2, 8, 32]
probe_accuracies = {
    "Probe k=num_random_words": [0.95, 0.95, 0.94],
    "Probe k=2*num_random_words": [0.93, 0.93, 0.93],
    "Probe k=3*num_random_words": [0.93, 0.93, 0.92],
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
