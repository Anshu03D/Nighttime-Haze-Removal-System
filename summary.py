import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("metrics/results.csv")

print("\n===== OVERALL RESULTS =====\n")

print("Average Brightness Gain (%) :",
      round(df["Brightness Gain (%)"].mean(), 2))

print("Average Contrast Gain (%) :",
      round(df["Contrast Gain (%)"].mean(), 2))

print("Average Entropy :",
      round(df["Entropy"].mean(), 2))

print("Average Processing Time (s) :",
      round(df["Processing Time (s)"].mean(), 2))
plt.figure(figsize=(8,4))

plt.plot(
    df["Brightness Gain (%)"],
    marker="o"
)

plt.title("Brightness Gain Across Test Images")
plt.xlabel("Image Number")
plt.ylabel("Brightness Gain (%)")

plt.savefig("report_results/brightness_gain.png")
plt.show()