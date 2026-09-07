import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score

bright = pd.read_csv("results/bright_results.csv")
dark = pd.read_csv("results/dark_results.csv")
dim = pd.read_csv("results/dim_results.csv")
print(bright.columns)



summary = pd.DataFrame({
    "Lighting": ["Bright", "Dark", "Dim"],
    "Average FPS": [
        bright["FPS"].mean(),
        dark["FPS"].mean(),
        dim["FPS"].mean()
    ],
    "Average Confidence": [
        bright["Confidence"].mean(),
        dark["Confidence"].mean(),
        dim["Confidence"].mean()
    ],
    "Detection Rate": [
        (bright["Detected"] > 0).mean(),
        (dark["Detected"] > 0).mean(),
        (dim["Detected"] > 0).mean()
    ]
})

plt.figure()

plt.bar(summary["Lighting"], summary["Average FPS"])

plt.xlabel("Lighting Condition")
plt.ylabel("Average FPS")
plt.title("Average FPS Under Different Lighting Conditions")

plt.show()

plt.figure()

plt.bar(summary["Lighting"], summary["Average Confidence"])

plt.xlabel("Lighting Condition")
plt.ylabel("Average Confidence")
plt.title("Average YOLO Confidence Under Different Lighting Conditions")

plt.show()

plt.figure()

plt.bar(summary["Lighting"], summary["Detection Rate"])

plt.xlabel("Lighting Condition")
plt.ylabel("Detection Rate (%)")
plt.title("Detection Rate Under Different Lighting Conditions")

plt.show()



###################################

from sklearn.metrics import precision_score, recall_score, f1_score

bright_detected = (bright["Detected"] > 0).astype(int)
dim_detected = (dim["Detected"] > 0).astype(int)
dark_detected = (dark["Detected"] > 0).astype(int)

# Bright
precision = precision_score(
    bright["Ground_Truth"],
    bright_detected
)

recall = recall_score(
    bright["Ground_Truth"],
    bright_detected
)

f1 = f1_score(
    bright["Ground_Truth"],
    bright_detected
)

# Dim
precision1 = precision_score(
    dim["Ground_Truth"],
    dim_detected
)

recall1 = recall_score(
    dim["Ground_Truth"],
    dim_detected
)

f1_1 = f1_score(
    dim["Ground_Truth"],
    dim_detected
)

# Dark
precision2 = precision_score(
    dark["Ground_Truth"],
    dark_detected
)

recall2 = recall_score(
    dark["Ground_Truth"],
    dark_detected
)

f1_2 = f1_score(
    dark["Ground_Truth"],
    dark_detected
)




summary = pd.DataFrame({
    "Lighting": ["Bright", "Dim", "Dark"],
    "Average FPS": [
        bright["FPS"].mean(),
        dim["FPS"].mean(),
        dark["FPS"].mean()
    ],
    "Average Confidence": [
        bright["Confidence"].mean(),
        dim["Confidence"].mean(),
        dark["Confidence"].mean()
    ],
    "Precision": [
        precision,
        precision1,
        precision2
    ],
    "Recall": [
        recall,
        recall1,
        recall2
    ],
    "F1 Score": [
        f1,
        f1_1,
        f1_2
    ]
})

print("\n=== TapoVision Results ===")
print(summary)