import matplotlib.pyplot as plt
import csv

# Read the data from the CSV file
with open('Testing.csv', 'r') as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)  # Read the header
    data = list(csv_reader)    # Read the rest of the data

# Parse the data
diseases = [row[-1] for row in data]
symptom_counts = [sum(int(x) for x in row[:-1]) for row in data]

# Create the bar graph
plt.figure(figsize=(15, 10))
bars = plt.bar(range(len(diseases)), symptom_counts)

# Customize the graph
plt.title('Number of Symptoms per Disease', fontsize=16)
plt.xlabel('Diseases', fontsize=12)
plt.ylabel('Number of Symptoms', fontsize=12)
plt.xticks(range(len(diseases)), diseases, rotation=90, ha='right')

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height}',
             ha='center', va='bottom')

# Adjust layout and display the graph
plt.tight_layout()
plt.show()