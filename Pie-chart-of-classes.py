train_paths = []
train_labels = []

for label in os.listdir(train_path):
    for image in os.listdir(train_path+'/'+label):
        train_paths.append(train_path+label+'/'+image)
        train_labels.append(label)


test_paths = []
test_labels = []

for label in os.listdir(test_path):
    for image in os.listdir(test_path+'/'+label):
        test_paths.append(test_path+label+'/'+image)
        test_labels.append(label)

        
colors = ['#aeaeee', '#ea56dc']

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

# Plot pie chart for test data
ax1.pie([len([x for x in test_labels if x=='parasitized']),
         len([x for x in test_labels if x=='uninfected'])],
        labels=['parasitized', 'uninfected'],
        colors=colors, autopct='%.1f%%', explode=(0.025,0.025))
ax1.set_title('Pie Chart of Test Data')

# Plot pie chart for train data
ax2.pie([len([x for x in train_labels if x=='parasitized']),
         len([x for x in train_labels if x=='uninfected'])],
        labels=['parasitized', 'uninfected'],
        colors=colors, autopct='%.1f%%', explode=(0.025,0.025))
ax2.set_title('Pie Chart of Train Data')

# Adjust layout
plt.tight_layout()
plt.show()
