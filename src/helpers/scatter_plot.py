import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# Generate random data
num_points = 1000000
hue = np.random.choice(['orange', 'cyan', 'purple'], size=num_points)

# Adjust x and y based on hue
x = np.random.randn(num_points)
y = np.random.randn(num_points)
x[hue == 'orange'] += 3.2
y[hue == 'orange'] += 1.8
x[hue == 'cyan'] -= 1.2
y[hue == 'cyan'] += 3.3
x[hue == 'purple'] -= 2.2
y[hue == 'purple'] -= 2.2

# Plotting parameters
point_size = 0.09
alpha = 0.1

# Create scatter plots for each hue
plt.figure(figsize=(10, 6))
plt.scatter(x[hue == 'orange'], y[hue == 'orange'], c='orange', s=point_size, alpha=alpha, label='Orange')
plt.scatter(x[hue == 'cyan'], y[hue == 'cyan'], c='cyan', s=point_size, alpha=alpha, label='Cyan')
plt.scatter(x[hue == 'purple'], y[hue == 'purple'], c='purple', s=point_size, alpha=alpha, label='Purple')

# Create color patches for legend
orange_patch = mpatches.Patch(color='orange', label='Sport')
cyan_patch = mpatches.Patch(color='cyan', label='Business')
purple_patch = mpatches.Patch(color='purple', label='Politics')

# Customize the plot
plt.title("Yahoo! Answer")
plt.legend(handles=[orange_patch, cyan_patch, purple_patch], title_fontsize='large', frameon=True, facecolor='white',bbox_to_anchor=(0.95, 1))
plt.axis('off')

# Show the plot
plt.show()
