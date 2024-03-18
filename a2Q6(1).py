import matplotlib.pyplot as plt

# Data points
a = [4, 5, 12, 29, 30, 36, 36, 54, 58, 70, 72, 76, 78, 82, 87, 90, 90, 92, 95, 98]
b = [49, 4, 28, 18, 65, 32, 1, 29, 76, 12, 26, 55, 4, 15, 95, 70, 55, 84, 14, 21]
labels = [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]

# Scatter plot
plt.scatter(a, b, c=labels, cmap='bwr')
plt.xlabel('a')
plt.ylabel('b')
plt.title('Scatter Plot of 2-D Data')
plt.show()

