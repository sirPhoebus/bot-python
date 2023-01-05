import matplotlib.pyplot as plt

# Data for the first series (70-75%)
x1 = ['A', 'B', 'C']
y1 = [65, 80, 99]

# Data for the second series (80-85%)
x2 = ['A', 'B', 'C']
y2 = [80, 82, 84]

# Data for the third series (95-99%)
x3 = ['A', 'B', 'C']
y3 = [95, 97, 99]

# Create the plot
fig, ax = plt.subplots()

# Plot the first series
ax.bar(x1, y1, color='blue')


# Show the plot

plt.show()
