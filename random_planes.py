__author__ = ["David Cardona-Vasquez"]
__copyright__ = "Copyright 2024, Graz University of Technology"
__credits__ = ["David Cardona-Vasquez"]
__license__ = "MIT"
__maintainer__ = "David Cardona-Vasquez"
__status__ = "Development"


import matplotlib.pyplot as plt
import seaborn as sns


import numpy as np
import pandas as pd

def generate_random_line(full_random=False):
    """Generate a random line passing through two random points in the range [0, 1] x [0, 1]."""
    points = np.random.rand(2, 2)  # Generate two random points in the range [0, 1] x [0, 1]
    # Compute the coefficients of the line passing through the two points using the equation of a line

    if full_random:
        slope = (points[1, 1] - points[0, 1]) / (points[1, 0] - points[0, 0])
    else:
        slope = 1
    intercept = points[0, 1] - slope * points[0, 0]
    return slope, intercept


def classify_points(points, lines, col_names={'x': 'cap_factor', 'y': 'demand', 'label':'basis'}):
    """Classify points based on which side of each line they lie."""
    labels = []
    groups = [[] for _ in range(len(lines) + 1)]  # Initialize empty lists for each group
    for i, point in points.iterrows():
        for i, line in enumerate(lines):
            # Check which side of the line the point lies on
            if point[col_names['y']] < line[0]*point[col_names['x']] + line[1]:  # Point lies below the line
                labels.append(i+1)
                groups[i+1].append(point)
                break
        else:
            labels.append(0)  # Point lies above or on all lines
            groups[0].append(point)  # Point lies above or on all lines

    points[col_names['label']] = labels

    return groups

def get_effective_lines(points, lines):
    """Return the lines that effectively divide the points into different groups."""
    id_effective_lines = points['basis'].unique().tolist()
    effective_lines = []
    for i in id_effective_lines:
        if i != 0:
            effective_lines.append(lines[i - 1])

    return effective_lines


if __name__ == "__main__":
    # Generate a set of random points
    # np.random.see
    num_points = 500
    points = np.random.rand(num_points, 2)

    # Generate a random number of lines (between 1 and 5)
    num_lines = 3
    lines = [generate_random_line(False) for _ in range(num_lines)]

    #lines = sorted(lines, key=lambda line: line[1])

    df_points = pd.DataFrame(points)
    df_points.columns = ['x', 'y']
    df_points['basis'] = -1

    # Classify points based on which side of each line they lie
    groups = classify_points(df_points, lines, col_names={'x': 'x', 'y': 'y', 'label':'basis'})



    # Print the points in each group
    for i, group in enumerate(groups):
        print(f"Group {i+1}: {group}")

    sns.scatterplot(data=df_points, x='x', y='y', hue='basis')
    # plt.scatter(df_points['x'], df_points['y'], c=df_points['label'])
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    x_values = np.linspace(0, 1, 100)
    # Plot each line
    for i, line in enumerate(lines):
        slope, intercept = line
        y_values = slope*x_values + intercept
        plt.plot(x_values, y_values, color=colors[i % len(colors)])
        plt.ylim(0, 1)

    plt.title('All Lines')
    plt.show()

    effective_lines = get_effective_lines(df_points, lines)
    sns.scatterplot(data=df_points, x='x', y='y', hue='basis')
    # plt.scatter(df_points['x'], df_points['y'], c=df_points['label'])
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    x_values = np.linspace(0, 1, 100)
    # Plot each line
    for i, line in enumerate(effective_lines):
        slope, intercept = line
        y_values = slope*x_values + intercept
        plt.plot(x_values, y_values, color=colors[i % len(colors)])
        plt.ylim(0, 1)

    plt.title('Effective Lines')
    plt.show()



