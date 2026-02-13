import numpy as np

def generate_heatmap(trajectory, ideal_path):

    heatmap = []

    for point in trajectory:
        dist = min(
            [np.linalg.norm(np.array(point)-np.array(p)) for p in ideal_path]
        )

        if dist < 10:
            heatmap.append(("green", point))
        elif dist < 25:
            heatmap.append(("yellow", point))
        else:
            heatmap.append(("red", point))

    return heatmap
