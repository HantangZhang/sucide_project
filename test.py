import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 3)

for i, (C, axes_row) in enumerate(zip((1, 0.1, 0.01), axes)):

    print(i)
    print((C, axes_row))