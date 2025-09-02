import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
import numpy as np

class ROISelector:

    def __init__(self, image):
        self.image = image
        self.coords = None
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.image.T, cmap='viridis', vmax=0.1 * self.image.max()) 
        # Saturation at 10% of the max value is used.
        # The image is transposed for visualization purpose.
        self.selector = PolygonSelector(self.ax, self.onselect)
        plt.title("Draw ROI, then close the window.")
        plt.show()

    def onselect(self, verts):
        self.coords = verts

    def get_mask(self):
        if self.coords is None:
            return None
        
        ny, nx = self.image.shape
        y_grid, x_grid = np.mgrid[:ny, :nx]
        points = np.vstack((x_grid.ravel(), y_grid.ravel())).T
        path = Path(self.coords)
        mask = path.contains_points(points).reshape((ny, nx))

        return mask.T # because we transposed the image