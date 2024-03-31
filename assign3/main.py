from tkinter import Tk
from tkinter.filedialog import askopenfilename
import cv2
import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def create_gaussian_filter(x_coord, y_coord, sigma):
    return np.exp(-(x_coord**2 + y_coord**2) / (2 * sigma**2))


def apply_filter_and_inverse_fft(filter_matrix, fourier_shifted):
    filtered_fourier = filter_matrix * fourier_shifted
    filtered_image = np.real(scipy.fft.ifft2(scipy.fft.ifftshift(filtered_fourier)))
    return filtered_fourier, filtered_image


def plot_filtered_fourier(filtered_fourier):
    fourier_plot.clear()
    fourier_plot.plot_surface(
        X_coord, Y_coord, np.log(np.abs(filtered_fourier) + 1), cmap="viridis"
    )
    fourier_plot.set_zlim(0, 10)
    fourier_plot.set_zticks([5, 10])
    fourier_plot.set_xticks([])
    fourier_plot.set_yticks([])
    fourier_plot.set_title("H(u,v) * F(u,v)")


def plot_gaussian_filter(filter_matrix, sigma):
    filter_plot.clear()
    filter_plot.plot_surface(X_coord, Y_coord, filter_matrix, cmap="viridis")
    filter_plot.set_zticks([0.5, 1])
    filter_plot.set_xticks([])
    filter_plot.set_yticks([])
    filter_plot.set_title("H(u,v) [" + r"$\sigma$" + f" = {sigma}]")


def update_filter_on_slider_change(sigma_value):
    filter_matrix = create_gaussian_filter(
        X_coord_shifted, Y_coord_shifted, sigma_value
    )
    filtered_fourier, filtered_image = apply_filter_and_inverse_fft(
        filter_matrix, fourier_shifted
    )
    plot_gaussian_filter(filter_matrix, sigma_value)
    plot_filtered_fourier(filtered_fourier)
    image_plot.set_data(filtered_image)


root = Tk()
root.withdraw()
filename = askopenfilename()

image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (128, 128)) / 255.0
assert len(image.shape) == 2, "Image must be grayscale"

fourier_transform = scipy.fft.fft2(image)
fourier_shifted = scipy.fft.fftshift(fourier_transform)

fig = plt.figure(figsize=(13, 4))
image_axis = fig.add_subplot(151)
image_axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
image_axis.axis("off")
image_axis.set_title("Image f(x,y)")

x = np.linspace(0, fourier_shifted.shape[0] - 1, fourier_shifted.shape[0])
y = np.linspace(0, fourier_shifted.shape[1] - 1, fourier_shifted.shape[1])
X_coord, Y_coord = np.meshgrid(x, y)
X_coord_shifted = X_coord - fourier_shifted.shape[0] / 2
Y_coord_shifted = Y_coord - fourier_shifted.shape[1] / 2

fourier_axis = fig.add_subplot(152, projection="3d")
fourier_axis.plot_surface(
    X_coord, Y_coord, np.log(np.abs(fourier_shifted) + 1), cmap="viridis"
)
fourier_axis.set_zlim(0, 10)
fourier_axis.set_zticks([5, 10])
fourier_axis.set_xticks([])
fourier_axis.set_yticks([])
fourier_axis.set_title("F(u,v)")

filter_matrix = create_gaussian_filter(X_coord_shifted, Y_coord_shifted, 20)

filter_plot = fig.add_subplot(153, projection="3d")
filter_plot.plot_surface(X_coord, Y_coord, filter_matrix, cmap="viridis")
plot_gaussian_filter(filter_matrix, 20)

filtered_fourier, filtered_image = apply_filter_and_inverse_fft(
    filter_matrix, fourier_shifted
)
fourier_plot = fig.add_subplot(154, projection="3d")
plot_filtered_fourier(filtered_fourier)

image_plot_axis = fig.add_subplot(155)
image_plot = image_plot_axis.imshow(filtered_image, cmap="gray", vmin=0.0, vmax=1.0)
image_plot_axis.axis("off")
image_plot_axis.set_title("Filtered g(x,y)")

slider_axis = plt.axes([0.25, 0.05, 0.65, 0.03])
sigma_slider = Slider(
    slider_axis, "Gaussian " + r"$\sigma$", 1, 50, valinit=20, valstep=1
)
sigma_slider.on_changed(update_filter_on_slider_change)
plt.show()
