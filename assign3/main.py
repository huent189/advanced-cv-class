from tkinter import Tk
from tkinter.filedialog import askopenfilename
import cv2
import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def create_gaussian_filter(x, y, sigma):
    return np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) 

def apply_filter_and_ifft(H, fourier_shifted):
    G = H * fourier_shifted
    g = np.real(scipy.fft.ifft2(scipy.fft.ifftshift(G)))
    return G, g
def plot_filtered_fourier(G):
    ax_HF.clear()
    ax_HF.plot_surface(X_coord, Y_coord, np.log(np.abs(G)).clip(0, None), cmap='viridis')
    ax_HF.set_zlim(0, 10)
    ax_HF.set_zticks([5, 10])
    ax_HF.set_xticks([])
    ax_HF.set_yticks([])
    ax_HF.set_title("H(u,v) * F(u,v)")
def plot_gaussian_filter(H, sigma):
    ax_H.clear()
    ax_H.plot_surface(X_coord, Y_coord, H, cmap='viridis')
    ax_H.set_zticks([0.5, 1])
    ax_H.set_xticks([])
    ax_H.set_yticks([])
    ax_H.set_title("H(u,v) [" + r'$\sigma$' + f" = {sigma}]")
def slider_callback(value):
    H = create_gaussian_filter(X_coord_shifted, Y_coord_shifted, value)
    G, g = apply_filter_and_ifft(H, fourier_shifted)
    plot_gaussian_filter(H, value)
    plot_filtered_fourier(G)
    g_fig.set_data(g)

root = Tk()
root.withdraw()
filename = askopenfilename()

image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (128, 128)) / 255.0
assert len(image.shape) == 2, "Image must be grayscale"

fourier = scipy.fft.fft2(image)
fourier_shifted = scipy.fft.fftshift(fourier)

fig = plt.figure()
ax = fig.add_subplot(151)
ax.imshow(image, cmap='gray', vmin=0.0, vmax=1.0)
ax.axis('off')
ax.set_title('Image f(x,y)')

x = np.linspace(0, fourier_shifted.shape[0]-1, fourier_shifted.shape[0])
y = np.linspace(0, fourier_shifted.shape[1]-1, fourier_shifted.shape[1])
X_coord, Y_coord = np.meshgrid(x, y)
X_coord_shifted = X_coord - fourier_shifted.shape[0] / 2
Y_coord_shifted = Y_coord - fourier_shifted.shape[1] / 2

ax = fig.add_subplot(152, projection='3d')
ax.plot_surface(X_coord, Y_coord, np.log(np.abs(fourier_shifted)), cmap='viridis')
ax.set_zlim(0, 10)
ax.set_zticks([5, 10])
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('F(u,v)')

H = create_gaussian_filter(X_coord_shifted, Y_coord_shifted, 20)

ax_H = fig.add_subplot(153, projection='3d')
ax_H.plot_surface(X_coord, Y_coord, H, cmap='viridis')
plot_gaussian_filter(H, 20)

G, g = apply_filter_and_ifft(H, fourier_shifted)
ax_HF = fig.add_subplot(154, projection='3d')
plot_filtered_fourier(G)

ax_g = fig.add_subplot(155)
g_fig = ax_g.imshow(g, cmap='gray', vmin=0.0, vmax=1.0)
ax_g.axis('off')
ax_g.set_title('Filtered g(x,y)')

axslider = plt.axes([0.25, 0.05, 0.65, 0.03])
sldr = Slider(axslider, 'Gaussian ' + r'$\sigma$',1, 50, valinit=20, valstep=1)
sldr.on_changed(slider_callback)
plt.show()