from tkinter import Tk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from img_utils import open_raw_img, adjust_exposure, resize_raw_img

def slider_callback(value):
    img = adjust_exposure(image, value)
    img_figure.set_data(img)

root = Tk()
root.withdraw() 
filename = askopenfilename()
image = open_raw_img(filename)
image = resize_raw_img(image, 0.25)

fig, ax = plt.subplots()
img_figure = ax.imshow(image, cmap='gray', vmin=0.0, vmax=1.0)
ax.axis('off')

axslider = plt.axes([0.25, 0.05, 0.65, 0.03])
sldr = Slider(axslider, 'Digital Exposure', 0, 10, valinit=1)
sldr.on_changed(slider_callback)
plt.show()
