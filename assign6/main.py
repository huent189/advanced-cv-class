import numpy as np
import matplotlib.pyplot as plt
import cv2
current_frame = -1
sharpness_scores = None
def load_focus_stacks(filename):
    """
    Load the focus stacks from a given .npy file.

    :param filename: Path to the .npy file containing the focus stacks.
    :return: 4D numpy array containing the focus stacks.
    """
    return np.load(filename)

def compute_sobel_energy(focus_stacks):
    """
    Compute the sharpness of each frame in the focus stack using the Sobel filter.

    :param focus_stacks: 4D numpy array containing the focus stacks.
    :return: 1D numpy array containing the sharpness value for each frame.
    """
    sobel_energies = []

    for frame in focus_stacks:
        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
        
        # Apply Sobel filter
        sobel_x = cv2.Sobel(gray_frame, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_frame, cv2.CV_32F, 0, 1, ksize=3)
        # Calculate the gradient magnitude
        sobel_energy = np.sqrt(sobel_x**2 + sobel_y**2)
        
        sobel_energies.append(sobel_energy)

    return np.array(sobel_energies)

def plot_sharpness(sharpness_scores, ax):
    """
    Plot the sharpness scores for each frame in the focus stack.

    :param sharpness_scores: 1D numpy array containing the sharpness scores for each frame.
    """
    ax.clear()
    max_shaprness_index = np.argmax(sharpness_scores)
    ax.plot(sharpness_scores, marker='o')
    ax.plot(current_frame, sharpness_scores[current_frame], 'ro')
    ax.axvline(x=max_shaprness_index, color='r', linestyle='--', label='Max Sobel Response Index')
    ax.set_xlabel('Image Index')
    ax.set_ylabel('Sobel Response')
    ax.set_title('Sobel Response vs Image Index')
    ax.legend()

def display_image(focus_stacks, index, ax):
    ax.imshow(focus_stacks[index], vmin=0, vmax=1)
    ax1.add_patch(plt.Rectangle((0,0), 100, 100, linewidth=1, edgecolor='white', facecolor='white'))
    ax.set_title(f'Image {index + 1} of {focus_stacks.shape[0]}')
    ax.axis('off')
    ax.set_aspect('equal')

def adjust_frame(event, focus_stacks, ax1, ax2):
    global current_frame
    if event.key == 'right':
        current_frame = min(current_frame + 1, len(focus_stacks) - 1)
    elif event.key == 'left':
        current_frame = max(current_frame - 1, 0)
    display_image(focus_stacks, current_frame, ax1)
    plot_sharpness(sharpness_scores, ax2)
    plt.draw()

def on_click(event, focus_stacks, sobel_energies, ax1, ax2):
    global current_frame, sharpness_scores
    if event.xdata is not None and event.ydata is not None:
        # Calculate coordinates for the 100x100 window
        x, y = int(event.xdata), int(event.ydata)
        if x > 100 and y > 100:
            window = sobel_energies[:, y-50:y+50, x-50:x+50]

            # Compute sharpness within the window
            sharpness_scores = np.mean(window, axis=(1, 2))
            ax1.clear()
            ax1.add_patch(plt.Rectangle((x-50, y-50), 100, 100, linewidth=1, edgecolor='r', facecolor='none'))
        else:
            sharpness_scores = np.mean(sobel_energies, axis=(1, 2))
            ax1.clear()
        # Find the frame with maximum sharpness
        max_sharpness_index = np.argmax(sharpness_scores)
        
        # Display the box around the local region
        step = 1 if max_sharpness_index > current_frame else -1
        while True:
            display_image(focus_stacks, current_frame, ax1)
            plot_sharpness(sharpness_scores, ax2)
            plt.draw()
            if current_frame == max_sharpness_index:
                break
            current_frame += step
            plt.pause(0.1)
            

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print('Usage: python main.py <filename>')
        sys.exit(1)
    focus_stacks = load_focus_stacks(sys.argv[1])
    sobel_energies = compute_sobel_energy(focus_stacks)
    sharpness_scores = np.mean(sobel_energies, axis=(1, 2))
    current_frame = np.argmax(sharpness_scores)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plot_sharpness(sharpness_scores, ax2)
    display_image(focus_stacks, current_frame, ax1)
    cid_key = fig.canvas.mpl_connect('key_press_event', lambda event: adjust_frame(event, focus_stacks, ax1, ax2))
    cid_click = fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, focus_stacks, sobel_energies, ax1, ax2))
    plt.show()

