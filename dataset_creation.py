import cv2
import os

# Initialize global variables
current_class = None
output_folder = None
# Dictionary to store output folders for each class
output_folders = {
    1: './processed/1',
    2: './processed/2',
    3: './processed/3'
}  
point = None

def mouse_callback(event, x, y, flags, param):
    global current_class, output_folder, point

    if event == cv2.EVENT_LBUTTONDOWN:
        if current_class is not None:
            # Save subimage
            filename = f'{output_folder}/{current_class}_{len(os.listdir(output_folder))}.jpg'
            subimage = img[max(0, y - half_height):min(y + half_height, img.shape[0]),
                           max(0, x - half_width):min(x + half_width, img.shape[1])]
            cv2.imwrite(filename, subimage)
            print(f'Saved {filename}')

def create_output_folders():
    global output_folders

    for i in range(3):
        folder_name = f'Class_{i}'
        os.makedirs(folder_name, exist_ok=True)
        output_folders[i + 1] = folder_name

# Dictionary to store subimage dimensions for each class
subimage_dimensions = {
    1: (60, 60),
    2: (70, 70),
    3: (80, 80)
}

# Folder path containing the images
folder_path = './imgs'

# Get a list of all image file names in the folder
image_files = [file for file in os.listdir(folder_path) if file.endswith(('.jpg', '.jpeg', '.png'))]

# Create output folders
create_output_folders()

while True:
    class_input = input("Enter the class of cans you want to select (1:circular, 2:oval, or 3:rectangular) [Press 'q' to quit]: ")
    
    if class_input == 'q':
        break
    
    current_class = int(class_input)
    output_folder = output_folders[current_class]
    
    # Get subimage dimensions for the current class
    half_width, half_height = subimage_dimensions[current_class]

    # Iterate through each image file
    for image_file in image_files:
        # Load the image
        image_path = os.path.join(folder_path, image_file)
        img = cv2.imread(image_path)

        # Create a window and bind the mouse callback function
        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', mouse_callback)

        while True:
            cv2.imshow('Image', img)

            # Wait for key press and handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Press 'q' to exit
                break

        cv2.destroyAllWindows()
