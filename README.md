# Evilsgt-Image-Maps-2
Extension of the Evilsgt-Image-Maps Project
https://github.com/thehamzza/Evilsgt-Image-Maps

## V4 Explanation 
Evilsgt-Image-Maps-2

Vary.py
This code generates a face using a Generative Shape Model (GSM) and allows the user to adjust some of the shape parameters interactively via TrackBars. Here is a breakdown of the code:
1.	The necessary libraries are imported: OpenCV (cv2), NumPy (np), and Pandas (pd).
2.	The island_annotation.csv file is read in using Pandas and the 'image_path' column is dropped.
3.	The data is transposed and converted to a NumPy array.
4.	The covariance matrix of the data is calculated using NumPy's cov() function.
5.	The covariance matrix can come out as complex numbers with very small imaginary values, so these are made real using NumPy's abs() function.
6.	The eigenvalues and eigenvectors of the covariance matrix are calculated using NumPy's eig() function.
7.	The mean face is calculated as the average of all faces in the data.
8.	The mean features are calculated by averaging the features across all faces, and the mean snake is reshaped to a 2D matrix.
9.	A NumPy array is created to hold the generated face.
10.	The initial shape parameters for the generative shape model are set to 0.
11.	A function is defined to annotate an image with the island contour and landmarks.
12.	A function is defined to display the generated image and update it as the user adjusts the shape parameters via the TrackBars.
13.	Six callback functions are defined, one for each of the six shape parameters being adjusted. Each callback function updates the corresponding shape parameter and calls the displayImage() function to update the generated image.
14.	A window is created to display the generated image and TrackBars are added for each of the six shape parameters being adjusted.
15.	The generated face is initially displayed in the window and the user can adjust the shape parameters via the TrackBars.
16.	The program waits for the user to press the spacebar before closing the window.
Island_Annotation.py:
This code is an implementation of active contour segmentation, a technique used to extract objects from images. The code reads images from a folder, applies the active contour algorithm on each image to extract the shape of the object, and saves the annotated image along with the snake (a curve representing the object's boundary) as a CSV file.
1.	The code starts by importing necessary libraries, including scikit-image, numpy, pandas, and matplotlib. It creates an empty pandas dataframe with 800 columns.
2.	It uses the `glob` module to get a list of all `.jpg` files in the `images` folder.
3.	It creates a new folder called `images_annotated` if it does not already exist.
4.	It loops through each image in the `image_paths` list, loads the image using `imread`, and pads it to make it square.
5.	It creates a circle with a radius equal to the smaller side of the image and centers it in the middle of the image.
6.	It uses the active contour algorithm (`active_contour` function from scikit-image) on the Gaussian-filtered image to extract the snake.
7.	It appends the snake coordinates to the pandas dataframe `df`.
8.	It plots the original image along with the extracted snake and saves the annotated image in the `images_annotated` folder.
9.	After looping through all the images, the code adds the `image_path` column to the pandas dataframe and saves it as a CSV file called `island_annotation.csv`.
10.	The code extracts the feature matrix from `df` by dropping the `image_path` column and transposing the data. It then calculates the mean of each feature.
11.	It reshapes the mean features into a snake and saves the snake as an image called `mean_island.jpg`.
12.	Finally, it loads a new image called `fn01.jpg` and puts the mean snake on it. It saves the annotated image as `fn01_annot.jpg`.
Generate_island_rotation.py:
Certainly! Here's an explanation of the code:
1.	The first few lines import various modules from the scikit-image library, as well as the built-in "os" module.
2.	The next two lines read in an image file called "IsleOfMan.jpg" and convert it to grayscale using the "rgb2gray" function from scikit-image's color module.
3.	The following line applies Otsu's thresholding method to the grayscale image, which is a way of automatically determining the best threshold for separating the image into foreground and background pixels.
4.	The next line converts the thresholded image to a binary image, where all foreground pixels are set to 1 and all background pixels are set to 0.
5.	The "if not os.path.exists("images"):" block checks whether a directory called "images" already exists in the current working directory. If not, it creates one using the "os.mkdir" function.
6.	The "for i in range(12):" loop rotates the binary image 12 times, by 10 degrees at a time, using the "rotate" function from scikit-image's transform module. Each rotated image is saved as a JPEG file in the "images" directory using the "imsave" function from the io module. The file names are simply numbered from 1 to 12.
In summary, this code reads in an image, converts it to grayscale, applies Otsu's thresholding method to convert it to a binary image, and then saves 12 rotated versions of the binary image as JPEG files. The purpose of this code is not entirely clear from this snippet alone, but it appears to be a simple image processing exercise that demonstrates some of the capabilities of scikit-image.

