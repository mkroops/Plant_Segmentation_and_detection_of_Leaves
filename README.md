# Plant_Segmentation_and_detection_of_Leaves

A. Literature Review 
Detection of leaves from plant is challenging task in com-
puter vision and image processing. There are several approches
has been proposed to solve this task such as deep learning
methods and traditional segmentation methods.
Traditional Segmentation algorithms which uses some math-
ematical operations to segment the objects. Some Popular
algorithms are watershed, edge detection, thresholding, region
growing, histograms etc. These segmentations are used in
blood cell counting, segmentation of cotton leaves, segmenting
satellite image etc. Although they might not always be as
accurate as deep learning based approaches Faster U-Net, R-
CNN or YOLO etc but these techniques are still relevant
and helpful in many applications, especially when there is a
shortage of labelled data or computational power. Based on
application and computational power one can choose between
deep Learning and Traditional segmentation to detect the
leaves from plant.
B. Methodology
Methodology used to detect leaves from plant are Gaus-
sianBlur, Thresholding, Erosion, Dilation, Distance Transform,
Connected Components, Watershed algorithm, Find Contours,
Dice Similarity Score, Absolute Mean Difference.
C. Chosen Image Dataset
To apply above methodologies Plant 2 and Plant 5 images
are chosen to segment the leaves in plant as shown in Fig 1.
D. Grayscaling
Grayscaling is process which involves changing an image
from another colour space such as RGB, CMYK, HSV, etc
(a) Plant 2 (b) Plant 5
Fig. 1: Original Image
to a variety of grayscales. Grayscaling are single dimension
whereas RGB are three dimensions . Grayscaling has been
used initially because it is faster to load and easier to store
and it is also reveal more detail in image [6]. cv2.COLOR
BGR2GRAY has been used to convert RGB image to gray
image.
E. Gaussian Blur
The most popular smoothing method for removing noise
from images and videos is Gaussian blur. To create the
smoothed image in this method, an image must be convolved
with a Gaussian kernel. The kernel size can be adjusted to
meet your requirements. To ensure that the edges of the kernel
are close to zero, the standard deviation of the Gaussian
distribution in the X and Y directions should be carefully
determined. The 3 x 3 and 5 x 5 Gaussian kernels are shown
in Fig 2 of Plant 5 . To specify the area around each pixel
we must select the proper kernel size. If it is too big the image
may lose some of its fine details and appear blurry. You cannot
remove image noise if it is too little.
(a) 3*3 Kernel (b) 5*5 Kernel
Fig. 2: Gaussian Blur
F. Thresholding
Thresholding is the technique used for separating fore-
ground pixels from background pixels. Such a thresholding
technique typically requires a threshold and a grayscale image
as inputs which gives result as a binary image. Otsu
and adaptive threshhoding techniques are used to compare
as shown in Fig 3. Otsu method is variance based technique
where the threshold value where the weighted variance be-
tween the foreground and background pixels is the least is
found whereas adaptive thresholding that calculates a threshold
value for each pixel based on the local intensity of the image
. For this Image dataset Otsu method has adapted.
(a) OTSU (b) Adaptive
Fig. 3: Thresholding
G. Erosion
Erosion is a morphological process that reduces the size
of the foreground object borders and eliminates small noise
patches. 3x3 rectangular kernel has been used to control the
degree of erosion. Iteration indicates how many times
the input image should undergo the erosion operation. The
edges of the leaves in the image will have one layer of pixels
removed with each iteration. One iteration has been used
as shown in Fig 4.
H. Dilation
Dilation is a morphological process that enlarges the fore-
ground objects boundaries and fills in small gaps [9]. The
segments of the plant are divided into regions and their borders
are smoothed out through this operation to fill in any gaps. The
size of stucturing element which determines size of dilation.
Iteration defines number of dilations to be performed. More
pixels will be added to the objects bounds as a result of
a greater dilation process as iteration is increased [9]. One
iteration has been used as shown in fig 4.
(a) Erosion (b) Dilation
Fig. 4: Erosion and Dilation
I. Distance Transform
The inputs for the distance transform operator are typically
binary pictures. The foreground points grey level intensities
are adjusted in this operation to reflect how far away each one
is from the nearest 0 value which is border. cv2.DIST
L2 parameter has been used which stands for the Euclidean
distance. cv2.DIST L1 for Manhattan distance and cv2.DIST C
for Chebyshev distance are two more distance types available
in cv2.distanceTransform function. 5*5 kernel has been
used. 0.45 and 0.22 distance threshold has been used to in
Plant 5 image. In Fig 5, Two leaves has been vanished in 0.45
threshold and for 0.22 threshold in right side two leaves is not
separated and it is considered as single leaves. Based on Image
dataset we have to apply distance threshold accordingly.
(a) 0.45 Threshold distance (b) 0.22 Threshold distance
Fig. 5: Distance Transform
J. Connected Components
Connected Component labelling is used to determine the
connectedness of ”blob” like regions in a binary image [11].
After using connected-component labelling to extract the blob,
we may still use contour characteristics to quantify the area.
The connected components of a binary thresholded plate image
can be calculated and the blobs can then be filtered depending
on characteristics like width, height, area, solidity, etc. In
Connected Component, Connectivity 4 has been applied which
is used to identify the relationships between different parts of
a network.
K. Watershed Algorithm
Watershed algorithm is an image segmentation method
that treats an image as a topographic surface and splits it
into segments based on its intensity or colour. Local
minima are used as markers to build the first segments and the
picture is flooded from these markers to fill in the valleys and
produce the final segments. This algorithm can use a priority
queue to keep track of the pixels and assigns each pixel to
the nearest marker. However, It will oversegment images
but based on marker based approach, it will fill known and
marked areas of objects with different level of water. Based on
distance transformation we will receive foreground and then by
subracting foreground and background, an unknown region is
created. By using Connected components, blobs are detected to
find markers which is filled with watershed algorithm. In
Fig 6 for Plant 5, It is noticed that 6 leaves are segmented with
distance threshold of 0.45 where with 0.22 distance threshold
Small leaves discovered at centre but in right side of image two
leaves are segmented into one. Similarly, In Fig 7 for Plant 2,
5 leaves are detected in 0.45 distance threshold but with 0.22
ditance threshold only 4 leaves are detected. Based on different
Image dataset, Threshhold can be applied accordingly.
(a) 0.45 Threshold distance (b) 0.22 Threshold distance
Fig. 6: Image Segmentation For Plant 2
(a) 0.45 Threshold distance (b) 0.22 Threshold distance
Fig. 7: Image Segmentation For Plant 5
L. Find Contour
The displayed shapes of the objects that are present in
the Figures that have been processed through the system
correspond to an abstract collection of segments and points
called a contour . A contour is a curve that connects all
of the continuous points along an object’s edge that are the
same intensity or colour [12]. Contour was used to find
with the largest area So it determines the width and height
of a bounding rectangle around the white region and then
draw a Yellow rectangle around the bounding box using the
cv2.rectangle function.
(a) Plant 2 (b) Plant 5
Fig. 8: Bounding Box - Object Detection
M. Dice Similarity Score
Dice Similarity Score is used to evaluate the similarity
between segmented image and labelled image. The score
ranges from 0 to 1, Where 1 indicates maximum overlap
between segmented image and labelled image and 0 indicates
no overlap.
The dice Score can be calculated as
DiceScore = 2 × |X ∩ Y |
|X| + |Y | (1)
Where X and Y are segmented and labelled images, |X|
represents the number of elements in each set, |Y | represents
the number of elements in each set and |X ∩ Y | represents the
intersection of the segmented image and labelled image .
Dice Similarity for Plant 2 and Plant 5 shown in Table 1.
TABLE I: Dice Similarity, Predicted and Actual Leaves for
Plants 2 and 5
Plant Dice Similarity Num of Predicted Leaves Actual Leaves
2 0.96 5 7
5 0.94 6 7
Dice Similarity Mean for Plant 2 and Plant 5 is 0.94652.
Dice Similarity score for each image is shown in Fig 9. All
images achieved dice Similarity Score above 0.9 which is near
to 1 indicates maximum overlap between segmented image and
labelled image
Fig. 9: Bar Graph For Dice Similarity Score For 16 Images
Over Dice Similarity Mean for 16 images is 0.9408
N. Absolute Mean Difference
The absolute mean difference is a metric for comparing two
sets of data and is frequently applied when comparing two
images. The average absolute difference between the corre-
sponding images of the two sets is used to calculate it.
Absolute Mean Difference is formulated as
Absolute Mean Difference = 1
N
N∑
i=1
|xi − yi| (2)
Where xi and yi are predicted leaves and actual leaves of
ith elements respectively. Absolute mean difference with lower
value indicates there is greater similarity between two sets of
values.
Plant AMD Num of Predicted Leaves Actual Leaves
Plant 1 2 5 7
Plant 2 2 5 7
Plant 3 3 5 8
Plant 4 3 5 8
Plant 5 2 6 8
Plant 6 1 6 7
Plant 7 3 5 8
Plant 8 0 6 6
Plant 9 2 4 6
Plant 10 2 9 7
Plant 11 1 7 8
Plant 12 2 7 7
Plant 13 4 4 8
Plant 14 2 6 8
Plant 15 1 7 6
Plant 16 0 7 7
TABLE II: Absolute Mean Difference for different plants
AMD stands for Absolute Mean Difference. Overall Ab-
solute mean difference for all 16 Images is 1.875 which
is moderately performing well and Overall Predicted leaves
count is 94 out of 114 actual leaves
