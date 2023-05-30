import cv2
import copy
import numpy as np
import csv
import matplotlib.pyplot as plt
import sys

counter = 0
DS_values = []

def detect_leaves(i):
    i = i+1
    empty_str = ''
    if i < 10 :
        empty_str = '0'
    original_image = 'plant_image_dataset/plant0' + empty_str + str(i) +'_rgb.png'
    ground_truth =  'plant_image_dataset/plant0' + empty_str + str(i) +'_label.png'
    img = cv2.imread(original_image)
    mask = cv2.imread(ground_truth, 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)


    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    opening = cv2.erode(thresh, kernel, iterations=1)
    sure_bg = cv2.dilate(opening, kernel, iterations=1)
    gradient = cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, kernel)
    #cv2.imshow('bg', sure_bg)

   
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.436*dist_transform.max(), 255, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)  
    unknown = cv2.subtract(sure_bg, sure_fg)

    # watershed algorithm
    _, markers = cv2.connectedComponents(sure_fg, connectivity=4)
    markers += 1
    markers[unknown == 255] = 0

    input_im = copy.deepcopy(img)
    markers = cv2.watershed(input_im , markers)

    num_leaves = np.amax(markers) - 1
    print('Number of leaves: {}'.format(num_leaves))

    input_im[markers == -1] = [255, 0, 0]
    flag = False

    high_value_colors = [[],[],[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [128, 0, 128], [0, 255, 255], [255, 0, 255], [128, 128, 0],[0, 128, 128], [255, 128, 128], [255, 128, 255]]
    low_value_colors = [[],[],[245, 0, 0], [0, 245, 0], [0, 0, 245], [245, 245, 0], [118, 0, 118], [0, 245, 245], [245, 0, 245], [118, 118, 0],[0, 118, 118], [245, 118, 245]]
    for i in range(2, markers.max() + 1):
        input_im[markers == i] = [randint(0, 255), randint(0, 255), randint(0, 255)] if flag else high_value_colors[i]

    image_seg = copy.deepcopy(input_im)

    image_seg[markers < 2] = [0, 0, 0]         

    for i in range(2, markers.max() + 1):
        color_mask = cv2.inRange(image_seg, np.array(low_value_colors[i]), np.array(high_value_colors[i]))
        cnts, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        con = max(cnts, key=cv2.contourArea)
        max_width = max(con, key=lambda x: x[0][0])[0][0]
        min_width = min(con, key=lambda x: x[0][0])[0][0]
        max_height = max(con, key=lambda x: x[0][1])[0][1]
        min_height = min(con, key=lambda x: x[0][1])[0][1]
        x, y, w, h = cv2.boundingRect(con[0])
        w = max_width - min_width
        h = max_height - min_height
        img = cv2.rectangle(img,(min_width, min_height),(x+w,y+h),(0,255,255),2)

    cv2.waitKey(1)
    global counter
    counter = counter + 1

 
    file_name7 = "E:\\UOL\\Computer Vision\\Assignment\\output\\" + "Bounding_box" + "_" + str(counter)+".png"
    cv2.imwrite(file_name7, img)

    intersection = cv2.bitwise_and(mask, sure_bg)

    # Dice Similarity Score
    ds = (2.0 * cv2.countNonZero(intersection)) / (cv2.countNonZero(mask) + cv2.countNonZero(sure_bg))
    DS_values.append(ds)
    print('Dice Similarity Score: {:.2f}'.format(ds))
    return ds, num_leaves
    
def performance_metrics():
    ds_mean = 0
    total_leaves = 0
    actual_leaves_count = 0
    with open('leaf_counts.csv', 'r') as f:
        reader = csv.reader(f)
        leaf_counts = [int(row[1]) for row in reader]
    
    abs_diff = []
    count = 0
    for i in range(0, 16):
        print(i+1)
        ds, num_leaves = detect_leaves(i)
        ds_mean = ds_mean + ds
        total_leaves = total_leaves + num_leaves
        diff = abs(num_leaves - leaf_counts[i])
        actual_leaves_count = actual_leaves_count + leaf_counts[i]
        print("Num of leaves", num_leaves)
        print("ABS", diff)
        print("Leaf_counts", leaf_counts[i])
        #print("count",i)
        abs_diff.append(diff)

    mean_diff = sum(abs_diff) / len(abs_diff)
    print("Accuracy", total_leaves / actual_leaves_count)
    print("Mean Difference", mean_diff)
    #cv2.imshow('Segmented plant', result)
    print("count",i)
    print("Mean of Dice Similarity {}".format(ds_mean/16))
    print("Total Leaves", total_leaves)
    print("Actual Leaves", actual_leaves_count)

def plot_bar():

    image_numbers = list(range(1, len(DS_values) + 1))

    plt.bar(image_numbers, DS_values)
    plt.xlabel('Images')
    plt.ylabel('DS Score')
    plt.title('DS Score for Each Image')

    plt.show()

def main():
    try:
        performance_metrics()
        plot_bar()
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")
        cv2.destroyAllWindows()
        sys.exit(0)
if __name__ == '__main__':
    main()
