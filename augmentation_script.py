import imgaug as ia
import imgaug.augmenters as iaa
import imgaug.parameters as iap
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from PIL import Image
import numpy as np
import os
import imageio
import argparse

# get input folders (images, annotations) and output folders (images, annotations)
parser = argparse.ArgumentParser()
parser.add_argument("images_folder", help="Folder of images for augmentation")
parser.add_argument("labels_folder", help="Folder of annotations for augmentation")
parser.add_argument("aug_images_folder", help="Target folder of images after augmentation")
parser.add_argument("aug_labels_folder", help="Target folder of annotations after augmentation")
parser.add_argument("--train_image_number", type=int, help="Number of training examples")
parser.add_argument("--image_size", type=int, help="Size of augmented images (squarred)")
parser.add_argument("--iteration", type=int, help="Number of time, we iterate over the images")


args = parser.parse_args()

TRAINING_EXAMPLES = 100
IMAGE_SIZE = 640
ITERATION_NUMBER = 3

if args.train_image_number:
    TRAINING_EXAMPLES = int(args.train_image_number)

if args.image_size:
    IMAGE_SIZE = int(args.image_size)

if args.iteration:
    ITERATION_NUMBER = int(args.iteration)





IMAGES_FOLDER = "C:/Users/CharlEm OLE/Downloads/TPT-BIHAR-OLE/TPT2_ZAPATEKO/SHOES_ID/6"
ANNOTS_FOLDER = "C:/Users/CharlEm OLE/Downloads/TPT-BIHAR-OLE/TPT2_ZAPATEKO/SHOES_ID/6-annoté/renom"
AUGMENTED_IMAGES_FOLDER = "C:/Users/CharlEm OLE/Downloads/TPT-BIHAR-OLE/TPT2_ZAPATEKO/SHOES_ID/6_new"
AUGMENTED_ANNOTS_FOLDER = "C:/Users/CharlEm OLE/Downloads/TPT-BIHAR-OLE/TPT2_ZAPATEKO/SHOES_ID/Anot-6-modif"

IMAGES_FOLDER = args.images_folder
ANNOTS_FOLDER = args.labels_folder
AUGMENTED_IMAGES_FOLDER = args.aug_images_folder
AUGMENTED_ANNOTS_FOLDER = args.aug_labels_folder

""" 
IMAGES_FOLDER = "/home/nathan/id_shoes_pictures/TBS 11"
ANNOTS_FOLDER = "/home/nathan/id_shoes_pictures/TBS 11/lbl"
AUGMENTED_IMAGES_FOLDER = "/home/nathan/id_shoes_pictures/TBS 11 aug"
AUGMENTED_ANNOTS_FOLDER = "/home/nathan/id_shoes_pictures/TBS 11 aug/lbl" 
"""


# verify if counter.txt exists, if not we create it
if os.path.isfile(AUGMENTED_IMAGES_FOLDER+"/counter.txt"):
    # read numbers of augmented images already saved
    count_file_read = open(AUGMENTED_IMAGES_FOLDER+"/counter.txt", 'r')
    COUNT = int(count_file_read.read().split()[0])
    count_file_read.close()
else:
    count_file_read = open(AUGMENTED_IMAGES_FOLDER+"/counter.txt", 'w')
    count_file_read.close()
    COUNT = 0

# create train and val folders if not exists
for parent_dir in [AUGMENTED_IMAGES_FOLDER, AUGMENTED_ANNOTS_FOLDER]:
    for child_dir in ["train", "val"]:
        path = os.path.join(parent_dir, child_dir)
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as error:
            continue
    


ia.seed(1)

def from_yolo_to_real_coordinates(box, image):
    """ Convert (x,y,w,h) from yolo to real coordinates (xmin,ymin,xmax,ymax) """
    img_h, img_w, _ = image.shape
    x1, y1 = int((box[0] + box[2]/2)*img_w), int((box[1] + box[3]/2)*img_h)
    x2, y2 = int((box[0] - box[2]/2)*img_w), int((box[1] - box[3]/2)*img_h)
    return x1, y1, x2, y2
    
def from_real_coordinates_to_yolo(box, image):
    """ Convert (xmin,ymin,xmax,ymax) from real coordinates to yolo (x,y,w,h) """
    img_h, img_w, _ = image.shape
    xmin, ymin, xmax, ymax = box
    xcen = float((xmin + xmax)) / 2 / img_w
    ycen = float((ymin + ymax)) / 2 / img_h
    w = float((xmax - xmin)) / img_w
    h = float((ymax - ymin)) / img_h
    return xcen, ycen, w, h


if __name__=="__main__":

    # get filenames of images
    list_files = os.listdir(IMAGES_FOLDER)
    list_files = [filename for filename in list_files if filename[-4:] in [".jpg", ".png"]]

    # open images with pillow
    # pil_images = [Image.open(IMAGES_FOLDER+"/"+fil) for fil in list_files]

    # get filepaths of image's annotations
    annotation_images = [ANNOTS_FOLDER+"/"+filename[:-4]+".txt" for filename in list_files]

    # convert images to numpy ndarray
    numpy_images = [np.asarray(Image.open(IMAGES_FOLDER+"/"+fil).resize((IMAGE_SIZE,IMAGE_SIZE)), dtype=np.uint8) 
                        for fil in list_files]

    list_boxes_denormalized = []
    coeff = ""

    # save bounding boxes of all the images in a list
    for i, filepath in enumerate(annotation_images):
        with open(filepath, 'r') as file:
            tab = file.read()
            coeff = tab.split()[0]
            coef = tab.split()[1:]
            boxes_normalized = list(map(lambda x: float(x), coef))
            boxes_denormalized = list(from_yolo_to_real_coordinates(boxes_normalized, numpy_images[i]))
            boxes_denormalized.append(coeff)
            list_boxes_denormalized.append(boxes_denormalized)

    # create custom object with all the bounding boxes
    bbs = [BoundingBoxesOnImage([BoundingBox(x1=bounding_boxes[0],y1=bounding_boxes[1],x2=bounding_boxes[2],y2=bounding_boxes[3], label=bounding_boxes[4])], 
            shape=numpy_images[i].shape)
            for i, bounding_boxes in enumerate(list_boxes_denormalized)]


    # create a sequence to apply transformations on images
    seq = iaa.Sequential([
            iaa.Fliplr(0.5),  # horizontal flips
            iaa.Sometimes(
                0.5,
                iaa.GaussianBlur(sigma=iap.Uniform(0, 0.5))
            ),  # But we only blur about 50% of all images.
            iaa.LinearContrast((0.75, 1.5)),  # Strengthen or weaken the contrast in each image.
            iaa.Multiply((0.75, 1.2)),],  # change luminosity

            random_order=True)  # apply augmenters in random order

    # set changes on image randomly and compute augmented images with corresponding labels
    seq_det = seq.to_deterministic()
    images_aug = seq_det.augment_images(numpy_images * ITERATION_NUMBER)
    bbs_aug = seq_det.augment_bounding_boxes(bbs * ITERATION_NUMBER)

    list_new_names = []

    # save augmented images 
    for i, image in enumerate(images_aug):
        new_name = str(COUNT)
        list_new_names.append(new_name)

        if i<TRAINING_EXAMPLES:
            imageio.imwrite(AUGMENTED_IMAGES_FOLDER+"/train/"+coeff+"_"+new_name+".jpg", image)

        else:
            imageio.imwrite(AUGMENTED_IMAGES_FOLDER+"/val/"+coeff+"_"+new_name+".jpg", image)

        print("Augmented image {} saved".format(i+1))
        COUNT += 1

    # save corresponding annotations
    for i, bounding_boxes_on_images in enumerate(bbs_aug):

        bounding_boxes = bounding_boxes_on_images.bounding_boxes[0]
        x1, y1, x2, y2 = from_real_coordinates_to_yolo([bounding_boxes.x1, bounding_boxes.y1, bounding_boxes.x2, bounding_boxes.y2], images_aug[i])
        label = bounding_boxes.label

        if i<TRAINING_EXAMPLES:
            fichier = open(AUGMENTED_ANNOTS_FOLDER+"/train/"+coeff+"_"+list_new_names[i]+".txt", "a")

        else:
            fichier = open(AUGMENTED_ANNOTS_FOLDER+"/val/"+coeff+"_"+list_new_names[i]+".txt", "a")

        fichier.write(str(label))
        fichier.write(" ")
        fichier.write(str(x1))
        fichier.write(" ")
        fichier.write(str(y1))
        fichier.write(" ")
        fichier.write(str(x2))
        fichier.write(" ")
        fichier.write(str(y2))
        fichier.close()
        print("Augmented label {} saved".format(i))

    # save the new number of augmented images
    count_file_saved = open(AUGMENTED_IMAGES_FOLDER+"/counter.txt", 'w')
    count_file_saved.write(str(COUNT))
    count_file_saved.close()