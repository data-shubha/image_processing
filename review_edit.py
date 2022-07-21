import numpy as np
# import matplotlib.pyplot as plt
import time
# import matplotlib.pyplot as plt
import os
from skimage import io
# from skimage.filters.rank import entropy
# from skimage.morphology import disk
from skimage.morphology import closing
from skimage.color import rgb2gray
# import matplotlib as plt
import cv2
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Folder where annotated images go
annotated_file_name = "Annotated Cracks"
# Folder where image highlighting road markings are saved
road_marking_folder = "Road Markings"
# Location of the video processed
video_file_name = "D:\\2022 Summer Startup Work\\Pavment Distress One File\\Video\\testVid.mov"
# Location of images processed
test_image_folder_name = "Test Images"
# Where grey scaled and closed images are saved
save_image_folder_name = "Processed Images"
# Where black/white images are saved
final_image_folder_name = "Cracks"


def get_image_list():
    """Get all images in the folder of tested images"""
    img_list = os.listdir(test_image_folder_name)
    for i in range(len(img_list)):
        img_list[i] = test_image_folder_name + '\\' + img_list[i]
    return img_list


# List of images to process

imagesToAverage = get_image_list()


def find_next_right(image, pixel, maxI, maxJ):
    """Find the next black pixel to the right of PIXEL in IMAGE with height MAXI and length MAXJ"""
    if pixel[1] >= maxJ - 1:
        return "End image"
    elif image[pixel[0]][pixel[1] + 1] == 0:
        return [pixel[0], pixel[1] + 1]
    elif pixel[0] == 0 or pixel[0] >= maxI - 1:
        return "End image"
    elif image[pixel[0] - 1][pixel[1] + 1] == 0:
        return [pixel[0] - 1, pixel[1] + 1]
    elif image[pixel[0] + 1][pixel[1] + 1] == 0:
        return [pixel[0] + 1, pixel[1] + 1]
    elif image[pixel[0] + 1][pixel[1]] == 0:
        return [pixel[0] + 1, pixel[1]]
    else:
        return "End of crack"


def findNextDown(image, pixel, maxI, maxJ):
    """Find the next black pixel below PIXEL in IMAGE with height MAXI and length MAXJ"""
    if pixel[0] >= maxI - 1:
        return "End image"
    elif image[pixel[0] + 1][pixel[1]] == 0:
        return [pixel[0] + 1, pixel[1]]
    elif pixel[1] >= maxJ - 1 or pixel[1] == 0:
        return "End image"
    elif image[pixel[0] + 1][pixel[1] + 1] == 0:
        return [pixel[0] + 1, pixel[1] + 1]
    elif image[pixel[0] + 1][pixel[1] - 1] == 0:
        return [pixel[0] + 1, pixel[1] - 1]
    elif image[pixel[0]][pixel[1] + 1] == 0:
        return [pixel[0], pixel[1] + 1]
    else:
        return "End of crack"


def maxFinder(image, jumpSize, output, function, name="crack"):
    """Find the longest output of FUNCTION checking every JUMP SIZE pixel in IMAGE, saving with name NAME if OUTPUT"""
    maxI = len(image)
    maxJ = len(image[0])
    maxLen = 0
    maxList = []
    for i in range(0, maxI, jumpSize):
        for j in range(0, maxJ, jumpSize):
            print([i, j])
            if [i, j] in maxList:
                continue
            else:
                if image[i][j] == 1 and [i, j] not in maxList:
                    pixel = [i, j]
                    traversed = [[i, j]]
                    count = 0
                    next = function(image, pixel, maxI, maxJ)
                    while type(next) != str:
                        count += 1
                        traversed.append(next)
                        next = function(image, next, maxI, maxJ)
                    if count > maxLen:
                        maxLen = count
                        maxList = traversed
    if output:
        annotated_image = []
        for i in range(maxI):
            for j in range(maxJ):
                if image[i][j] == 0:
                    annotated_image.append([0, 0, 0])
                else:
                    annotated_image.append([255, 255, 255])
        annotated_image = np.reshape(annotated_image, (maxI, maxJ, 3))
        print(len(annotated_image), len(annotated_image[0]), maxList)
        for pixel in maxList:
            annotated_image[pixel[0]][pixel[1]] = [255, 0, 0]
        io.imsave(annotated_file_name + "\\" + name + '.jpg', annotated_image)
        # print(annotatedImage)
    return maxLen


def vertical_crack_check(image, trim_parameters=[0, 0], threshold=0.5, save=False, name="crack"):
    """Find a vertical crack in IMAGE after being trimmed by TRIM PARAMETERS; if more than THRESHOLD
    pixels in a column are black, identify a crack and save under name NAME if SAVE"""
    conformity_to_vertical_crack = []
    cracks = []
    trimmed = image[:]
    trimmed = trimmed[:, trim_parameters[1]:len(trimmed[0]) - trim_parameters[1]]
    trimmed = trimmed[trim_parameters[0]:len(trimmed) - trim_parameters[0]]
    maxI = len(trimmed)
    maxJ = len(trimmed[0])

    print(len(image), len(image[0]), len(trimmed))
    for i in range(len(trimmed[0])):
        conformity_to_vertical_crack.append(sum(trimmed[:, i]) / len(trimmed[:, i]))
        if conformity_to_vertical_crack[i] < threshold:
            cracks.append(i)
    print(cracks)
    if save:
        annotated_image = []
        for i in range(maxI):
            for j in range(maxJ):
                if trimmed[i][j] == 0:
                    annotated_image.append([0, 0, 0])
                else:
                    annotated_image.append([255, 255, 255])
        annotated_image = np.reshape(annotated_image, (maxI, maxJ, 3))
        for i in range(maxI):
            for j in range(maxJ):
                if j in cracks:
                    annotated_image[i][j] = [255, 0, 0]
        io.imsave(annotated_file_name + "\\" + name + ".jpg", annotated_image)
    return [conformity_to_vertical_crack, cracks]


def horizontal_crack_check(image, trim_parameters=[0, 0], threshold=0.5, save=False, name="crack"):
    """Find a horizontal crack in IMAGE after being trimmed by TRIM PARAMETERS;
    if more than THRESHOLD pixels in a row are black, identify a crack and save under nameNAME if SAVE"""
    conformity_to_horizontal_crack = []
    cracks = []
    trimmed = image[:]
    trimmed = trimmed[:, trim_parameters[1]:len(trimmed[0]) - trim_parameters[1]]
    trimmed = trimmed[trim_parameters[0]:len(trimmed) - trim_parameters[0]]
    maxI = len(trimmed)
    maxJ = len(trimmed[0])

    print(len(image), len(image[0]), len(trimmed))
    for i in range(len(trimmed)):
        conformity_to_horizontal_crack.append(sum(trimmed[i, :]) / len(trimmed[i, :]))
        if conformity_to_horizontal_crack[i] < threshold:
            cracks.append(i)
    print(cracks)
    if save:
        annotated_image = []
        for i in range(maxI):
            for j in range(maxJ):
                if trimmed[i][j] == 0:
                    annotated_image.append([0, 0, 0])
                else:
                    annotated_image.append([255, 255, 255])
        annotated_image = np.reshape(annotated_image, (maxI, maxJ, 3))
        for i in range(maxI):
            for j in range(maxJ):
                if i in cracks:
                    annotated_image[i][j] = [0, 255, 0]
        io.imsave(annotated_file_name + "\\" + name + ".jpg", annotated_image)
    return [conformity_to_horizontal_crack, cracks]


def get_video_frames(folder_name):
    """Get the frames in the FOLDER NAME folder"""
    img_list = os.listdir(folder_name)
    for i in range(len(img_list)):
        img_list[i] = folder_name + '\\' + img_list[i]
    return img_list


def get_adjacent(i, j, maxI, maxJ):
    """Get all pixels adjacent to the pixel at I, J in an image of size MAXI by MAXJ"""
    top_left = [i - 1, j - 1]
    left = [i, j - 1]
    bottom_left = [i + 1, j - 1]
    bottom = [i + 1, j]
    top = [i - 1, j]
    top_right = [i - 1, j + 1]
    right = [i, j + 1]
    bottom_right = [i + 1, j + 1]
    adjacent = [top_left, left, bottom_left, top, bottom, top_right, right, bottom_right]
    for i in adjacent:
        for j in range(len(i)):
            if i[j] < 0:
                i[j] = 0
            if j == 0 and i[j] >= maxI:
                i[j] = maxI - 1
            if j == 1 and i[j] >= maxJ:
                i[j] = maxJ - 1

    return adjacent


def empty_image_copy(image):
    """Return an empty image with the same size as IMAGE"""
    return np.reshape(np.zeros(len(image) * len(image[0]) * 3), (len(image), len(image[0]), 3))


def process_image(image_name, save):
    """Greyscale and close image at IMAGE NAME; save if SAVE"""
    image = io.imread(image_name)
    # print(image)
    gray_image = rgb2gray(image)
    closed_image = closing(gray_image)
    if save:
        io.imsave(save_image_folder_name + "\\" + image_name[len(test_image_folder_name):], closed_image)
    return closed_image


def cracks_in_image(image_name, threshold, save, closes=1):
    """Find cracks in image at IMAGE NAME, saving if SAVE and closing CLOSES times;
    pixel set to white/black if over/under THRESHOLD"""
    processed_image = process_image(image_name, True)
    for i in range(len(processed_image)):
        for j in range(len(processed_image[0])):
            if processed_image[i][j] < threshold:
                processed_image[i][j] = 0
            else:
                processed_image[i][j] = 1
    for i in range(closes):
        processed_image = closing(processed_image)
    if save:
        io.imsave(final_image_folder_name + "\\" + image_name[len(test_image_folder_name):], processed_image)
    return processed_image


def markings_in_image(image_name, save, save_folder=final_image_folder_name):
    """Highlight white/yellow road markings in image at IMAGE NAME, saving in SAVE FOLDER if SAVE"""
    processed_image = process_image(image_name, True)
    for i in range(len(processed_image)):
        for j in range(len(processed_image[0])):
            if processed_image[i][j] < 0.05:
                processed_image[i][j] = 0
            else:
                processed_image[i][j] = 1
    if save:
        io.imsave(save_folder + "\\" + image_name[len(test_image_folder_name):], processed_image)
    return processed_image


def noise_removal(image):
    """Attempt to smooth noise in IMAGE, do not use; very time and resource intensive, not as good as closing"""
    returned = image[:]
    for i in range(len(image)):
        for j in range(len(image[0])):
            print(i, j)
            count = 0
            for pixel in get_adjacent(i, j, len(image), len(image[0])):
                count += image[pixel[0]][pixel[1]]
            if count / len(get_adjacent(i, j, len(image), len(image[0]))) >= 0.5:
                returned[i][j] = 1
            else:
                returned[i][j] = 0
    return returned


def read_frames_from_image(image_path, folder_name, num_frames=-1):
    """Read and save NUM FRAMES (or all if not given) frames from video at IMAGE PATH in folder FOLDER NAME"""
    cam = cv2.VideoCapture(image_path)

    try:

        # creating a folder named data
        if not os.path.exists(folder_name):
            os.makedirs(num_frames)

    # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')

    # frame
    currentframe = 0

    while True:

        # reading from frame
        ret, frame = cam.read()

        if ret:
            # if video is still left continue creating images
            name = './data/frame' + str(currentframe) + '.jpg'
            print('Creating...' + name)

            # writing the extracted images
            cv2.imwrite(name, frame)

            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
            # print(currentframe, numFrames, currentframe==numFrames)
            if currentframe == num_frames:
                return "Done"
        else:
            break

    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()


def __main__():
    start_time = time.time()
    frame_folder_name = "data"
    num_frames = 400
    read_frames_from_image(video_file_name, frame_folder_name, num_frames)
    frames = get_video_frames(frame_folder_name)
    processed_frames = []
    crack_data = []
    count = 0
    for frame in frames:
        processed_frames.append(cracks_in_image(frame, 0.3, True)[:])
    for frame in processed_frames:
        crack_data.append(vertical_crack_check(frame, [400, 800], 0.6, True, "vertical" + str(count)))
        crack_data.append(horizontal_crack_check(frame, [400, 800], 0.6, True, "horizontal" + str(count)))
        count += 1
    print(crack_data)
    print("--- %s seconds ---" % (time.time() - start_time))


if (__name__ == '__main__'):
    __main__()
