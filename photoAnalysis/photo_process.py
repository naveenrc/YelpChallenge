import os
from utilities.fileOperations import FileOperations
import time
import cv2
import sys
from multiprocessing import Pool


def resize_image(file):
    """
    Resize image to im_size x im_size which follows INTER_AREA interpolation

    Arguments: reads image

    Returns: writes resized image
    """
    path = os.path.dirname(__file__)
    temp_path = os.path.join(path, '../../YelpPhotos/') + file + '.jpg'
    img = cv2.imread(temp_path)
    im_size = int(input('Enter image size(64 if you need 64x64 images):'))
    resized = cv2.resize(img, (im_size, im_size), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(path, '../yelpData/resized48/') + file + '.png', resized)


if __name__ == '__main__':
    print('Image resizing starts: ')
    fo = FileOperations()
    path = os.path.dirname(__file__)

    # path to photos.json
    photos_path = os.path.join(path, '../yelpData/') + 'photos.json'
    photos_df = fo.getfile(photos_path)

    # map strings labels to integers
    labels = {'food': 0,
              'drink': 1,
              'inside': 2,
              'outside': 3,
              'menu': 4,
              'other': 5}

    start = time.clock()
    photos_df['label_int'] = photos_df['label'].map(lambda x: labels[x])
    stop = time.clock()
    # Write labels to file
    photos_df['label_int'].T.\
        to_csv(os.path.join(path, '../yelpData/')+'labels.csv', index=False)
    print('Written labels to csv')
    print('Time taken to map labels to integers: ' + str(stop-start))
    print('------------------------------------------------------------------------')

    # Record time
    start = time.clock()
    images = photos_df['photo_id']
    # Multiprocessing
    p = Pool(4)
    # Display progress
    for i, _ in enumerate(p.imap_unordered(resize_image, images), 1):
        sys.stderr.write('\rdone {0:%}'.format(i / len(images)))
    p.close()
    p.join()
    # Record stop time
    stop = time.clock()

    print('\nTime taken to resize images except food images: ' + str(stop-start))
