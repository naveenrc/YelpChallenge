import os
from utilities.fileOperations import FileOperations
import numpy as np

if __name__ == '__main__':
    fo = FileOperations()
    path = os.path.dirname(__file__)

    photos_path = os.path.join(path, '../yelpData/') + 'photos.json'
    photos_df = fo.getfile(photos_path).set_index('photo_id')

    # Images count
    print('Total images: ' + str(len(photos_df)))

    # Business count
    print('Total Businesses: ' + str(photos_df['business_id'].unique().size))

    # Duplicate images
    print('Duplicated images: ' + str(photos_df.reset_index().duplicated(subset=['photo_id']).any()))

    # Sanity check for missing yelpData
    print('Any images without labels: ' + str(photos_df['label'].isnull().any()))
    print('Any missing business id: ' + str(photos_df['business_id'].isnull().any()))

    print('-----------------------------------------')
    print('Distribution of images among 5 labels:')
    # Display image labels count
    print(photos_df['label'].value_counts().reset_index().
          rename(columns={'index': 'label', 'label': 'count'}))

    print('-----------------------------------------')
    photos_df['caption'].replace('', np.nan, inplace=True)
    photos_df.dropna(subset=['caption'], inplace=True)
    print('Number of photos with caption: ' + str(len(photos_df)))
