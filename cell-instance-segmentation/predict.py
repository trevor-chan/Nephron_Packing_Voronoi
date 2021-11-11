import os
import argparse
import cv2
from PIL import Image
import numpy as np

from google_drive_downloader import GoogleDriveDownloader as gdd

from modules import BrightfieldPredictor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path',
                        '-i',
                        default='in.jpeg')
    parser.add_argument('--out_path',
                        '-o',
                        default='out.jpeg')
    parser.add_argument('--confidence',
                        '-c',
                        type=float,
                        default=0.6)

    args = parser.parse_args()



    if not os.path.exists('models/bright-field.pth'):
        gdd.download_file_from_google_drive(file_id='1dpn0xVD4pJmRtqzyLTUJ2ERjwLvRxWVM',
                                            dest_path='./models/bright-field.pth',
                                            unzip=False)
        # 12I6W9SeHFmDSHLoJKp3iNSry3gw8ILAJ <- old model weights

    image = cv2.imread(args.in_path)
    model = BrightfieldPredictor(weights_path='models/bright-field.pth',
                                 confidence=args.confidence)

    image = np.pad(image, ((30, 30), (30, 30), (0, 0)),
          mode='constant', constant_values=0)

    predictions = model.predict_large(image, nmsalg='bbox')
    out_img = model.visualize(image, predictions)
    out_img.save(args.out_path)

if __name__ == '__main__':
    main()
