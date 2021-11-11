import os
import argparse
import cv2
from PIL import Image
    
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
        gdd.download_file_from_google_drive(file_id='12I6W9SeHFmDSHLoJKp3iNSry3gw8ILAJ',
                                            dest_path='./models/bright-field.pth',
                                            unzip=False)
    
    image = cv2.imread(args.in_path)
    model = BrightfieldPredictor(weights_path='models/bright-field.pth',
                                 confidence=args.confidence)
    
    #out_image = model.predict_large(image)
    out_image = model.predict_large_overlap(image)
    out_image = Image.fromarray(out_image)
    out_image.save(args.out_path)
    
if __name__ == '__main__':
    main()