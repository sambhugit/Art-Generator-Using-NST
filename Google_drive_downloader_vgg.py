from google_drive_downloader import GoogleDriveDownloader as gdd
import os
current_dir = os.path.dirname(os.path.realpath('Art_Generator.py'))

def weight_file_id():

    gdd.download_file_from_google_drive(file_id='16yez1ApDalqjFVj-kcKlpb16QkMm4Yi5',
                                    dest_path= os.path.join(current_dir,'Imagenet/imagenet-vgg-verydeep-19.mat'),
                                    unzip=False)
