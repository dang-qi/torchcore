import cv2
import os
import glob

def extract_frames(video_path, out_path, resize=None):
    '''
        video_path: the path of video
        out_path: the output images folder
        resize: the size you want to resize the extracted frame, for example, (800, 450), (width, height)
    '''
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    while success:
        path = os.path.join(out_path, '{:06d}.jpg'.format(count))
        if resize is not None:
            image = cv2.resize(image, resize)
        cv2.imwrite(path, image)     # save frame as JPEG file
        success,image = vidcap.read()
        #print('Read a new frame: ', success)
        if count% 100==0:
            print('Save frame {}'.format(count))
        count += 1

def generate_video(frame_folder, out_path, fourcc='MJPG', frame_rate=24):
    '''
        frame_folder: the folder where the frame images are in
        out_path: the path of output video
    '''
    images = sorted(glob.glob("{}/*.*".format(frame_folder)))
    image = cv2.imread(images[0])
    height, width, layers = image.shape
    size = (width,height)
    out = cv2.VideoWriter(out_path,cv2.VideoWriter_fourcc(*fourcc), frame_rate, size)
    for image_path in images:
        image = cv2.imread(image_path)
        out.write(image)
    out.release()