# inference by pytorch without cloning repo
import torch


def show_model_list():
    x = torch.hub.get_dir()
    print(x)

    list = torch.hub.list('ultralytics/yolov5')
    for i in list:
        print(i)


def load_image():
    import cv2
    from PIL import Image
    from PIL import ImageGrab
    # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
    #   file:       imgs = 'data/images/zidane.jpg'  # str or PosixPath
    #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
    #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
    #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
    #   numpy:           = np.zeros((640,1280,3))  # HWC
    #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
    #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images
    img1 = 'test_img.jpg'
    img2 = cv2.imread('data/images/bus.jpg')[..., ::-1]
    img3 = Image.open('data/images/zidane.jpg')
    imgs = [img1, img2, img3]
    # imgs = ImageGrab.grab()
    imgs = img1
    return imgs


def build_model(src):
    if src == 'local':
        mdl = torch.hub.load('', 'yolov5s', source='local')
        print('Model built successfully from local.')
    elif src == 'repo':
        mdl = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom
        print('Model built successfully from repo.')
    else:
        print('Source error.')
    return mdl


if __name__ == '__main__':
    # source of torch.hub.load
    model = build_model('local')
    img = load_image()
    results = model(img)

    # Results print() show() save() crop() pandas()
    results.print()
    # results.show()
    # results.crop()
    # crops = results.crop()
    # print(crops)
    # ans = results.pandas().xyxy[0]
    # ans = results.pandas().xyxy[0].sort_values('xmin')
    ans = results.pandas().xyxy[0].to_json(orient='records')
    print(ans)
