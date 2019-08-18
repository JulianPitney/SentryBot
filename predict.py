from keras.optimizers import Adam
import numpy as np
from losses.keras_ssd_loss import SSDLoss
from models.keras_mobilenet_v2_ssdlite import mobilenet_v2_ssd
import os
import cv2
import datetime
from tqdm import tqdm

class MobileNetV2SSD:

    def __init__(self):

        batch_size = 32
        self.image_size = (300, 300, 3)
        n_classes = 80
        mode = 'inference_fast'
        l2_regularization = 0.0005
        min_scale = 0.1  # None
        max_scale = 0.9  # None
        scales = None  # [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
        aspect_ratios_global = None
        aspect_ratios_per_layer = [[1.0, 2.0, 0.5], [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0], [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                   [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0], [1.0, 2.0, 0.5], [1.0, 2.0, 0.5]]
        two_boxes_for_ar1 = True
        steps = None  # [8, 16, 32, 64, 100, 300]
        offsets = None  # [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        clip_boxes = False
        variances = [0.1, 0.1, 0.2, 0.2]
        coords = 'centroids'
        normalize_coords = True
        subtract_mean = [123, 117, 104]
        divide_by_stddev = 128
        swap_channels = None
        confidence_thresh = 0.01
        iou_threshold = 0.45
        top_k = 10
        nms_max_output_size = 400
        return_predictor_sizes = False

        model = mobilenet_v2_ssd(self.image_size, n_classes, mode, l2_regularization, min_scale, max_scale, scales,
                                 aspect_ratios_global, aspect_ratios_per_layer, two_boxes_for_ar1, steps,
                                 offsets, clip_boxes, variances, coords, normalize_coords, subtract_mean,
                                 divide_by_stddev, swap_channels, confidence_thresh, iou_threshold, top_k,
                                 nms_max_output_size, return_predictor_sizes)

        # 2: Load the trained weights into the model.
        weights_path = os.path.join("pretrained_weights", "ssdlite_coco_loss-4.8205_val_loss-4.1873.h5")
        model.load_weights(weights_path, by_name=True)
        # 3: Compile the model so that Keras won't complain the next time you load it.
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
        model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
        self.model = model

    def predict(self, img, threshold=0.85):
        ori_img_size = img.shape
        img = cv2.resize(img, (self.image_size[0], self.image_size[1]))
        padding_frame = np.zeros(self.image_size)
        padding_frame[:, :, 0] = img
        padding_frame[:, :, 1] = img
        padding_frame[:, :, 2] = img
        img = np.asarray(padding_frame).astype(np.uint8)
        y_pred = self.model.predict(np.asarray([img])).tolist()
        np.set_printoptions(precision=2, suppress=True, linewidth=90)
        xmin, ymin, xmax, ymax = -1, -1, -1, -1

        def confidence_key(item):
            return item[0][1]
        y_pred.sort(key=confidence_key)
        box = y_pred[0][0]
        if box[0] == 1 and box[1] >= threshold:
            xmin = int(box[2] * ori_img_size[0] / self.image_size[0])
            ymin = int(box[3] * ori_img_size[1] / self.image_size[1])
            xmax = int(box[4] * ori_img_size[0] / self.image_size[1])
            ymax = int(box[5] * ori_img_size[1] / self.image_size[0])
        print(ori_img_size)
        print(xmin, ymin, xmax, ymax)
        return xmin, ymin, xmax, ymax
        
class Tracker:
    def __init__(self):
        self.M = MobileNetV2SSD()
        self.tracker = cv2.TrackerMOSSE_create()
        self.status = False
        self.image_size = (300, 300, 3)

    def track(self, frame):
        # frame = cv2.resize(frame, (self.image_size[0], self.image_size[1]))


        if not self.status:
            xmin, ymin, xmax, ymax = self.M.predict(frame)
            if xmin != -1:
                self.tracker.init(frame, (xmin, ymin, xmax, ymax))
                self.status = True
        else:
            (success, box) = self.tracker.update(frame)
            if not success:
                self.status = False
                xmin, ymin, xmax, ymax = -1, -1, -1, -1
            else:
                (x, y, w, h) = [int(v) for v in box]
                xmin, ymin, xmax, ymax = x, y, x + w, y + h
        return xmin, ymin, xmax, ymax


if __name__ == '__main__':
    T = Tracker()
    img_path = os.path.join("experiments", "imgs", "000001.jpg")
    img = cv2.imread(img_path)
    start_time = datetime.datetime.now()
    for i in tqdm(range(100)):
        xmin, ymin, xmax, ymax = T.track(img)
    end_time = datetime.datetime.now()
    print("FPS: %.3f"%(100. / float((end_time-start_time).seconds)))
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2, 8)
    cv2.imshow('test', img)
    cv2.waitKey()
