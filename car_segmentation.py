import cv2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
coco_metadata = MetadataCatalog.get("coco_2017_val")

from detectron2.projects import point_rend

cfg = get_cfg()
point_rend.add_pointrend_config(cfg)
cfg.merge_from_file("./pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml")
cfg.MODEL.WEIGHTS = "./model_final_ba17b9.pkl"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.DEVICE = "cpu"
predictor = DefaultPredictor(cfg)
im = cv2.imread("./back.jpg")
outputs = predictor(im)

# print(outputs['instances']._fields.keys())
v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1., instance_mode=ColorMode.IMAGE_BW)
point_rend_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
cv2.namedWindow('result', cv2.WINDOW_NORMAL)
cv2.resizeWindow('result', 1920, 1080)
cv2.imshow('result', point_rend_result[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()