from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference_single_image
import numpy as np
import cv2
import torch

class Encoder(object):
    def __init__(self, yaml_path = '/workspace/detectron2/configs/VG-Detection/'):
        """
        """
        super().__init__()

        #init configuration
        self.cfg = get_cfg()
        self.cfg.merge_from_file(yaml_path + "faster_rcnn_R_101_C4_caffe.yaml")
        self.cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
        self.cfg.INPUT.MIN_SIZE_TEST = 600
        self.cfg.INPUT.MAX_SIZE_TEST = 1000
        self.cfg.MODEL.RPN.NMS_THRESH = 0.7
        # VG Weight
        self.cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe.pkl"

        self.detector = DefaultPredictor(self.cfg)

        self.MIN_BOXES = 10
        self.MAX_BOXES = 100
        self.conf_thresh = 0.2

    def encode(self, imgs):
        instances_list, features_list = self.doit(imgs)

        outs = []

        for img, instances, features in zip(imgs, instances_list, features_list):
            
            instances = instances.to('cpu')
            features  = features.to('cpu')
            
            num_objects = len(instances)
            boxes = instances.pred_boxes.tensor.cpu()
            conf = instances._fields['scores'].cpu()
            keep_boxes = np.where(conf >= self.conf_thresh)[0]

            if len(keep_boxes) < self.MIN_BOXES:
                keep_boxes = np.argsort(-1 * conf)[:self.MIN_BOXES]
            elif len(keep_boxes) > self.MAX_BOXES:
                keep_boxes = np.argsort(-1 * conf)[:self.MAX_BOXES]

            boxes    = boxes[keep_boxes]
            features = features[keep_boxes]

            img_h = img.shape[0]
            img_w = img.shape[1]

            box_width = boxes[:,2] - boxes[:, 0]
            box_height = boxes[:, 3] - boxes[:, 1]

            scaled_width = box_width / img_w
            scaled_height = box_height / img_h
            scaled_x = boxes[:, 0] / img_w
            scaled_y = boxes[:, 1] / img_h

            scaled_width = scaled_width[..., np.newaxis]
            scaled_height = scaled_height[..., np.newaxis]
            scaled_x = scaled_x[..., np.newaxis]
            scaled_y = scaled_y[..., np.newaxis]

            spatial_feat = np.concatenate((scaled_x, scaled_y, scaled_x + scaled_width, scaled_y + scaled_height,
                                            scaled_width, scaled_height), axis=1)
            full_feat = np.concatenate((features, spatial_feat), axis=1)

            classes = instances._fields['pred_classes'][keep_boxes]

            outs.append([full_feat, classes.numpy()])
        
        return outs

    def doit(self, raw_images):
        with torch.no_grad():
            # Preprocessing
            inputs = []
            for raw_image in raw_images:
                image = self.detector.transform_gen.get_transform(raw_image).apply_image(raw_image)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                inputs.append(
                    {"image": image, "height": raw_image.shape[0], "width": raw_image.shape[1]})
            images = self.detector.model.preprocess_image(inputs)

            # Run Backbone Res1-Res4
            features = self.detector.model.backbone(images.tensor)

            # Generate proposals with RPN
            proposals, _ = self.detector.model.proposal_generator(images, features, None)

            # Run RoI head for each proposal (RoI Pooling + Res5)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            features = [features[f] for f in self.detector.model.roi_heads.in_features]
            box_features = self.detector.model.roi_heads._shared_roi_transform(features, proposal_boxes)
        
            # (sum_proposals, 2048), pooled to 1x1
            feature_pooled = box_features.mean(dim=[2, 3])

            # Predict classes and boxes for each proposal.
            pred_class_logits, pred_proposal_deltas = self.detector.model.roi_heads.box_predictor(feature_pooled)
            rcnn_outputs = FastRCNNOutputs(
                self.detector.model.roi_heads.box2box_transform,
                pred_class_logits,
                pred_proposal_deltas,
                proposals,
                self.detector.model.roi_heads.smooth_l1_beta,
            )

            # Fixed-number NMS
            instances_list, ids_list = [], []
            probs_list = rcnn_outputs.predict_probs()
            boxes_list = rcnn_outputs.predict_boxes()
            for probs, boxes, image_size in zip(probs_list, boxes_list, images.image_sizes):
                for nms_thresh in np.arange(0.3, 1.0, 0.1):
                    instances, ids = fast_rcnn_inference_single_image(
                        boxes, probs, image_size,
                        score_thresh=0.2, nms_thresh=nms_thresh, topk_per_image=self.MAX_BOXES
                    )
                    if len(ids) >= self.MIN_BOXES:
                        break
                instances_list.append(instances)
                ids_list.append(ids)

            # Post processing for features
            # (sum_proposals, 2048) --> [(p1, 2048), (p2, 2048), ..., (pn, 2048)]
            features_list = feature_pooled.split(rcnn_outputs.num_preds_per_image)
            roi_features_list = []
            for ids, features in zip(ids_list, features_list):
                roi_features_list.append(features[ids].detach())

            # Post processing for bounding boxes (rescale to raw_image)
            raw_instances_list = []
            for instances, input_per_image, image_size in zip(instances_list, inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                raw_instances = detector_postprocess(instances, height, width)
                raw_instances_list.append(raw_instances)

            return raw_instances_list, roi_features_list

    def __call__(self, imgs):
        return self.encode(imgs)









        
