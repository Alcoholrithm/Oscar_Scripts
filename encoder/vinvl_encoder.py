from numpy.core.fromnumeric import transpose
import torch
from maskrcnn_benchmark.config import cfg
from scene_graph_benchmark.config import sg_cfg
from scene_graph_benchmark.AttrRCNN import AttrRCNN
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
import numpy as np
from encoder.base_encoder import BaseEncoder

class Encoder(BaseEncoder):
    def __init__(self,
                    config_file = "/workspace/scene_graph_benchmark/sgg_configs/vgattr/vinvl_x152c4.yaml",
                    opts = ["TEST.IMS_PER_BATCH", "2", "MODEL.WEIGHT", "/workspace/shared/pretrained/vinvl_vg_x152c4.pth", "MODEL.ROI_HEADS.NMS_FILTER", "1", "MODEL.ROI_HEADS.SCORE_THRESH", "0.2", "TEST.IGNORE_BOX_REGRESSION", "True", "MODEL.ATTRIBUTE_ON", "True", "TEST.OUTPUT_FEATURE", "True"],
                    ckpt = "/workspace/shared/pretrained/vinvl_vg_x152c4.pth",
                    DEVICE = "cuda",
                    MIN_BOXES = 10,
                    MAX_BOXES = 100,
                    conf_threshold = 0.2):
        cfg.set_new_allowed(True)
        cfg.merge_from_other_cfg(sg_cfg)
        cfg.set_new_allowed(False)
        cfg.merge_from_file(config_file)
        cfg.local_rank = 0
        cfg.merge_from_list(opts)

        self.model = AttrRCNN(cfg)
        self.model.to(DEVICE)

        checkpointer = DetectronCheckpointer(cfg, self.model)
        _ = checkpointer.load(ckpt, use_latest=True)

        self.DEVICE = DEVICE

        self.MIN_BOXES = MIN_BOXES
        self.MAX_BOXES = MIN_BOXES

        self.conf_threshold = conf_threshold

    def permute_img(self, img):
        img = torch.Tensor(img).to(self.DEVICE)
        img = img.permute((2, 0, 1))
        return img

    def compute_on_image(self, images):
        self.model.eval()
        with torch.no_grad():
            output = self.model(images, None)
        output = [o.to('cpu') for o in output]
        return output

    def encode(self, imgs):
    
        for i in range(len(imgs)):
            imgs[i] = self.permute_img(imgs[i])
        pred = self.compute_on_image(imgs)

        outs = []
        for img, p in zip(imgs, pred):

            features = p.get_field('box_features')
            boxes = []
            for bs in p.get_field('boxes_all'):
                boxes.append(bs[0].numpy())
            boxes = np.array(boxes)
            conf = p.get_field('scores')
            labels = p.get_field('labels')
            
            keep_boxes = np.where(conf >= self.conf_threshold)[0]

            boxes    = boxes[keep_boxes]
            features = features[keep_boxes]
            labels = labels[keep_boxes]
            
            img_h = img.shape[1]
            img_w = img.shape[2]

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
            outs.append([full_feat, labels])

        return outs



