import numpy as np
import torch
from typing import List, Dict
import copy
from argparse import Namespace
import dill

from . import Define
from .next_frame_classifier import NextFrameClassifier
from .utils import detect_peaks, max_min_norm, replicate_first_k_frames


class SegmentModel(object):
    def __init__(self, config):
        self.config = config
        self.model_path = config["src"]

        # load weights and peak detection params
        ckpt = torch.load(self.model_path, map_location=lambda storage, loc: storage)
        hp = Namespace(**dict(ckpt["hparams"]))

        self.model = NextFrameClassifier(hp)
        weights = ckpt["state_dict"]
        weights = {k.replace("NFC.", ""): v for k,v in weights.items()}
        self.model.load_state_dict(weights)
        self.peak_detection_params = config["peak-detection-params"]['cpc_1']

        # Origin pickle object consists dependence, can not change module name "solver.py".
        # self.peak_detection_params = dill.loads(ckpt['peak_detection_params'])['cpc_1']

    def predict(self, wav: np.array, prominence=None) -> List[float]:
        """
        Predict from numpy float array, please ensure sample rate is 16000. 
        """
        if Define.DEBUG:
            self.log(f"running inferece using ckpt: {self.model_path}")

        if prominence is not None:
            self.peak_detection_params["prominence"] = prominence
        audio = torch.from_numpy(wav).unsqueeze(0)

        # run inference
        preds = self.model(audio)  # get scores
        preds = preds[1][0]  # get scores of positive pairs
        preds = replicate_first_k_frames(preds, k=1, dim=1)  # padding
        preds = 1 - max_min_norm(preds)  # normalize scores (good for visualizations)
        preds = detect_peaks(x=preds,
                            lengths=[preds.shape[1]],
                            prominence=self.peak_detection_params["prominence"],
                            width=self.peak_detection_params["width"],
                            distance=self.peak_detection_params["distance"])  # run peak detection on scores
        preds = preds[0] * 160 / Define.SAMPLE_RATE  # transform frame indexes to seconds

        if Define.DEBUG:
            self.log("Predicted boundaries (in seconds):")
            self.log(preds)

        return preds

    def log(self, msg):
        print("[UnsupSeg]: ", msg)


def load_model_from_tag(tag: Dict):
    config = copy.deepcopy(tag)
    config['src'] = f"{Define.REPO_PATH}/{config['src']}"
    return SegmentModel(config)
