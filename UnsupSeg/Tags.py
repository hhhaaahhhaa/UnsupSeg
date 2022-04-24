class ModelTag(object):
    """
    Availabel tags:
        TIMIT, BUCKEYE
    """

    TIMIT = {
        "src": "pretrained_models/timit+_pretrained.ckpt",
        "peak-detection-params": {
            'cpc_1': {'prominence': 0.05, 'width': None, 'distance': None, 'epoch': 2}
        }
    }
    BUCKEYE = {
        "src": "pretrained_models/buckeye+_pretrained.ckpt",
        "peak-detection-params": {
            'cpc_1': {'prominence': 0.07, 'width': None, 'distance': None, 'epoch': 7}
        },
    }
