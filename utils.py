import os
import datetime
import re
def generate_log_name(model_name, augmentation, datasets):
    d = {'name': model_name, 'augmentation':augmentation, 'datasets': [ds['name'] for ds in datasets]}

    logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(d.items())))
    ))