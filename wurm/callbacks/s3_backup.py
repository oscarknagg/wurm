import boto3
import os
import warnings
from typing import Dict
import torch

from .core import Callback, Optional


class S3Backup(Callback):
    """Backs up selected files to S3.

    Args:
        bucket: S3 bucket to backup files.
        backup_dict: Dictionary mapping from local file path to s3 filepath. This can
            contain formatting which will be filled with the vlaues of keys in `logs`
            in the same way as ModelCheckpoint.
        interval: Number of steps between backups.
    """
    def __init__(self, bucket: str, backup_dict: Dict[str, str], interval: int = 1000):
        super(S3Backup, self).__init__()
        self.bucket = bucket
        self.backup_dict = backup_dict
        self.interval = interval
        self.s3 = boto3.client('s3')

    def after_step(self,
                   logs: Optional[dict],
                   obs: Optional[Dict[str, torch.Tensor]] = None,
                   rewards: Optional[Dict[str, torch.Tensor]] = None,
                   dones: Optional[Dict[str, torch.Tensor]] = None,
                   infos: Optional[Dict[str, torch.Tensor]] = None):
        for local_file, s3_file in self.backup_dict:
            local_file = local_file.format(**logs)
            s3_file = s3_file.format(**logs)
            if os.path.exists(local_file):
                self.s3.upload_file(local_file, self.bucket, s3_file)
            else:
                warnings.warn('Expected file {} was not found.'.format(local_file))
