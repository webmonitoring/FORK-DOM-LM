import sagemaker
from sagemaker.pytorch import PyTorch
from dotenv import load_dotenv
import os

sess = sagemaker.Session()
load_dotenv()
role = os.environ["SAGEMAKER_ROLE_ARN"]
bucket = os.environ["SAGEMAKER_BUCKET"]

estimator = PyTorch(
    entry_point="entry_point_sagemaker.sh",
    source_dir  = f"s3://{bucket}/domlm/code/",   # points at the folder containing train.py & src/
    role        = role,
    framework_version="2.0.0",
    py_version       ="py310",
    instance_count   = 1,
    instance_type    ="ml.p3.2xlarge",
    volume_size      = 100,                       # GB
    max_run          = 24*60*60,                  # seconds
    hyperparameters  = {
        "train-data-dir": f"/opt/ml/input/data/train",
        "epochs": 5,
        "per-device-batch": 8,
        "gradient-accum": 1,
        "mlm-prob": 0.15,
    },
    input_mode="File",
    input_data_config=[
        {
            "ChannelName":"train",
            "DataSource":{
                "S3DataSource":{
                    "S3Uri": f"s3://{bucket}/domlm/data/swde_ae_preprocessed.zip",
                    "S3DataType":"S3Prefix",
                    "S3InputMode":"File"
                }
            }
        }
    ],
    output_path = f"s3://{bucket}/domlm/output/",
    checkpoint_s3_uri = f"s3://{bucket}/domlm/checkpoints/",
    checkpoint_local_path = "/opt/ml/checkpoints",
    enable_sagemaker_metrics=True,
)

# Launch asynchronously
estimator.fit(wait=False)
print("TrainingJob started:", estimator.latest_training_job.name)
