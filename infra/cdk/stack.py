"""AWS CDK Python stack — alternative to Terraform."""
from __future__ import annotations

import aws_cdk as cdk
from aws_cdk import (
    aws_s3 as s3,
    aws_ecr as ecr,
    aws_iam as iam,
)
from constructs import Construct


class MyProjectStack(cdk.Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # S3 data bucket
        data_bucket = s3.Bucket(
            self, "DataBucket",
            removal_policy=cdk.RemovalPolicy.RETAIN,
            # TODO: add encryption, versioning
        )

        # ECR repository
        repo = ecr.Repository(
            self, "ModelRepo",
            repository_name="my-project",   # TODO: rename
            image_scan_on_push=True,
        )

        # SageMaker execution role
        sagemaker_role = iam.Role(
            self, "SageMakerRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess")
            ],
        )

        # TODO: add VPC, ECS, Lambda, or other resources


app = cdk.App()
MyProjectStack(app, "MyProjectStack")
app.synth()
