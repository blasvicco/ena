// Lib imports
import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';
import { Construct } from 'constructs';

// App imports
import { IProps } from '@/modules/interfaces';

// The study's results: one prefix per replicate (s3://<bucket>/exp_k/), kept current by each machine while it runs.
// Unlike cb-infra's buckets (RETAIN), this one is destroyed with the stack, objects included: the results are
// downloaded first (scripts/download.sh), and nothing is left behind once the study is over.
export class CS3 extends Construct {
  public readonly bucket: s3.Bucket;

  constructor(scope: Construct, id: string, props: IProps) {
    super(scope, id);
    const { config } = props;
    const name = config.aws.construct.s3.name;
    this.bucket = new s3.Bucket(
      this,
      `${id}-bucket-${name}`,
      {
        autoDeleteObjects: true,
        bucketName: name,
        blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
        encryption: s3.BucketEncryption.S3_MANAGED,
        enforceSSL: true,
        removalPolicy: cdk.RemovalPolicy.DESTROY,
      },
    );

    // Tagging the S3
    cdk.Tags.of(this.bucket).add('Name', name);
  }
}
