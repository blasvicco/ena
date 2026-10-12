// Lib imports
import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';

// App imports
import { CEC2, CECR, CIAM, CNet, CS3 } from '@/construct';
import { logger } from '@/modules/logger';
import { IProps, IResources } from '@/modules/interfaces';

// The HalfCheetah study's cloud run: the image registry, the results bucket, the network, the machines' role and
// their launch template. Everything is destroyed with the stack (cdk destroy), the bucket's objects included.
export class SExperiment extends cdk.Stack {
  public readonly resources: Partial<IResources> = {};

  constructor(scope: Construct, id: string, props: IProps) {
    super(scope, id, props);

    const { config } = props;
    logger.info(`Building Stack for ${config.stack.name}.`);

    // Initialize ECR
    this.resources.ecr = new CECR(this, `${config.stack.name}-ecr`, { config });

    // Initialize S3
    this.resources.s3 = new CS3(this, `${config.stack.name}-s3`, { config });

    // Initialize Net
    this.resources.net = new CNet(this, `${config.stack.name}-net`, { config });

    // Initialize IAM - Depends on ECR, S3
    this.resources.iam = new CIAM(this, `${config.stack.name}-iam`, { config });

    // Initialize EC2 launch template - Depends on ECR, IAM, NET, S3
    this.resources.ec2 = new CEC2(this, `${config.stack.name}-ec2`, { config });
  }
}
