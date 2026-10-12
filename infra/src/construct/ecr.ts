// Lib imports
import * as cdk from 'aws-cdk-lib';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import { Construct } from 'constructs';

// App imports
import { IProps } from '@/modules/interfaces';

// The study's image (hub/cloud/Dockerfile), pushed by scripts/push-image.sh and pulled by every machine.
export class CECR extends Construct {
  public readonly repository: ecr.Repository;

  constructor(scope: Construct, id: string, props: IProps) {
    super(scope, id);
    const { config } = props;
    this.repository = new ecr.Repository(
      this,
      `${id}-repository`,
      {
        repositoryName: config.aws.construct.ecr.name,
        // Removes the repo when the stack is deleted
        removalPolicy: cdk.RemovalPolicy.DESTROY,
        // Automatically deletes images inside when the repo is deleted
        emptyOnDelete: true,
        imageScanOnPush: false,
        // Tags are the commit the image was built from, plus 'latest'
        imageTagMutability: ecr.TagMutability.MUTABLE,
      },
    );

    // Add a lifecycle rule to clean up old untagged images to save costs
    this.repository.addLifecycleRule({
      description: 'Remove untagged images after N days.',
      maxImageAge: cdk.Duration.days(config.aws.construct.ecr.expire),
      rulePriority: 1,
      tagStatus: ecr.TagStatus.UNTAGGED,
    });

    // Clean up old versioned/tagged images
    this.repository.addLifecycleRule({
      description: 'Keep only the last N tagged images to reduce costs',
      maxImageCount: config.aws.construct.ecr.max,
      tagStatus: ecr.TagStatus.ANY,
      rulePriority: 2,
    });

    // Tagging the ECR
    cdk.Tags.of(this.repository).add('Name', config.aws.construct.ecr.name);
  }
}
