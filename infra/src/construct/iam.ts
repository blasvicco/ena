// Lib imports
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import { Construct } from 'constructs';

// App imports
import { IProps, IStackResources } from '@/modules/interfaces';

// The machines' instance role: pull the study's image, write the results bucket, and nothing else.
export class CIAM extends Construct {
  public readonly RMachine: iam.Role;

  constructor(scope: Construct, id: string, props: IProps) {
    super(scope, id);
    const stack = scope as unknown as IStackResources;
    const { config } = props;
    const { region } = config.aws;

    // 1. Instance role (CDK wires the instance profile through the launch template)
    const RMachineName = `${id}-machine-${region}`;
    this.RMachine = new iam.Role(
      this,
      RMachineName,
      {
        roleName: RMachineName,
        assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
      },
    );

    // 2. SSM Session Manager — a shell on a stuck machine for debugging, with no inbound rule and no SSH key
    // (AL2023 ships the SSM Agent running by default).
    this.RMachine.addManagedPolicy(
      iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSSMManagedInstanceCore'),
    );

    // 3. ECR — log in and pull the study's image (grantPull includes the account-level ecr:GetAuthorizationToken)
    stack.resources.ecr.repository.grantPull(this.RMachine);

    // 4. S3 — read and write the results bucket only
    stack.resources.s3.bucket.grantReadWrite(this.RMachine);

    // Tagging
    cdk.Tags.of(this.RMachine).add('Name', RMachineName);
    cdk.Tags.of(this.RMachine).add('Region', region);
  }
}
