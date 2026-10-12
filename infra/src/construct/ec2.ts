// Lib imports
import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as fs from 'fs';
import * as path from 'path';
import { Construct } from 'constructs';

// App imports
import { IProps, IStackResources } from '@/modules/interfaces';

// The launch template every study machine starts from. scripts/launch.sh runs one machine per replicate, tagged
// Name=exp_k and Project=<stack name>; the boot script (user-data.sh) reads its Name, runs that replicate, keeps
// s3://<bucket>/exp_k/ current, and powers off, which terminates the machine (on-demand, never spot: a replicate
// runs about 11 hours and cannot resume).
export class CEC2 extends Construct {
  public readonly launchTemplate: ec2.LaunchTemplate;

  // Every machine launched from this template is tagged with this key/value (scripts/launch.sh); the status and
  // terminate scripts select machines by it.
  public readonly tagKey = 'Project';
  public readonly tagValue: string;

  constructor(scope: Construct, id: string, props: IProps) {
    super(scope, id);
    const stack = scope as unknown as IStackResources;
    const { config } = props;
    const ec2Config = config.aws.construct.ec2;
    this.tagValue = config.stack.name;

    // 1. Boot script, templated with this stack's bucket, image and limits
    const image = `${stack.resources.ecr.repository.repositoryUri}:${ec2Config.imageTag}`;
    const script = fs.readFileSync(path.join(__dirname, 'user-data.sh'), 'utf8')
      .replace('__BUCKET__', stack.resources.s3.bucket.bucketName)
      .replace('__IMAGE__', image)
      .replace('__MAX_HOURS__', `${ec2Config.maxHours}`)
      .replace('__REGION__', config.aws.region)
      .replace('__REPLICATES__', `${ec2Config.replicates}`)
      .replace('__SYNC_MINUTES__', `${ec2Config.syncMinutes}`);

    // 2. Launch template — ARM AMI + instance type + role (CDK wires the instance profile) + security group + root
    // volume. No subnet baked in: scripts/launch.sh passes --subnet-id (one of SubnetIds below); public subnets give
    // each machine its public IP.
    this.launchTemplate = new ec2.LaunchTemplate(
      this,
      `${id}-lt`,
      {
        blockDevices: [{
          deviceName: '/dev/xvda',
          volume: ec2.BlockDeviceVolume.ebs(ec2Config.volumeGb, {
            deleteOnTermination: true,
            volumeType: ec2.EbsDeviceVolumeType.GP3,
          }),
        }],
        // Powering off at the end of the replicate (or at maxHours) terminates the machine, so billing stops
        instanceInitiatedShutdownBehavior: ec2.InstanceInitiatedShutdownBehavior.TERMINATE,
        // The boot script reads the machine's Name tag (exp_k) from the instance metadata
        instanceMetadataTags: true,
        instanceType: new ec2.InstanceType(ec2Config.instanceType),
        machineImage: ec2.MachineImage.latestAmazonLinux2023({ cpuType: ec2.AmazonLinuxCpuType.ARM_64 }),
        requireImdsv2: true,
        role: stack.resources.iam.RMachine,
        securityGroup: stack.resources.net.sg,
        userData: ec2.UserData.custom(script),
      },
    );

    // Tagging
    cdk.Tags.of(this.launchTemplate).add('Name', `${config.stack.name}-machine`);

    // ── Outputs the scripts read (scripts/common.sh) ──
    new cdk.CfnOutput(scope, 'LaunchTemplateId', {
      description: 'Launch template of the study machines (scripts/launch.sh)',
      value: this.launchTemplate.launchTemplateId!,
    });
    new cdk.CfnOutput(scope, 'SubnetIds', {
      description: 'Public subnets the machines are spread over, comma-separated',
      value: stack.resources.net.vpc.publicSubnets.map((subnet) => subnet.subnetId).join(','),
    });
    new cdk.CfnOutput(scope, 'BucketName', {
      description: 'Results bucket, one prefix per replicate (exp_k/)',
      value: stack.resources.s3.bucket.bucketName,
    });
    new cdk.CfnOutput(scope, 'RepositoryUri', {
      description: 'ECR repository of the study image (scripts/push-image.sh)',
      value: stack.resources.ecr.repository.repositoryUri,
    });
    new cdk.CfnOutput(scope, 'TagKey', {
      description: 'Tag key/value every study machine carries',
      value: `${this.tagKey}=${this.tagValue}`,
    });
  }
}
