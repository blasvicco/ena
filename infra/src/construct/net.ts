// Lib imports
import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import { Construct } from 'constructs';

// App imports
import { IProps } from '@/modules/interfaces';

// Public subnets only, no NAT gateway: each machine reaches ECR and S3 over its own public IP (S3 through the free
// gateway endpoint) and accepts no inbound connection (Session Manager works over the outbound path).
export class CNet extends Construct {
  public readonly sg: ec2.SecurityGroup;
  public readonly vpc: ec2.Vpc;

  constructor(scope: Construct, id: string, props: IProps) {
    super(scope, id);
    const { config } = props;
    const region = config.aws.region;
    this.vpc = new ec2.Vpc(
      this,
      `${id}-vpc`,
      {
        ipAddresses: ec2.IpAddresses.cidr(config.aws.construct.net.cidr),
        maxAzs: 2,
        subnetConfiguration: [{
          name: 'Public',
          subnetType: ec2.SubnetType.PUBLIC,
          cidrMask: 24,
        }],
        // The machines only pull an image and copy results: no NAT cost
        natGateways: 0,
      },
    );

    // S3 Gateway Endpoint (Free)
    this.vpc.addGatewayEndpoint(
      `${id}-vpc-s3-gateway`,
      { service: ec2.GatewayVpcEndpointAwsService.S3 },
    );

    // Security group — outbound only (ECR, S3, package mirrors, SSM). No inbound.
    this.sg = new ec2.SecurityGroup(
      this,
      `${id}-sg`,
      {
        vpc: this.vpc,
        allowAllOutbound: true,
        description: 'ena-experiment study machines - outbound internet only',
      },
    );

    // Tagging the VPC
    cdk.Tags.of(this.vpc).add('Name', config.aws.construct.net.name);
    cdk.Tags.of(this.vpc).add('Region', region);
    cdk.Tags.of(this.sg).add('Name', `${config.aws.construct.net.name}-sg`);
  }
}
