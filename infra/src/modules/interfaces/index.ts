// Lib imports
import * as cdk from 'aws-cdk-lib';

// App imports
import { CEC2, CECR, CIAM, CNet, CS3 } from '@/construct';
import { QosConfig } from '@/qos/schema';

export interface IProps extends cdk.StackProps {
  readonly config: QosConfig;
};

export interface IResources {
  ec2: CEC2;
  ecr: CECR;
  iam: CIAM;
  net: CNet;
  s3: CS3;
};

export interface IStackResources {
  readonly resources: IResources;
};
