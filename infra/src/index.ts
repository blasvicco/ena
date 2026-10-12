// Lib imports
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';

// App imports
import * as rawQos from '@/qos/qos.json';
import { logger } from '@/modules/logger';
import { SchemaQOS } from '@/qos/schema';
import { SExperiment } from '@/stack/experiment';

// Constants
const app = new cdk.App();

try {
  // 1. Validate the JSON against the schema
  rawQos.aws.account = rawQos.aws.account || process.env.CDK_DEFAULT_ACCOUNT || '';
  rawQos.aws.region = rawQos.aws.region || process.env.CDK_DEFAULT_REGION!;
  const qos = SchemaQOS.parse(rawQos);

  // 2. Initialize the Experiment Stack
  // An empty account (no credentials yet) synthesizes an environment-agnostic template
  new SExperiment(
    app,
    `${qos.stack.name}`,
    {
      env: {
        account: qos.aws.account || undefined,
        region: qos.aws.region,
      },
      config: qos,
    },
  );
} catch (err) {
  logger.error({ err }, 'Failed to initialize stack');
  process.exit(1);
}
