// Lib imports
import { z as zod } from 'zod';

export const SchemaQOS = zod.object({
  stack: zod.object({
    name: zod.string(),
  }),
  aws: zod.object({
    account: zod.string(),
    region: zod.string(),
    construct: zod.object({
      // The study's machines: one replicate each (tag Name=exp_k), terminated when it ends or at maxHours
      ec2: zod.object({
        imageTag: zod.string().default('latest'),
        instanceType: zod.string().default('c8g.xlarge'),
        maxHours: zod.number().min(1).max(48).default(16), // hard cap per machine, hours
        replicates: zod.number().min(1).default(20), // the study's --num-experiments
        syncMinutes: zod.number().min(1).max(60).default(15), // results copied to S3 this often
        volumeGb: zod.number().min(20).max(200).default(30),
      }),
      ecr: zod.object({
        expire: zod.number().min(1).max(14).default(7),
        max: zod.number().min(1).max(10).default(5),
        name: zod.string(),
      }),
      net: zod.object({
        azs: zod.number().min(1).max(6).default(3), // availability zones (one public subnet each) the machines spread over
        cidr: zod.string().regex(/^([0-9]{1,3}\.){3}[0-9]{1,3}\/[0-9]{1,2}$/),
        name: zod.string(),
      }),
      s3: zod.object({
        name: zod.string(),
      }),
    }),
  }),
});

export type QosConfig = zod.infer<typeof SchemaQOS>;
