# Cloud run of the HalfCheetah study (AWS CDK)

The study (`app/src/half_cheetah.py`) runs once in AWS: **one replicate per on-demand `c7g.2xlarge`** (8 vCPU Graviton, like the local runs, so wall-clock times stay comparable), 20 replicates, region `eu-south-2`. Each machine pulls the study image from ECR, runs its replicate (`--replicate k`), keeps `s3://ena-experiment/exp_k/` current every 15 minutes, and powers off when it ends or at the hard cap (16 h), which terminates it. Everything here is destroyed with the stack, the bucket's objects included, so **download the results first**.

The CDK code follows the conventions of `cb-infra`: TypeScript run by ts-node, a Zod-validated QoS file, `S*` stacks, `C*` constructs, construct ids `${stack}-<svc>`.

```
src/                     CDK project (mounted at /home/app in the toolbox)
  index.ts               entry point: validates qos/qos.json, builds SExperiment
  stack/experiment.ts    SExperiment: ECR, S3, network, IAM, launch template
  construct/             ecr.ts s3.ts net.ts iam.ts ec2.ts, user-data.sh (each machine's boot script)
  qos/                   schema.ts, qos.template.json (copy to qos.json; git-ignored)
scripts/                 push-image.sh (host), launch.sh status.sh download.sh terminate.sh (toolbox), common.sh
```

## Before the first deploy
1. An AWS profile in `~/.aws` (the toolbox mounts it read-only; set `AWS_PROFILE` if it is not `default`).
2. A quota of **160 vCPUs** for "Running On-Demand Standard (A, C, D, H, I, M, R, T, Z) instances" in `eu-south-2` (Service Quotas).
3. `cp src/qos/qos.template.json src/qos/qos.json`, then set `aws.account` (or leave it empty to use the profile's account).
4. The bucket name `ena-experiment` must be free globally; if the deploy reports it taken, change `construct.s3.name`.

## Procedure
All `aws`/`cdk` commands run in the toolbox: `docker compose up -d ena_infra`, then `docker compose exec ena_infra bash` (or prefix each command with `docker compose exec ena_infra`).

| Step | Command | Notes |
|---|---|---|
| Bootstrap (once per account/region) | `npm run cdk -- bootstrap` | |
| Deploy | `npm run cdk -- deploy ena-experiment` | creates ECR, S3, network, role, launch template |
| Push the image (host) | `infra/scripts/push-image.sh` | builds `hub/cloud/Dockerfile` for arm64, tags it with the commit; refuses uncommitted image inputs unless `--allow-dirty` |
| Reproduction check | `bash /home/scripts/launch.sh 1` | replicate 1 is the local validation replicate: after ~25 min (ENA done), `exp_1/live_training_ENA-Combined.csv` in S3 must equal `app/half_cheetah_validation_outputs/exp_1/`'s |
| Launch the rest | `bash /home/scripts/launch.sh 2-20` | one machine each, tagged `Name=exp_k`, `Project=ena-experiment`; running or complete replicates are skipped |
| Follow | `bash /home/scripts/status.sh` | machines, and each replicate's last status line (`run.log`) |
| Download | `bash /home/scripts/download.sh` | into `app/half_cheetah_outputs/`, then `python src/tools/paper_results.py` in the experiment container |
| Terminate stragglers | `bash /home/scripts/terminate.sh` | before destroy |
| Destroy everything | `npm run cdk -- destroy ena-experiment` | deletes the bucket (and its objects) and the repository |

A replicate that failed or hit the cap can be relaunched with `launch.sh k`: the runner restarts it from scratch (the same seeds). A machine can be inspected through Session Manager (no SSH key, no inbound rule); its boot log is copied to `exp_k/boot.log`, and `exp_k/image.txt` records the image digest and commit it ran.

## Cost (approximate)
20 machines × about 11 h on demand ≈ $65, plus negligible S3, ECR and public-IP charges; no NAT gateway.
