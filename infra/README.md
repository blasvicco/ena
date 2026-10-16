# Cloud run of the HalfCheetah study (AWS CDK)

The study (`app/src/half_cheetah.py`) runs once in AWS: **one replicate per on-demand `c8g.xlarge`** (4 vCPU Graviton4, ARM like the local runs; every replicate on the same type, so the agents' wall-clock times are comparable; chosen by the timing probes: about 7.8 h and $1.3 per replicate), 20 replicates, region `eu-west-1` (Ireland; `eu-south-2` had no `c7g.2xlarge` capacity for the account). Each machine pulls the study image from ECR, runs its replicate (`--replicate k`), keeps `s3://<bucket>/exp_k/` (`ena-experiment-w1`) current every 15 minutes, and powers off when it ends or at the hard cap (40 h; a replicate takes about 8 h on `c8g.xlarge`), which terminates it. Everything here is destroyed with the stack, the bucket's objects included, so **download the results first**.

The CDK code follows the conventions of `cb-infra`: TypeScript run by ts-node, a Zod-validated QoS file, `S*` stacks, `C*` constructs, construct ids `${stack}-<svc>`.

```
src/                     CDK project (mounted at /home/app in the toolbox)
  index.ts               entry point: validates qos/qos.json, builds SExperiment
  stack/experiment.ts    SExperiment: ECR, S3, network, IAM, launch template
  construct/             ecr.ts s3.ts net.ts iam.ts ec2.ts, user-data.sh (each machine's boot script)
  qos/                   schema.ts, qos.template.json (copy to qos.json; git-ignored)
scripts/                 push-image.sh (host), launch.sh status.sh download.sh terminate.sh probe.sh probe_queue.sh probe_report.sh (toolbox), common.sh
```

## Before the first deploy
1. An AWS profile in `~/.aws` (the toolbox mounts it read-only; defaults to `personal` in `docker-compose.yml`, so set `export AWS_PROFILE=<profile>` if using `default` or another profile).
2. A quota of **80–160 vCPUs** for "Running On-Demand Standard (A, C, D, H, I, M, R, T, Z) instances" (code `L-1216C47A`) in the stack's region (Service Quotas → Amazon EC2; 20 machines × 4 vCPUs = 80 vCPUs minimum to run all replicates concurrently).
3. `cp src/qos/qos.template.json src/qos/qos.json`, then set `aws.account` (or leave it empty to use the profile's account).
4. The bucket name (`construct.s3.name`) must be free globally. A name deleted in one region cannot be re-created in another for up to about an hour (S3 answers 409, "conflicting conditional operation"); pick another name rather than wait. The scripts read the stack's name and region from `qos.json`.

## Procedure
All `aws`/`cdk` commands run in the toolbox: `docker compose up -d ena_infra`, then `docker compose exec ena_infra bash` (or prefix each command with `docker compose exec ena_infra`).

| Step | Command | Notes |
|---|---|---|
| Bootstrap (once per account/region) | `npm run cdk -- bootstrap` | |
| Deploy | `npm run cdk -- deploy ena-experiment` | creates ECR, S3, network, role, launch template |
| Push the image (host) | `infra/scripts/push-image.sh` | builds `hub/cloud/Dockerfile` for arm64, tags it with the commit; refuses uncommitted image inputs unless `--allow-dirty` |
| Launch | `bash /home/scripts/launch.sh 1-20` | one machine each, tagged `Name=exp_k`, `Project=ena-experiment`; running or complete replicates are skipped |
| Follow | `bash /home/scripts/status.sh` | machines, and each replicate's last status line (`run.log`) |
| Download | `bash /home/scripts/download.sh` | into `app/half_cheetah_outputs/`, then `python src/tools/paper_results.py` in the experiment container |
| Terminate stragglers | `bash /home/scripts/terminate.sh` | before destroy |
| Destroy everything | `npm run cdk -- destroy ena-experiment` | deletes the bucket (and its objects) and the repository |

## Timing probes (plan the batch before paying for it)
A probe trains one agent alone (the study's replicate 1, same seeds) on a chosen instance type for a few minutes and terminates itself; its results stay under `s3://<bucket>/probes/`, apart from the study's `exp_k/`. Every machine, probe or study, records its pace each minute in `progress.csv` (time, agent, env-step reached).
```
for T in c7g.xlarge c7g.2xlarge c8g.xlarge c8g.2xlarge; do
  bash /home/scripts/probe.sh $T ENA-Combined 90   # ENA trains fully: its measured time
  bash /home/scripts/probe.sh $T PPO 45            # fitted rate x 3M steps
  bash /home/scripts/probe.sh $T SAC 60            # fitted rate past its 10k random steps x 3M steps
done
bash /home/scripts/probe_report.sh                 # per agent: steps/s and phase hours; per type: replicate hours and 20 x cost
aws s3 rm s3://<bucket>/probes --recursive --region <region>   # when done
```

A replicate that failed or hit the cap can be relaunched with `launch.sh k`: the runner restarts it from scratch (the same seeds). A machine can be inspected through Session Manager (no SSH key, no inbound rule); its boot log is copied to `exp_k/boot.log`, and `exp_k/image.txt` records the image digest and commit it ran.

## Cost and Duration
20 machines × about 7.8 h × $0.171 ≈ $27 on demand (measured by the timing probes), plus negligible S3, ECR and public-IP charges; no NAT gateway. At a 16-vCPU quota, 4 machines run at once (5 waves, about 40 h); 80 vCPUs run all 20 at once.

In the completed confirmation study across all 20 replicates, average wall-clock training times on `c8g.xlarge` were:
- **ENA-Combined:** 32.0 min (candidate evaluation parallelized across 2 threads)
- **PPO:** 115.3 min
- **SAC:** 281.1 min (total runtime: ~7.1 h per replicate machine)
