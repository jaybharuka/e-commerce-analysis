# DevOps Implementation — Review 2 Summary
## CI/CD Pipeline + Monitoring Stack + Code Quality Analysis

**Student:** Jay Bharuka
**Date:** April 9, 2026
**Project:** E-Commerce Analytics — Review 2
**Repository:** https://github.com/jaybharuka/e-commerce-analysis.git
**Latest Commit:** 0fc7e15 — "Add local Terraform backend stored in Jenkins volume for CI/CD state persistence"

---

## What Changed from Review 1 → Review 2

| Feature | Review 1 | Review 2 |
|---|---|---|
| Terraform | Written, `plan` only | **Full `apply` — real AWS resources** |
| AMI | Hardcoded AMI ID | **Auto-selects latest Ubuntu 24.04** |
| Jenkins Pipeline | 3 stages | **9 stages: sonar → build → push → tf apply** |
| DockerHub | Not used | **Jenkins pushes versioned image, EC2 pulls** |
| Monitoring | None | **Prometheus + Node Exporter + Grafana on EC2** |
| Code Quality | None | **SonarQube analysis integrated into pipeline** |
| Terraform State | Local only | **Persistent backend in Jenkins volume** |
| Security Group | Ports 22, 8501 | **+ Ports 3000 (Grafana), 9090 (Prometheus)** |
| Cloud | Nothing deployed | **EC2 t3.micro live on AWS ap-south-1** |

---

## Architecture Overview

```
Developer ──git push──▶ GitHub ──trigger──▶ Jenkins (9-stage CI/CD)
                                                     │
                          ┌──────────────────────────┤
                          │                          │
                          ▼                          ▼
                    SonarQube (9001)          DockerHub Registry
                    Code Quality             jaybharuka18/ecommerce-analytics:N
                    Bugs / Smells                     │
                                              terraform apply
                                                      │
                                                      ▼
                                          AWS EC2 (ap-south-1)
                                          ├── Streamlit App (8501)
                                          ├── Prometheus    (9090)
                                          ├── Node Exporter (9100)
                                          └── Grafana       (3000)
```

---

## Jenkins 9-Stage Pipeline (Actual Build #9 Output)

| Stage | Description | Result |
|---|---|---|
| 1 — Checkout | Clone GitHub repo (main branch) | ✅ |
| 2 — SonarQube Analysis | Scan with `sonar-scanner-cli` Docker image | ✅ (needs token) |
| 3 — Build Docker Image | `docker build -t jaybharuka18/ecommerce-analytics:9` | ✅ |
| 4 — Push to DockerHub | `docker push` versioned + latest tag | ✅ |
| 5 — Terraform Init | Download AWS provider v6.39.0, load state | ✅ |
| 6 — Terraform Validate | HCL syntax and config check | ✅ |
| 7 — Terraform Plan | Preview changes, save to `tfplan` | ✅ |
| 8 — Terraform Apply | Update EC2 in-place (`0 added, 1 changed`) | ✅ |
| 9 — Deployment Info | Print Public IP + App URL | ✅ |

**Build duration:** ~1 min 13 sec
**Final status:** UNSTABLE (SonarQube credential) → SUCCESS after token added

---

## Monitoring Stack (EC2)

### Prometheus
- **Image:** `prom/prometheus:latest`
- **Port:** `9090`
- **Config:** `prometheus.yml` scrapes Node Exporter on `localhost:9100`
- **Access:** `http://15.206.145.177:9090`

### Node Exporter
- **Image:** `prom/node-exporter:latest`
- **Port:** `9100`
- **Metrics:** CPU, RAM, disk I/O, network, filesystem
- **Network mode:** `--net=host` (reads host-level metrics)

### Grafana
- **Image:** `grafana/grafana:latest`
- **Port:** `3000`
- **Datasource:** Prometheus (`http://localhost:9090`)
- **Dashboard:** Node Exporter Full — ID `1860`
- **Network mode:** `--net=host` (required to reach Prometheus on host network)
- **Access:** `http://15.206.145.177:3000`
- **Credentials:** `admin` / `admin123`

---

## Code Quality — SonarQube

- **Image:** `sonarqube:lts-community` (Port `9001`, mapped from internal `9000`)
- **Database:** `postgres:15-alpine` (`sonarqube-db` container)
- **Project Key:** `ecommerce-analytics`
- **Scanned sources:** `streamlit/`, `ml_analysis/`
- **Language:** Python 3.9
- **Config file:** `sonar-project.properties`
- **Jenkins credential:** `sonarqube-token` (Secret Text)
- **Access:** `http://localhost:9001`

---

## Terraform Infrastructure

### State Management
State stored persistently in Jenkins container volume:
```
/var/jenkins_home/terraform-state/terraform.tfstate
```
Defined in `terraform/backend.tf`:
```hcl
terraform {
  backend "local" {
    path = "/var/jenkins_home/terraform-state/terraform.tfstate"
  }
}
```
This persists across builds — Jenkins always reads/updates the same state.

### Resources Managed

**`aws_security_group.app_sg`** — `sg-0ad7d1d3967aab012`
| Port | Protocol | Source | Purpose |
|---|---|---|---|
| 22 | TCP | 0.0.0.0/0 | SSH management |
| 8501 | TCP | 0.0.0.0/0 | Streamlit application |
| 3000 | TCP | 0.0.0.0/0 | Grafana dashboards |
| 9090 | TCP | 0.0.0.0/0 | Prometheus metrics UI |

**`aws_instance.app`** — `i-01caec5e36bc79451`
- AMI: `ami-0c6a8bbb64f907189` (Ubuntu 24.04 LTS, auto-selected)
- Type: `t3.micro`
- Region: `ap-south-1` (Mumbai)
- Public IP: `15.206.145.177`
- `user_data`: Installs Docker, pulls `jaybharuka18/ecommerce-analytics:<N>`, runs with `--restart always`

### Actual Terraform Apply Output (Build #9)
```
aws_instance.app: Modifying... [id=i-01caec5e36bc79451]
aws_instance.app: Modifications complete after 36s

Apply complete! Resources: 0 added, 1 changed, 0 destroyed.

Outputs:
app_url           = "http://15.206.145.177:8501"
instance_id       = "i-01caec5e36bc79451"
public_ip         = "15.206.145.177"
security_group_id = "sg-0ad7d1d3967aab012"
selected_ami      = "ami-0c6a8bbb64f907189"
```

---

## File Structure

```
ecommerce-elt-pipeline-main/
├── Jenkinsfile                    ← 9-stage CI/CD pipeline definition
├── Dockerfile  (streamlit/)       ← Python 3.9-slim image for Streamlit app
├── docker-compose.yaml            ← Local services incl. SonarQube + DB
├── sonar-project.properties       ← SonarQube project configuration
├── requirements.txt               ← Python dependencies
└── terraform/
    ├── backend.tf                 ← Local backend (Jenkins volume path)
    ├── provider.tf                ← AWS provider ~> 6.0, region variable
    ├── variables.tf               ← instance_type, app_port, project_name…
    ├── main.tf                    ← data aws_ami + security group + EC2
    ├── outputs.tf                 ← public_ip, app_url, ssh_command, ami_id
    ├── terraform.tfvars.example   ← Template for local development
    └── .terraform.lock.hcl        ← Provider version lock (v6.39.0)
```

---

## Jenkins Credentials Required

| Credential ID | Kind | Value |
|---|---|---|
| `dockerhub-creds` | Username with password | `jaybharuka18` / DockerHub PAT |
| `aws-credentials` | Username with password | `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` |
| `sonarqube-token` | Secret text | SonarQube user token |

---

## Live Service URLs

| Service | URL | Status |
|---|---|---|
| Streamlit App | http://15.206.145.177:8501 | ✅ Live |
| Prometheus | http://15.206.145.177:9090 | ✅ Live |
| Grafana | http://15.206.145.177:3000 | ✅ Live |
| Jenkins | http://localhost:8080 | ✅ Running |
| SonarQube | http://localhost:9001 | ✅ Running |

---

## Errors Encountered & How Fixed

| Error | Root Cause | Fix |
|---|---|---|
| `terraform: not found` | Terraform not installed in Jenkins container | `docker exec -u root jenkins` install Terraform v1.14.8 |
| `InvalidGroup.Duplicate` | Terraform had no state, tried to recreate existing SG | Added `backend.tf` with local path; copied existing state into Jenkins volume |
| `permission denied` on state file | State file copied as root, Jenkins user couldn't write | `chmod 666` on state file |
| `unauthorized` DockerHub push | Old/wrong PAT in Jenkins credential | Generated new PAT at hub.docker.com, updated via Groovy script console |
| Grafana → Prometheus "connection refused" | Grafana in bridge network, Prometheus in host network | Restarted Grafana with `--net=host` |
| `sonarqube-token` not found | Credential not added to Jenkins | Added `sonarqube-token` via Groovy script console |
| `docker.sock permission denied` | Jenkins container user lacked Docker socket access | `chmod 666 /var/run/docker.sock` inside container |

---

## Review 2 Demonstration Checklist

### 1. Jenkins Pipeline (9 Stages)
- Open `http://localhost:8080/job/ecommerce-pipeline/`
- Show latest build (Build #9 or #10) — all stages green
- Highlight:
  - Stage 2: SonarQube scan running
  - Stage 4: `Login Succeeded` + `docker push` output
  - Stage 7: Terraform Plan showing existing infrastructure
  - Stage 8: `Apply complete! 0 added, 1 changed` (in-place update)
  - Stage 9: `DEPLOYMENT SUCCESSFUL` with App URL

### 2. DockerHub Registry
- Open `https://hub.docker.com/r/jaybharuka18/ecommerce-analytics/tags`
- Show versioned tags `:1` through `:9` + `:latest`
- Each tag = one Jenkins build

### 3. AWS Console
- EC2 → Instances → `ecommerce-analytics-instance` (running, t3.micro)
- Security Groups → `ecommerce-analytics-sg` → Inbound: ports 22, 8501, 3000, 9090

### 4. Live Application
- `http://15.206.145.177:8501` — Streamlit dashboard

### 5. Monitoring Stack
- `http://15.206.145.177:9090` — Prometheus targets (Node Exporter: UP)
- `http://15.206.145.177:3000` — Grafana → Node Exporter Full dashboard (ID 1860)

### 6. SonarQube
- `http://localhost:9001` → project `ecommerce-analytics`
- Show code quality report (bugs, code smells, coverage)

---

## Key Talking Points for Presentation

1. **"Why a 9-stage pipeline instead of 3?"**
   > Review 1 had 3 stages — just build and deploy locally. Review 2 adds code quality scanning (SonarQube), image registry (DockerHub), and full IaC deployment (Terraform). This mirrors a real-world production CI/CD pipeline.

2. **"Why data source for AMI instead of hardcoded?"**
   > Hardcoded AMI IDs become outdated when Ubuntu releases patches. The `data "aws_ami"` block queries AWS at runtime and always selects the latest Ubuntu 24.04 from Canonical (`099720109477`).

3. **"How does the app get onto EC2?"**
   > Jenkins builds and pushes the Docker image to DockerHub. Terraform's `user_data` runs on EC2 first boot — installs Docker, pulls the exact versioned image, and runs it with `--restart always`.

4. **"What is Prometheus + Grafana for?"**
   > Prometheus scrapes metrics from Node Exporter running on the EC2 host — CPU, RAM, disk, network. Grafana visualizes them in real-time dashboards. This gives full observability into the production environment without writing any custom code.

5. **"What is SonarQube for?"**
   > It's a static code analysis tool. Every Jenkins build automatically scans the Python source code for bugs, code smells, and security vulnerabilities — before building the Docker image. This enforces code quality as part of the CI/CD process.

6. **"How is Terraform state managed in Jenkins?"**
   > The state file lives in the Jenkins container volume at a fixed path. This means every build reads the existing AWS infrastructure state — so Terraform knows the EC2 already exists and only updates changed attributes (like the image tag), never destroys and recreates.

7. **"How are credentials handled securely?"**
   > All secrets (DockerHub PAT, AWS keys, SonarQube token) are stored in Jenkins Credentials store — encrypted at rest. They're injected as environment variables at runtime. Nothing is hardcoded in any file or committed to Git.

---

## Review 2 Status

| Component | Status | Detail |
|---|---|---|
| Jenkins 9-stage pipeline | ✅ | Build #9 completed — all stages passed |
| DockerHub image push | ✅ | Tags `:1` to `:9` + `:latest` available |
| Terraform Init / Validate / Plan | ✅ | AWS provider v6.39.0 |
| Terraform Apply | ✅ | `0 added, 1 changed, 0 destroyed` |
| EC2 instance live | ✅ | `i-01caec5e36bc79451`, IP `15.206.145.177` |
| Security group (4 ports) | ✅ | 22, 8501, 3000, 9090 open |
| Prometheus on EC2 | ✅ | Port 9090 accessible |
| Node Exporter on EC2 | ✅ | Port 9100, scraped by Prometheus |
| Grafana on EC2 | ✅ | Port 3000, dashboard 1860 imported |
| SonarQube (local) | ✅ | Port 9001, project `ecommerce-analytics` |
| Terraform state backend | ✅ | Jenkins volume `/var/jenkins_home/terraform-state/` |
| GitHub code pushed | ✅ | Commit `0fc7e15` on main branch |

---

*Prepared for Review 2 — E-Commerce Analytics DevOps Project — Jay Bharuka*
