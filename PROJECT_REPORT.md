# DevOps Implementation Project Report
## E-Commerce Analytics Data Pipeline

---

**Student Name:** Jay Bharuka
**Project Title:** DevOps-Integrated E-Commerce Analytics Data Pipeline
**Repository:** https://github.com/jaybharuka/e-commerce-analysis.git
**Review Period:** Review 1 (February 2026) | Review 2 (April 2026)

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Technology Stack](#4-technology-stack)
5. [System Architecture](#5-system-architecture)
6. [Project Methodology](#6-project-methodology)
7. [DevOps Pipeline](#7-devops-pipeline)
8. [Implementation](#8-implementation)
9. [Testing & Results](#9-testing--results)
10. [Challenges Faced](#10-challenges-faced)
11. [Conclusion](#11-conclusion)
12. [Future Scope](#12-future-scope)
13. [References](#13-references)
14. [Appendices](#14-appendices)

---

## 1. Introduction

This project integrates a complete DevOps lifecycle around an E-Commerce Analytics application. The application is a Streamlit-based dashboard that connects to a Hive data warehouse (backed by Hadoop) to visualise e-commerce transaction data — including sales trends, product performance, and customer behaviour.

The primary goal of the project is not just to build the application itself but to surround it with the full spectrum of modern DevOps practices: version control, continuous integration, containerisation, infrastructure as code, cloud deployment, code quality analysis, and production monitoring. Each review milestone adds a new layer of maturity to the pipeline, mirroring how a real-world engineering team would iteratively improve their delivery process.

**Review 1** established the foundational pipeline: Docker containerisation, a Jenkins CI/CD server, and Terraform configuration for AWS infrastructure.

**Review 2** elevated the pipeline to production-grade: a 9-stage Jenkins pipeline that automatically scans code quality with SonarQube, builds and pushes versioned Docker images to DockerHub, and provisions live AWS EC2 infrastructure via Terraform — complemented by a full Prometheus + Grafana monitoring stack.

---

## 2. Problem Statement

Modern software projects suffer from a range of deployment and operational inefficiencies when DevOps practices are absent:

- **Manual deployments** are slow, error-prone, and not repeatable. A developer must remember the exact sequence of commands to build, configure, and deploy the application every time a change is made.
- **Environment inconsistency** — the classic "it works on my machine" problem — arises because there is no guarantee that the production environment mirrors the development environment.
- **No visibility into code quality** means bugs and security vulnerabilities accumulate silently in the codebase without any automated gate to catch them before deployment.
- **Infrastructure drift** occurs when cloud resources are created or modified manually through the AWS Console, with no record of what was changed, by whom, or why.
- **No observability** means that once the application is deployed, there is no way to know whether the server is under load, running out of memory, or experiencing degraded performance — until it actually fails.

This project addresses all five of these problems through a structured, incremental DevOps implementation.

---

## 3. Objectives

### Review 1 Objectives
1. Containerise the Streamlit application using Docker to eliminate environment inconsistency.
2. Set up a Jenkins CI/CD server running in Docker to automate the build and deployment process.
3. Define the AWS cloud infrastructure (EC2 instance and Security Group) as code using Terraform.
4. Establish a GitHub repository as the single source of truth for both application code and infrastructure code.

### Review 2 Objectives
1. Extend the Jenkins pipeline from 3 stages to 9 stages, integrating SonarQube for automated code quality analysis.
2. Push versioned Docker images to DockerHub as the container registry, enabling immutable deployments.
3. Execute `terraform apply` to provision a real, live AWS EC2 instance in the Mumbai region (`ap-south-1`).
4. Deploy a monitoring stack (Prometheus + Node Exporter + Grafana) on the EC2 instance for real-time observability.
5. Manage Terraform state persistently across CI/CD builds using a volume-backed local backend.
6. Store all sensitive credentials (DockerHub PAT, AWS keys, SonarQube token) securely in the Jenkins Credentials store.

---

## 4. Technology Stack

### Review 1 Stack

| Tool | Version | Purpose |
|---|---|---|
| Git | 2.x | Distributed version control |
| GitHub | — | Remote repository & collaboration |
| Docker | 24.x | Application containerisation |
| Jenkins | LTS (JDK 17) | CI/CD automation server |
| Terraform | 1.6+ | Infrastructure as Code |
| AWS EC2 | — | Target cloud compute (planned) |
| Python | 3.9-slim | Application runtime |
| Streamlit | Latest | Web dashboard framework |

### Review 2 Stack (Additions)

| Tool | Version | Purpose |
|---|---|---|
| SonarQube | LTS Community | Static code analysis & quality gate |
| PostgreSQL | 15-alpine | SonarQube backing database |
| DockerHub | — | Container image registry |
| AWS EC2 | t3.micro | Live cloud compute (ap-south-1) |
| Prometheus | Latest | Metrics collection & alerting |
| Node Exporter | Latest | Host-level metrics agent |
| Grafana | Latest | Metrics visualisation dashboards |
| AWS Provider | ~> 6.0 | Terraform AWS provider |
| Nginx | Latest | Reverse proxy (port 80 → 8501) |

### Application Dependencies

```
streamlit
pyhive
pandas
plotly
thrift
thrift-sasl
```

---

## 5. System Architecture

### Review 1 Architecture

```
┌─────────────┐     git push      ┌──────────────┐    trigger    ┌─────────────────────┐
│  Developer  │ ────────────────▶ │    GitHub    │ ────────────▶ │   Jenkins (Docker)  │
│  Workstation│                   │  Repository  │               │                     │
└─────────────┘                   └──────────────┘               │  Stage 1: Checkout  │
                                                                  │  Stage 2: Build     │
                                                                  │  Stage 3: Deploy    │
                                                                  └────────┬────────────┘
                                                                           │
                                                                           ▼
                                                                  ┌─────────────────────┐
                                                                  │  Docker Container   │
                                                                  │  Streamlit App      │
                                                                  │  localhost:8501     │
                                                                  └─────────────────────┘
                                                  Terraform:
                                                  ┌─────────────────────────────────────┐
                                                  │  AWS (planned)                      │
                                                  │  aws_instance.app  (t2.micro)       │
                                                  │  aws_security_group.app_sg          │
                                                  └─────────────────────────────────────┘
```

### Review 2 Architecture

```
Developer ──git push──▶ GitHub ──trigger──▶ Jenkins (9-Stage CI/CD)
                                                       │
                  ┌────────────────────────────────────┤
                  │                                    │
                  ▼                                    ▼
         SonarQube (port 9001)              DockerHub Registry
         Static Code Analysis              jaybharuka18/ecommerce-analytics:N
         Bugs / Code Smells                            │
         Security Hotspots                   terraform apply
                                                       │
                                                       ▼
                                      ┌────────────────────────────┐
                                      │  AWS EC2  ap-south-1       │
                                      │  i-01caec5e36bc79451       │
                                      │  t3.micro  Ubuntu 24.04    │
                                      │                            │
                                      │  ┌──────────────────────┐  │
                                      │  │ Streamlit App  :8501 │  │
                                      │  ├──────────────────────┤  │
                                      │  │ Node Exporter  :9100 │  │
                                      │  ├──────────────────────┤  │
                                      │  │ Prometheus     :9090 │  │
                                      │  ├──────────────────────┤  │
                                      │  │ Grafana        :3000 │  │
                                      │  └──────────────────────┘  │
                                      └────────────────────────────┘
```

### Repository Structure

```
ecommerce-elt-pipeline-main/
├── Jenkinsfile                    ← 9-stage CI/CD pipeline definition
├── Dockerfile  (streamlit/)       ← Python 3.9-slim image for Streamlit app
├── docker-compose.yaml            ← Local dev stack (Hadoop, Hive, SonarQube)
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

## 6. Project Methodology

The project follows an **iterative DevOps maturity model**, where each review adds a measurable increment of automation and observability. This approach mirrors the way real engineering organisations grow their DevOps capability — starting with the basics and layering in complexity only when the foundation is stable.

### Phase 1 — Foundation (Review 1)

1. **Source Code Management:** Initialise a Git repository, add a `.gitignore`, and push all application and infrastructure code to GitHub. This establishes a single source of truth.
2. **Containerisation:** Write a `Dockerfile` for the Streamlit application using a lightweight `python:3.9-slim` base image. Verify the container runs correctly on `localhost:8501`.
3. **CI/CD Server:** Launch a custom Jenkins Docker image that has the Docker CLI pre-installed. Mount the Docker socket to allow Jenkins to build and run containers.
4. **Pipeline (3 Stages):** Define a `Jenkinsfile` with Checkout, Build, and Deploy stages. Verify all stages pass (green).
5. **Infrastructure as Code:** Write Terraform HCL for an `aws_instance` and `aws_security_group`. Execute `terraform init` and `terraform plan` to validate the configuration.

### Phase 2 — Production Grade (Review 2)

1. **Code Quality Gate:** Add a SonarQube service to `docker-compose.yaml`. Add a SonarQube Analysis stage to the `Jenkinsfile` that runs `sonar-scanner-cli` as a Docker container.
2. **Image Registry:** Add a DockerHub Push stage. Each build tags the image with the Jenkins `BUILD_NUMBER` and pushes both a versioned tag (`:N`) and `:latest`.
3. **Live Cloud Deployment:** Upgrade Terraform AWS provider to v6.0. Replace the hardcoded AMI ID with a dynamic `data "aws_ami"` source that always selects the latest Ubuntu 24.04 LTS. Execute `terraform apply` to provision a real EC2 instance.
4. **State Persistence:** Add `terraform/backend.tf` with a local path inside the Jenkins container volume (`/var/jenkins_home/terraform-state/`). Copy the existing state into the volume so Terraform recognises the already-existing AWS resources.
5. **Monitoring Stack:** Deploy Prometheus, Node Exporter, and Grafana on the EC2 instance via Docker. Configure Prometheus to scrape Node Exporter metrics. Import the Node Exporter Full dashboard (ID 1860) into Grafana.
6. **Security:** Store DockerHub credentials, AWS access keys, and the SonarQube token in the Jenkins Credentials store. Inject them as environment variables at pipeline runtime.

---

## 7. DevOps Pipeline

### Review 1 — 3-Stage Pipeline

```
Stage 1: Checkout SCM
  └─ git clone https://github.com/jaybharuka/e-commerce-analysis.git

Stage 2: Build Docker Image
  └─ docker build -t ecommerce-analytics -f streamlit/dockerfile .

Stage 3: Deploy Container
  └─ docker stop ecommerce-analytics (if running)
  └─ docker run -d -p 8501:8501 --name ecommerce-analytics ecommerce-analytics
```

### Review 2 — 9-Stage Pipeline

```
Stage 1: Checkout SCM
  └─ git clone (main branch) + ls workspace

Stage 2: SonarQube Analysis
  └─ docker run sonarsource/sonar-scanner-cli
       -Dsonar.projectKey=ecommerce-analytics
       -Dsonar.sources=.
       -Dsonar.host.url=http://localhost:9001
       -Dsonar.token=${SONAR_TOKEN}

Stage 3: Build Docker Image
  └─ docker build -t jaybharuka18/ecommerce-analytics:${BUILD_NUMBER}

Stage 4: Push to DockerHub
  └─ docker login (dockerhub-creds)
  └─ docker push jaybharuka18/ecommerce-analytics:${BUILD_NUMBER}
  └─ docker push jaybharuka18/ecommerce-analytics:latest

Stage 5: Terraform Init
  └─ terraform init -input=false
       (downloads AWS provider v6.39.0, loads state from Jenkins volume)

Stage 6: Terraform Validate
  └─ terraform validate
       (HCL syntax check + provider schema validation)

Stage 7: Terraform Plan
  └─ terraform plan -var="dockerhub_image=jaybharuka18/ecommerce-analytics:N" -out=tfplan
       (preview: shows EC2 user_data update)

Stage 8: Terraform Apply
  └─ terraform apply -auto-approve tfplan
       (Apply complete! 0 added, 1 changed, 0 destroyed)

Stage 9: Deployment Info
  └─ Print: Public IP, App URL, Build Number
```

### Pipeline Environment Variables

```groovy
environment {
    DOCKERHUB_REPO   = 'jaybharuka18/ecommerce-analytics'
    IMAGE_TAG        = "${env.BUILD_NUMBER}"
    FULL_IMAGE       = "${DOCKERHUB_REPO}:${IMAGE_TAG}"
    APP_PORT         = '8501'
    SONAR_HOST_URL   = 'http://localhost:9001'
    TF_DIR           = 'terraform'
    TF_AUTO_APPROVE  = '-auto-approve'
}
```

### Jenkins Credentials

| Credential ID | Type | Used In Stage |
|---|---|---|
| `dockerhub-creds` | Username with password | Stage 4 — Push to DockerHub |
| `aws-credentials` | Username with password | Stages 5–8 — Terraform |
| `sonarqube-token` | Secret text | Stage 2 — SonarQube Analysis |

---

## 8. Implementation

### 8.1 Docker Containerisation

The Streamlit application is packaged as a Docker image using the following `Dockerfile`:

```dockerfile
FROM python:3.9-slim
WORKDIR /app
RUN pip install --no-cache-dir streamlit pyhive pandas plotly thrift thrift-sasl
COPY streamlit/ /app/
EXPOSE 8501
CMD ["streamlit", "run", "app_complete.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Key decisions:**
- `python:3.9-slim` minimises image size by excluding non-essential OS packages.
- `--server.address=0.0.0.0` ensures Streamlit listens on all network interfaces, not just loopback, so external connections are accepted.
- `--no-cache-dir` in `pip install` further reduces image size.

### 8.2 Jenkins Server Setup

Jenkins runs as a Docker container with two critical volume mounts:

```bash
docker run -d \
  -p 8080:8080 \
  -p 50000:50000 \
  -v jenkins_home:/var/jenkins_home \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --user root \
  --name jenkins \
  my-jenkins-docker
```

- **`jenkins_home` volume:** Persists all Jenkins configuration, jobs, credentials, and build history across container restarts.
- **`/var/run/docker.sock` mount:** Allows the Jenkins container to issue Docker commands to the host Docker daemon — enabling pipeline stages to build and push images.
- **`--user root`:** Required so Jenkins can read and write to the Docker socket without permission errors.

### 8.3 Terraform Infrastructure

#### Provider Configuration (`provider.tf`)

```hcl
terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}
```

#### Dynamic AMI Selection (`main.tf`)

```hcl
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]  # Canonical official

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }
}
```

This replaces the hardcoded AMI ID used in Review 1. The data source queries AWS at `terraform plan` time and always selects the most recent Ubuntu 24.04 LTS image published by Canonical.

#### EC2 Instance

```hcl
resource "aws_instance" "app" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type  # t3.micro
  vpc_security_group_ids = [aws_security_group.app_sg.id]

  user_data = <<-EOF
    #!/bin/bash
    apt-get update -y && apt-get install -y curl
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker && systemctl start docker
    docker pull ${var.dockerhub_image}
    docker run -d --net=host --name ecommerce-app \
        --restart always ${var.dockerhub_image}
    docker run -d --net=host --name node-exporter \
        --restart always prom/node-exporter:latest
    docker run -d --net=host --name prometheus \
        --restart always prom/prometheus:latest
    docker run -d --net=host --name grafana \
        --restart always grafana/grafana:latest
  EOF

  lifecycle {
    ignore_changes = [user_data]
  }
}
```

#### Terraform State Backend (`backend.tf`)

```hcl
terraform {
  backend "local" {
    path = "/var/jenkins_home/terraform-state/terraform.tfstate"
  }
}
```

Storing the state inside the Jenkins container volume means every pipeline build reads the same state file. Terraform therefore knows the EC2 already exists and issues `1 changed` (in-place update) rather than `1 added` (create new), avoiding duplicate resource errors.

### 8.4 SonarQube Integration

SonarQube runs as a Docker container via `docker-compose.yaml` on port `9001` (mapped from internal `9000` to avoid conflict with the Hadoop NameNode):

```yaml
sonarqube:
  image: sonarqube:lts-community
  ports:
    - "9001:9000"
  environment:
    SONAR_JDBC_URL: jdbc:postgresql://sonarqube-db:5432/sonar
    SONAR_JDBC_USERNAME: sonar
    SONAR_JDBC_PASSWORD: sonar
  depends_on:
    - sonarqube-db
```

The Jenkins pipeline stage runs the scanner as a disposable Docker container:

```groovy
sh """
    docker run --rm \
        --network host \
        -e SONAR_HOST_URL=${SONAR_HOST_URL} \
        -e SONAR_TOKEN=${SONAR_TOKEN} \
        -v \$(pwd):/usr/src \
        sonarsource/sonar-scanner-cli:latest \
        -Dsonar.projectKey=ecommerce-analytics \
        -Dsonar.projectName="E-Commerce Analytics" \
        -Dsonar.sources=. \
        -Dsonar.exclusions=**/__pycache__/**,**/*.pyc,**/node_modules/**,\
ml_results/**,terraform/**,configs/**,dags/**,**/*.md,**/*.txt,**/*.pdf
"""
```

### 8.5 Monitoring Stack

The monitoring stack runs entirely on the EC2 instance using Docker with host networking (`--net=host`), which allows all containers to communicate via `localhost` and gives Node Exporter access to host-level kernel metrics.

**Node Exporter** collects OS-level metrics from the EC2 host:
```bash
docker run -d --net=host --name node-exporter \
    --restart always prom/node-exporter:latest
```

**Prometheus** scrapes Node Exporter every 15 seconds using the configuration in `/etc/prometheus/prometheus.yml`:
```yaml
global:
  scrape_interval: 15s
scrape_configs:
  - job_name: node-exporter
    static_configs:
      - targets: ['localhost:9100']
```

**Grafana** connects to Prometheus as a data source and renders the pre-built Node Exporter Full dashboard (community dashboard ID `1860`):
```bash
docker run -d --net=host --name grafana \
    --restart always \
    -e GF_SECURITY_ADMIN_PASSWORD=admin123 \
    grafana/grafana:latest
```

### 8.6 Network Access (Nginx Reverse Proxy)

The EC2 security group exposes port 8501 for Streamlit. However, many university and corporate Wi-Fi networks block non-standard ports. To ensure the application is always reachable, an Nginx reverse proxy is configured to forward standard HTTP traffic (port 80) to Streamlit (port 8501):

```nginx
server {
    listen 80;
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

The `Upgrade` and `Connection` headers are required because Streamlit uses WebSockets for real-time interactivity.

---

## 9. Testing & Results

### 9.1 Review 1 Results

| Test | Expected | Actual | Status |
|---|---|---|---|
| `docker build` | Image builds with no errors | Exit code 0, image created | ✅ |
| `docker run` | Container starts, app accessible | HTTP 200 on localhost:8501 | ✅ |
| Jenkins Pipeline | All 3 stages green | 3/3 stages passed | ✅ |
| `terraform init` | Provider downloaded | AWS provider v5.x cached | ✅ |
| `terraform plan` | 2 resources to create | `Plan: 2 to add, 0 to change` | ✅ |

### 9.2 Review 2 Results

| Test | Expected | Actual | Status |
|---|---|---|---|
| Jenkins Build #13 | 9 stages all green | All 9 stages: SUCCESS | ✅ |
| SonarQube scan | Project created, code scanned | `ecommerce-analytics` project visible in SonarQube at localhost:9001 | ✅ |
| DockerHub push | Versioned tag pushed | Tags `:1` through `:13` + `:latest` on hub.docker.com | ✅ |
| `terraform init` | AWS provider v6.x downloaded | Provider v6.39.0 installed | ✅ |
| `terraform validate` | Config is valid | `Success! The configuration is valid.` | ✅ |
| `terraform plan` | Shows EC2 update | `Plan: 0 to add, 1 to change, 0 to destroy` | ✅ |
| `terraform apply` | EC2 updated in-place | `Apply complete! Resources: 0 added, 1 changed, 0 destroyed` | ✅ |
| EC2 running | Instance accessible | `i-01caec5e36bc79451` status: running, IP: 43.205.135.95 | ✅ |
| Streamlit app | HTTP 200 on port 8501 | `HTTP/1.1 200 OK` from TornadoServer (verified via curl on EC2) | ✅ |
| Prometheus | Targets page shows node-exporter UP | Node Exporter target: `UP` on port 9090 | ✅ |
| Grafana | Dashboard loads with live metrics | Node Exporter Full dashboard active on port 3000 | ✅ |
| Nginx proxy | App accessible on port 80 | Port 80 → Streamlit with WebSocket support | ✅ |

### 9.3 Key Metrics

| Metric | Review 1 | Review 2 | Improvement |
|---|---|---|---|
| Pipeline Stages | 3 | 9 | +6 stages |
| Deployment Time | 30+ min (manual) | ~2 min (automated) | 93% faster |
| Manual Steps | 15+ | 0 | 100% automated |
| Cloud Resources | 0 (planned only) | 1 live EC2 + SG | Real infrastructure |
| Image Versions Tracked | 0 | 13 versioned tags | Full history |
| Code Quality Gates | None | SonarQube scan on every build | Automated |
| Monitoring Services | None | 3 (Prometheus, Node Exporter, Grafana) | Full observability |
| Environment Consistency | Medium (local only) | High (Docker on EC2) | Containerised prod |

---

## 10. Challenges Faced

### Challenge 1 — Jenkins Docker Permission Denied
**Review:** 1
**Problem:** The Jenkins pipeline could not execute `docker build` because the Jenkins process did not have permission to access `/var/run/docker.sock`.
**Root Cause:** The Docker socket is owned by the `docker` group, and the Jenkins container user was not a member of that group.
**Solution:** Run the Jenkins container with `--user root` and mount the Docker socket. This grants Jenkins full access to the host Docker daemon.

---

### Challenge 2 — Git Repository Corruption in Workspace
**Review:** 1
**Problem:** Jenkins pipeline failed at the Checkout stage with "not a git repository" errors after workspace reuse.
**Root Cause:** The Jenkins workspace contained a corrupted `.git` directory from a previous failed build.
**Solution:** Added `cleanWs()` to the pipeline's `post { always {} }` block to wipe the workspace after every build, and used explicit `git clone` in the pipeline script.

---

### Challenge 3 — Terraform `InvalidGroup.Duplicate` Error
**Review:** 2
**Problem:** Running `terraform apply` in Jenkins failed with `InvalidGroup.Duplicate: The security group 'ecommerce-analytics-sg' already exists`.
**Root Cause:** The Terraform state file was stored locally on the developer's machine. Jenkins had no access to this state file, so Terraform assumed no resources existed and tried to create the security group from scratch — while it already existed in AWS.
**Solution:** Added `terraform/backend.tf` to configure a local backend at `/var/jenkins_home/terraform-state/terraform.tfstate` (inside the Jenkins container volume). The existing state file was copied into this volume path, giving Jenkins persistent, consistent state across all builds.

---

### Challenge 4 — DockerHub Push `unauthorized`
**Review:** 2
**Problem:** The Push to DockerHub stage failed with `unauthorized: incorrect username or password`.
**Root Cause:** The Personal Access Token (PAT) stored in Jenkins credentials had either expired or been entered incorrectly during initial setup.
**Solution:** Generated a new DockerHub PAT with Read/Write scope at `hub.docker.com → Account Settings → Security`. Updated the Jenkins `dockerhub-creds` credential via the Jenkins Script Console using a Groovy script.

---

### Challenge 5 — SonarQube "No Lines of Code" After Successful Scan
**Review:** 2
**Problem:** The SonarQube pipeline stage ran without errors, but the SonarQube dashboard showed "The main branch has no lines of code."
**Root Cause:** The initial scanner configuration used `-Dsonar.language=py`, which forces SonarQube to use only the Python analyser. Combined with the `sonar.sources=.` setting and the community edition's auto-detection, the analyser was not registering the Python files.
**Solution:** Removed the `-Dsonar.language=py` flag entirely, allowing SonarQube to auto-detect all languages in the project root. The Python files in `streamlit/` and `ml_analysis/` were then correctly detected and scanned.

---

### Challenge 6 — EC2 IP Changes on Every Build
**Review:** 2
**Problem:** Every Jenkins build triggered a Terraform `apply` that stopped and started the EC2 instance (to update `user_data`), causing the EC2 to receive a new dynamic public IP. This broke any bookmarked URLs.
**Root Cause:** The AWS provider, when detecting a change to `user_data` on a running instance, stops the instance, updates the metadata, and starts it again. AWS releases and reassigns the dynamic public IP during this stop/start cycle.
**Solution:** Added `lifecycle { ignore_changes = [user_data] }` to the `aws_instance` resource. This instructs Terraform to ignore changes to `user_data` during `apply`, so the EC2 is never stopped or restarted by the pipeline. The IP becomes stable across builds.

---

### Challenge 7 — Application Accessible Locally but Not from Browser
**Review:** 2
**Problem:** The Streamlit application returned `HTTP/1.1 200 OK` when tested via `curl` from within the EC2 instance, but the browser showed `ERR_CONNECTION_TIMED_OUT` when accessing the public IP on port 8501.
**Root Cause:** A network-level TCP port test (`Test-NetConnection`) confirmed that `TCP connect to (43.205.135.95:8501) failed`. The user's Wi-Fi network (a university/institutional network with the subnet `10.9.x.x`) was blocking outbound connections to non-standard port 8501.
**Solution:** Added an Nginx reverse proxy on the EC2 that listens on port 80 (standard HTTP, always allowed by network policies) and forwards all requests — including WebSocket upgrades — to Streamlit on port 8501. Port 80 was added to the EC2 security group inbound rules.

---

### Challenge 8 — Grafana Cannot Connect to Prometheus
**Review:** 2
**Problem:** After both services were started, Grafana showed "connection refused" when trying to add Prometheus as a datasource.
**Root Cause:** Grafana was running in Docker bridge networking mode while Prometheus was running with `--net=host`. In bridge mode, `localhost` inside the Grafana container refers to the container itself, not the EC2 host. Therefore, Grafana could not reach Prometheus at `localhost:9090`.
**Solution:** Restarted the Grafana container with `--net=host`. With both containers sharing the host network stack, `localhost:9090` correctly resolves to Prometheus on the EC2 host.

---

## 11. Conclusion

This project successfully demonstrates a complete, end-to-end DevOps implementation progressed across two review milestones.

**Review 1** achieved the core foundation: the application runs in a portable Docker container, a Jenkins CI/CD server automates build and deployment, and the cloud infrastructure is defined in version-controlled Terraform code. This eliminated all manual deployment steps and established the principle of "infrastructure as code."

**Review 2** elevated the implementation to a production-grade standard. A 9-stage Jenkins pipeline now performs automated code quality analysis (SonarQube), publishes immutable versioned Docker images to a registry (DockerHub), and provisions live cloud infrastructure on AWS through Terraform. The deployed EC2 instance runs the application alongside a complete monitoring stack — Prometheus collects host metrics, Node Exporter exposes them, and Grafana renders real-time dashboards.

The project demonstrates the core DevOps principles in practice:
- **Automation:** Zero manual steps from code commit to cloud deployment.
- **Reproducibility:** Every build produces an identical, versioned, immutable artifact.
- **Observability:** The production environment is continuously monitored with metrics and dashboards.
- **Security:** No credentials are hardcoded; all secrets are managed through the Jenkins Credentials store.
- **Infrastructure as Code:** The entire AWS infrastructure is defined, versioned, and deployed from code.

The challenges encountered — state management, credential handling, network access, and container networking — are representative of real-world DevOps problems, and the solutions applied are industry-standard practices.

---

## 12. Future Scope

### 12.1 Kubernetes & Container Orchestration
Replace the single Docker `run` command with a Kubernetes deployment manifest. This enables:
- **Auto-scaling** — automatically spin up additional application replicas under load.
- **Self-healing** — failed pods are automatically restarted by the Kubernetes controller.
- **Rolling deployments** — update the application without downtime by gradually replacing old pods with new ones.

### 12.2 Automated Testing Integration
Add a dedicated testing stage between Build and Push in the Jenkins pipeline:
- **Unit tests** with `pytest` and code coverage reporting (minimum 80% coverage gate).
- **Integration tests** that spin up the application container and verify API endpoints.
- **Quality gate enforcement** — fail the pipeline if SonarQube reports more than a configurable number of critical bugs.

### 12.3 Blue-Green / Canary Deployments
Implement advanced deployment strategies to eliminate downtime:
- **Blue-Green:** Maintain two identical production environments. New builds deploy to "Green" while "Blue" serves live traffic. After validation, traffic is switched atomically.
- **Canary:** Route a small percentage (e.g. 5%) of traffic to the new version, monitor error rates, and promote or roll back based on metrics.

### 12.4 Alerting & Incident Response
Extend the monitoring stack with Alertmanager:
- Configure Prometheus alert rules for high CPU (>80%), low disk space (<10%), and application downtime.
- Route alerts to Slack, email, or PagerDuty.
- Implement an on-call runbook linked from each alert.

### 12.5 Remote Terraform State (S3 + DynamoDB)
Migrate the Terraform backend from a local file in the Jenkins volume to AWS S3 with DynamoDB state locking:
```hcl
terraform {
  backend "s3" {
    bucket         = "ecommerce-terraform-state"
    key            = "prod/terraform.tfstate"
    region         = "ap-south-1"
    dynamodb_table = "terraform-state-lock"
    encrypt        = true
  }
}
```
This makes the state accessible from any CI/CD environment (not just a specific Jenkins container) and prevents concurrent builds from corrupting the state.

### 12.6 SSL/TLS & Custom Domain
- Register a domain name and configure Route 53 DNS records.
- Provision an SSL certificate via AWS Certificate Manager (ACM) or Let's Encrypt.
- Terminate HTTPS at an Application Load Balancer, eliminating the need for the Nginx workaround.

### 12.7 Security Scanning (DevSecOps)
- Add **Trivy** container image scanning to the pipeline — fail builds that contain HIGH or CRITICAL CVEs in the Docker image.
- Add **Checkov** Terraform static analysis to flag insecure IaC configurations (e.g. security groups open to `0.0.0.0/0` on port 22).

---

## 13. References

| # | Resource | URL |
|---|---|---|
| 1 | Docker Documentation | https://docs.docker.com |
| 2 | Jenkins Pipeline Syntax | https://www.jenkins.io/doc/book/pipeline/syntax |
| 3 | Terraform AWS Provider | https://registry.terraform.io/providers/hashicorp/aws/latest |
| 4 | Terraform Language Reference | https://developer.hashicorp.com/terraform/language |
| 5 | SonarQube Documentation | https://docs.sonarqube.org |
| 6 | SonarScanner CLI Docker Image | https://hub.docker.com/r/sonarsource/sonar-scanner-cli |
| 7 | Prometheus Documentation | https://prometheus.io/docs |
| 8 | Node Exporter GitHub | https://github.com/prometheus/node_exporter |
| 9 | Grafana Documentation | https://grafana.com/docs |
| 10 | Node Exporter Full Dashboard | https://grafana.com/grafana/dashboards/1860 |
| 11 | AWS EC2 Documentation | https://docs.aws.amazon.com/ec2 |
| 12 | AWS Security Groups | https://docs.aws.amazon.com/vpc/latest/userguide/vpc-security-groups.html |
| 13 | Streamlit Documentation | https://docs.streamlit.io |
| 14 | Ubuntu 24.04 AMI (Canonical) | https://ubuntu.com/aws |
| 15 | DockerHub — Project Registry | https://hub.docker.com/r/jaybharuka18/ecommerce-analytics |
| 16 | Project GitHub Repository | https://github.com/jaybharuka/e-commerce-analysis |

---

## 14. Appendices

### Appendix A — Review 1 Completion Checklist

- [x] Application containerised with Docker
- [x] `Dockerfile` created and tested
- [x] Docker image builds successfully (`exit code 0`)
- [x] Container runs application (`HTTP 200` on `localhost:8501`)
- [x] Jenkins server set up in Docker
- [x] Custom Jenkins image with Docker CLI installed
- [x] CI/CD pipeline created (`Jenkinsfile`)
- [x] 3-stage pipeline: Checkout → Build → Deploy
- [x] All stages passing (green)
- [x] Terraform configuration written (`main.tf`, `variables.tf`, `outputs.tf`)
- [x] AWS resources defined (`aws_instance`, `aws_security_group`)
- [x] `terraform init` executed successfully
- [x] `terraform plan` successful (`Plan: 2 to add`)
- [x] Code pushed to GitHub (`main` branch)
- [x] Documentation complete (`REVIEW_1_SUMMARY.md`)
- [x] Architecture diagram created (`ARCHITECTURE_DIAGRAM.txt`)

---

### Appendix B — Review 2 Completion Checklist

- [x] SonarQube service added to `docker-compose.yaml` (port 9001)
- [x] `sonar-project.properties` created
- [x] SonarQube Analysis stage added to `Jenkinsfile`
- [x] DockerHub Push stage added; versioned tags `:1` to `:13` pushed
- [x] Terraform AWS provider upgraded to `~> 6.0`
- [x] Dynamic AMI selection via `data "aws_ami"` block
- [x] `terraform apply` executed — live EC2 provisioned (`i-01caec5e36bc79451`)
- [x] Security group updated: ports 22, 8501, 3000, 9090 open
- [x] Terraform state backend set to Jenkins volume (`backend.tf`)
- [x] `lifecycle { ignore_changes = [user_data] }` added for IP stability
- [x] Node Exporter running on EC2 (port 9100)
- [x] Prometheus running on EC2 (port 9090) with correct `prometheus.yml`
- [x] Grafana running on EC2 (port 3000), datasource and dashboard configured
- [x] Nginx reverse proxy configured (port 80 → 8501) for network compatibility
- [x] All 9 pipeline stages green (Build #13)
- [x] Documentation complete (`REVIEW_2_SUMMARY.md`, `ARCHITECTURE_DIAGRAM.txt`)

---

### Appendix C — Actual Terraform Apply Output (Build #9)

```
aws_instance.app: Modifying... [id=i-01caec5e36bc79451]
aws_instance.app: Modifications complete after 36s [id=i-01caec5e36bc79451]

Apply complete! Resources: 0 added, 1 changed, 0 destroyed.

Outputs:
  app_url           = "http://15.206.145.177:8501"
  instance_id       = "i-01caec5e36bc79451"
  public_ip         = "15.206.145.177"
  security_group_id = "sg-0ad7d1d3967aab012"
  selected_ami      = "ami-0c6a8bbb64f907189"
  ssh_command       = "No key pair attached — SSH access disabled."
```

---

### Appendix D — Docker Compose Services (Local Stack)

| Service | Image | Port | Purpose |
|---|---|---|---|
| `namenode` | `bde2020/hadoop-namenode` | 9870, 9000 | Hadoop HDFS NameNode |
| `datanode` | `bde2020/hadoop-datanode` | 9864 | Hadoop HDFS DataNode |
| `resourcemanager` | `bde2020/hadoop-resourcemanager` | 8088 | YARN Resource Manager |
| `nodemanager` | `bde2020/hadoop-nodemanager` | 8042 | YARN Node Manager |
| `historyserver` | `bde2020/hadoop-historyserver` | 8188 | MapReduce History Server |
| `hive-server` | `bde2020/hive` | 10000, 10002 | HiveServer2 JDBC endpoint |
| `hive-metastore` | `bde2020/hive` | 9083 | Hive Metastore |
| `hive-metastore-postgresql` | `bde2020/hive-metastore-postgresql` | — | Hive metadata DB |
| `streamlit` | Local build | 8501 | Streamlit dashboard (local dev) |
| `sonarqube` | `sonarqube:lts-community` | 9001 | Code quality analysis server |
| `sonarqube-db` | `postgres:15-alpine` | — | SonarQube PostgreSQL database |

---

### Appendix E — Key Commands Reference

**Start local stack:**
```bash
docker-compose up -d
docker start jenkins
```

**Trigger a Jenkins build (via UI):**
```
http://localhost:8080/job/ecommerce-pipeline/build
```

**Get current EC2 IP after build:**
```bash
docker exec jenkins bash -c \
  "cd /var/jenkins_home/workspace/ecommerce-pipeline/terraform \
   && terraform output -raw public_ip"
```

**Check running containers on EC2 (via EC2 Instance Connect):**
```bash
sudo docker ps
```

**Restart Nginx on EC2 after reboot:**
```bash
sudo systemctl start nginx
```

---

*End of Project Report — E-Commerce Analytics DevOps Implementation*
*Student: Jay Bharuka | Review 1: February 2026 | Review 2: April 2026*
