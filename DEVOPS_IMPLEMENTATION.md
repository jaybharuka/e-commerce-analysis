# DevOps Implementation for E-Commerce Analytics Project

## Project Overview
This document outlines the DevOps implementation for the E-Commerce Analytics application, demonstrating a complete CI/CD pipeline with containerization, automation, and infrastructure as code.

---

## Architecture Overview

```
┌─────────────────┐
│   Developer     │
│   (Local Code)  │
└────────┬────────┘
         │
         │ git push
         ▼
┌─────────────────┐
│     GitHub      │
│   Repository    │
└────────┬────────┘
         │
         │ webhook/trigger
         ▼
┌─────────────────┐
│     Jenkins     │
│   CI/CD Server  │
│                 │
│  ┌───────────┐  │
│  │ Checkout  │  │
│  └─────┬─────┘  │
│        │        │
│  ┌─────▼─────┐  │
│  │   Build   │  │
│  │   Docker  │  │
│  └─────┬─────┘  │
│        │        │
│  ┌─────▼─────┐  │
│  │   Deploy  │  │
│  └───────────┘  │
└────────┬────────┘
         │
         │ docker run
         ▼
┌─────────────────┐
│  Docker Engine  │
│                 │
│  ┌───────────┐  │
│  │ Container │  │
│  │ Port 8501 │  │
│  └───────────┘  │
└────────┬────────┘
         │
         │ terraform apply (future)
         ▼
┌─────────────────┐
│   AWS Cloud     │
│                 │
│  ┌───────────┐  │
│  │ EC2 t2    │  │
│  │ micro     │  │
│  └───────────┘  │
│                 │
│  ┌───────────┐  │
│  │ Security  │  │
│  │  Group    │  │
│  └───────────┘  │
└─────────────────┘
```

---

## Tools & Technologies Implemented

| Layer              | Tool/Technology | Purpose                                    |
|--------------------|-----------------|---------------------------------------------|
| Version Control    | Git             | Source code management                      |
| Repository         | GitHub          | Remote code hosting                         |
| CI/CD              | Jenkins         | Automated build and deployment pipeline     |
| Containerization   | Docker           | Application packaging and isolation         |
| Infrastructure     | Terraform        | Infrastructure as Code (IaC)                |
| Cloud Provider     | AWS              | Cloud hosting (EC2, Security Groups)        |
| Application        | Streamlit        | E-commerce analytics dashboard              |
| Base Image         | Python 3.9 Slim  | Lightweight container base                  |

---

## Implementation Details

### 1. Docker Containerization

**File:** `streamlit/dockerfile`

**What we did:**
- Created a production-ready Dockerfile using Python 3.9 slim base image
- Installed dependencies: streamlit, pyhive, pandas, plotly, thrift, thrift-sasl
- Set working directory to `/app`
- Exposed port 8501 for Streamlit application
- Configured CMD to run `app_complete.py`

**Benefits:**
- ✅ Consistent environment across development and production
- ✅ Easy deployment and portability
- ✅ Isolated application dependencies

**Commands:**
```bash
# Build Docker image
docker build -t ecommerce-analytics -f streamlit/dockerfile .

# Run container
docker run -d -p 8501:8501 ecommerce-analytics
```

---

### 2. Jenkins CI/CD Pipeline

**File:** `Jenkinsfile`

**What we did:**
- Created a declarative Jenkins pipeline with 3 stages
- Stage 1: Checkout - Clone code from GitHub repository
- Stage 2: Build - Create Docker image from Dockerfile
- Stage 3: Deploy - Stop old container and run new one
- Added post-build actions for success/failure notifications

**Custom Jenkins Setup:**
- Built custom Jenkins Docker image with Docker CLI installed (`Dockerfile.jenkins`)
- Mounted Docker socket to allow Jenkins to build images
- Running Jenkins on port 8080

**Pipeline Flow:**
```
Checkout → Build Docker Image → Run Container → Post Actions
```

**Configuration:**
- Repository: https://github.com/jaybharuka/e-commerce-analysis.git
- Branch: main
- Container Port: 8501

---

### 3. Terraform Infrastructure as Code

**File:** `terraform/main.tf`

**What we did:**
- Configured AWS provider for `ap-south-1` region
- Created Security Group with:
  - SSH access (port 22)
  - Streamlit access (port 8501)
  - Egress traffic allowed
- Defined EC2 instance:
  - Instance type: `t2.micro` (free tier eligible)
  - AMI: Ubuntu 24.04 LTS
  - Tag: EcommerceAnalyticsApp

**Status:**
- ✅ Terraform configuration written
- ✅ `terraform init` completed
- ✅ `terraform plan` successful (2 resources to create)
- ⏸️ `terraform apply` not executed (for Review 1 demonstration only)

**Resources Ready to Deploy:**
```
Plan: 2 to add, 0 to change, 0 to destroy.
  - aws_instance.ecommerce_app
  - aws_security_group.ecommerce_sg
```

---

### 4. Git & GitHub Setup

**What we did:**
- Initialized Git repository
- Created `.gitignore` for:
  - Python cache files (`__pycache__`, `*.pyc`)
  - Environment files (`.env`, `.venv`, `venv/`)
  - Terraform state files (`*.tfstate`, `.terraform/`)
- Created `requirements.txt` with application dependencies
- Pushed all code to GitHub repository

**Repository Structure:**
```
ecommerce-elt-pipeline-main/
├── streamlit/
│   ├── dockerfile
│   ├── app_complete.py
│   └── ...
├── terraform/
│   └── main.tf
├── Jenkinsfile
├── Dockerfile.jenkins
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Deployment Workflow

### Local Development to Production:

1. **Developer makes changes** → Commits to local Git
2. **Push to GitHub** → `git push origin main`
3. **Jenkins detects changes** → Pipeline triggered
4. **Jenkins runs stages:**
   - Clones latest code
   - Builds Docker image
   - Stops old container
   - Starts new container
5. **Application accessible** → http://localhost:8501
6. **(Future) Terraform deployment** → Provision AWS resources
7. **(Future) Deploy to cloud** → Application runs on EC2

---

## Review 1 Achievements

### ✅ Completed Tasks:

1. **Containerization**
   - ✅ Dockerfile created and tested
   - ✅ Docker image builds successfully
   - ✅ Container runs application on port 8501

2. **CI/CD Pipeline**
   - ✅ Jenkins server running in Docker
   - ✅ Jenkins pipeline configured
   - ✅ Automated build and deployment working
   - ✅ All pipeline stages passing (green)

3. **Infrastructure as Code**
   - ✅ Terraform configuration written
   - ✅ AWS resources defined
   - ✅ `terraform plan` validates successfully

4. **Version Control**
   - ✅ Code committed to Git
   - ✅ Pushed to GitHub repository
   - ✅ Clean project structure

### 📊 Demonstration Points:

1. **Docker Screenshot** - Running container and application in browser
2. **Jenkins Screenshot** - Successful pipeline with green stages
3. **Terraform Screenshot** - `terraform plan` output showing infrastructure
4. **GitHub Repository** - Complete codebase with DevOps configurations

---

## Technical Challenges Solved

### 1. Jenkins Docker Socket Permission
**Problem:** Jenkins couldn't access Docker to build images
**Solution:** 
- Created custom Jenkins image with Docker CLI
- Mounted Docker socket with root permissions
- `docker run --user root -v /var/run/docker.sock:/var/run/docker.sock`

### 2. Git Repository Corruption in Jenkins
**Problem:** Jenkins workspace had corrupted git directory
**Solution:**
- Modified pipeline to use `git clone` directly in shell
- Clean workspace before each build
- `rm -rf .git && git clone`

### 3. Port Conflicts
**Problem:** Port 8501 already in use by previous containers
**Solution:**
- Added container cleanup in pipeline
- `docker stop && docker rm` before new deployment

### 4. File Path Issues
**Problem:** Case sensitivity in dockerfile name
**Solution:**
- Verified exact filename in repository
- Used correct path: `streamlit/dockerfile`

---

## Future Enhancements (Review 2 & 3)

### Planned Additions:

1. **Kubernetes Deployment**
   - Container orchestration
   - Auto-scaling
   - Load balancing

2. **Monitoring & Logging**
   - Prometheus for metrics
   - Grafana dashboards
   - ELK stack for logs

3. **Advanced CI/CD**
   - Automated testing stage
   - Code quality checks (SonarQube)
   - Security scanning
   - Blue-Green deployments

4. **Cloud Deployment**
   - Execute `terraform apply`
   - Deploy to AWS EC2
   - Configure DNS
   - SSL certificates

5. **Additional DevOps Tools**
   - Ansible for configuration management
   - HashiCorp Vault for secrets
   - ArgoCD for GitOps

---

## Commands Reference

### Docker Commands:
```bash
# Build image
docker build -t ecommerce-analytics -f streamlit/dockerfile .

# Run container
docker run -d -p 8501:8501 --name ecommerce-analytics ecommerce-analytics

# Stop container
docker stop ecommerce-analytics

# Remove container
docker rm ecommerce-analytics

# View logs
docker logs ecommerce-analytics

# List containers
docker ps
```

### Jenkins Commands:
```bash
# Run Jenkins container
docker run -d -p 8080:8080 -p 50000:50000 \
  -v jenkins_home:/var/jenkins_home \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --user root --name jenkins my-jenkins-docker

# Get initial password
docker exec jenkins cat /var/jenkins_home/secrets/initialAdminPassword
```

### Terraform Commands:
```bash
# Initialize Terraform
terraform init

# Validate configuration
terraform validate

# Plan infrastructure changes
terraform plan

# Apply changes (creates resources)
terraform apply

# Destroy infrastructure
terraform destroy
```

### Git Commands:
```bash
# Initialize repository
git init

# Add files
git add .

# Commit changes
git commit -m "message"

# Push to GitHub
git push origin main

# Check status
git status
```

---

## Repository Information

**GitHub Repository:** https://github.com/jaybharuka/e-commerce-analysis.git

**Branch:** main

**Key Files:**
- `streamlit/dockerfile` - Container definition
- `Jenkinsfile` - CI/CD pipeline
- `terraform/main.tf` - Infrastructure code
- `requirements.txt` - Python dependencies
- `Dockerfile.jenkins` - Custom Jenkins image

---

## Contact & Documentation

**Date Implemented:** February 21-22, 2026

**Technologies Versions:**
- Docker: 29.2.1
- Jenkins: 2.541.2
- Terraform: 1.14.5
- Python: 3.9
- AWS Provider: 6.33.0

---

## Conclusion

This implementation demonstrates a complete DevOps workflow for the E-Commerce Analytics project, incorporating modern best practices in containerization, continuous integration/deployment, and infrastructure as code. The project is ready for Review 1 presentation with all core DevOps components successfully implemented and documented.

**Status:** ✅ Ready for Review 1
