# DevOps Implementation — Review 2 Summary
## Real Cloud Deployment with Terraform + Docker on AWS EC2

**Student:** Jay Bharuka
**Date:** April 2026
**Project:** E-Commerce Analytics — Review 2
**Repository:** https://github.com/jaybharuka/e-commerce-analysis.git

---

## What Changed from Review 1 → Review 2

| Feature | Review 1 | Review 2 |
|---|---|---|
| Terraform | Written, `plan` only | **Full `apply` — real AWS resources** |
| AMI | Hardcoded AMI ID | **Auto-selects latest Ubuntu 24.04** |
| SSH Key | Not configured | **Key pair attached to EC2** |
| DockerHub | Not used | **Jenkins pushes image, EC2 pulls it** |
| Jenkins | 3 stages | **8 stages: build → push → terraform apply** |
| Cloud | Nothing deployed | **EC2 instance live on AWS ap-south-1** |

---

## Architecture (Review 2)

```
Developer
    │
    │  git push
    ▼
GitHub Repository
    │
    │  Webhook / Manual Trigger
    ▼
Jenkins CI/CD Pipeline
    ├── Stage 1: Checkout          → Clone latest code
    ├── Stage 2: Build Image       → docker build -t jaybharuka/ecommerce-analytics:<build>
    ├── Stage 3: Push to DockerHub → docker push (versioned + latest)
    ├── Stage 4: Terraform Init    → Download AWS provider
    ├── Stage 5: Terraform Validate→ Syntax check
    ├── Stage 6: Terraform Plan    → Preview: 2 resources to create
    ├── Stage 7: Terraform Apply   → Provision EC2 + Security Group
    └── Stage 8: Deployment Info   → Print Public IP + App URL
         │
         │  terraform apply
         ▼
    AWS EC2 (ap-south-1, t2.micro)
         │
         │  user_data bootstrap (on first boot)
         ├── apt-get install docker-ce
         ├── systemctl start docker
         └── docker run -d -p 8501:8501 jaybharuka/ecommerce-analytics:<build>
              │
              ▼
         http://<public-ip>:8501   ← Live Streamlit App
```

---

## File Structure (Terraform)

```
terraform/
├── provider.tf               ← AWS provider config, version constraints
├── variables.tf              ← All input variables with defaults
├── main.tf                   ← data aws_ami + security group + EC2 instance
├── outputs.tf                ← public_ip, app_url, ssh_command, ami_id
├── terraform.tfvars.example  ← Template — copy to terraform.tfvars and fill in
└── .terraform.lock.hcl       ← Provider version lock (auto-generated)
```

---

## File Explanations

### `provider.tf` — *"Which cloud and which version?"*
Tells Terraform to use AWS in `ap-south-1`. Version `~> 6.0` means use any 6.x release.
Credentials come from environment variables — **never hardcoded**.

### `variables.tf` — *"All the knobs you can turn"*
Defines every configurable value:
- `aws_region` — which AWS region to deploy in
- `instance_type` — `t2.micro` is free-tier eligible
- `key_name` — your EC2 SSH key pair name
- `dockerhub_image` — image Jenkins built and pushed (injected at runtime)
- `app_port`, `ssh_port` — network ports
- `project_name`, `environment` — used as resource tags

### `main.tf` — *"What to build"*
Three blocks:

1. **`data "aws_ami" "ubuntu"`** — Queries AWS API to find the **most recent Ubuntu 24.04 LTS**
   AMI published by Canonical (account `099720109477`). No hardcoded AMI IDs.

2. **`aws_security_group`** — Acts as a firewall:
   - Port 22 open → SSH access
   - Port 8501 open → Streamlit app access
   - All outbound open → EC2 can pull Docker images

3. **`aws_instance`** — The actual EC2 virtual machine with:
   - `user_data` bootstrap script that runs **once on first boot**
   - Installs Docker, starts it, pulls your image, runs the container

### `outputs.tf` — *"What to show after apply"*
After `terraform apply` completes, these values are printed:
- `public_ip` — the EC2's public IPv4 address
- `app_url` — `http://<ip>:8501` (click to open app)
- `ssh_command` — exact SSH command to log in
- `selected_ami` — which AMI Terraform chose (useful for auditing)

---

## Prerequisites Before Running

### Step 1 — AWS Account Setup
1. Create a free AWS account at https://aws.amazon.com
2. Go to **IAM → Users → Add User**
3. Attach policy: `AmazonEC2FullAccess`
4. Under **Security Credentials**, create **Access Key**
5. Save `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`

### Step 2 — Create EC2 Key Pair
1. AWS Console → **EC2 → Key Pairs → Create key pair**
2. Name it: `ecommerce-key`
3. Format: `.pem` (for Linux/Mac SSH)
4. Download the `.pem` file → save to `~/.ssh/ecommerce-key.pem`
5. Set permissions (Linux/Mac only): `chmod 400 ~/.ssh/ecommerce-key.pem`

### Step 3 — DockerHub Account
1. Create account at https://hub.docker.com
2. Create repository: `ecommerce-analytics`
3. Note your username (e.g., `jaybharuka`)

### Step 4 — Jenkins Credentials
In Jenkins → **Manage Jenkins → Credentials → System → Global → Add Credentials**:

| Credential ID | Kind | Username | Password |
|---|---|---|---|
| `dockerhub-creds` | Username with password | DockerHub username | DockerHub password |
| `aws-credentials` | Username with password | AWS_ACCESS_KEY_ID | AWS_SECRET_ACCESS_KEY |

---

## Running Terraform (Manual / Local)

### First Time Setup
```bash
# 1. Navigate to terraform directory
cd terraform

# 2. Copy example vars file and fill in your values
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars — set dockerhub_image and key_name

# 3. Initialize Terraform (download AWS provider ~> 6.0)
terraform init

# If you previously ran init with an older provider version, upgrade it:
terraform init -upgrade
```

### Check Configuration
```bash
# Validate HCL syntax (no AWS calls needed)
terraform validate

# Format code (optional, good practice)
terraform fmt
```

### Preview Changes
```bash
# See what Terraform WILL create — no changes made yet
terraform plan -var="dockerhub_image=jaybharuka/ecommerce-analytics:latest"

# Save the plan to a file (recommended — Jenkins uses this)
terraform plan \
  -var="dockerhub_image=jaybharuka/ecommerce-analytics:latest" \
  -out=tfplan
```

**Expected output:**
```
Plan: 2 to add, 0 to change, 0 to destroy.
  + aws_security_group.app_sg
  + aws_instance.app
```

### Deploy to AWS
```bash
# Apply the saved plan (no extra confirmation needed)
terraform apply tfplan

# OR apply directly with auto-approve (what Jenkins does)
terraform apply -auto-approve \
  -var="dockerhub_image=jaybharuka/ecommerce-analytics:latest"
```

**Expected output after apply:**
```
Apply complete! Resources: 2 added, 0 changed, 0 destroyed.

Outputs:

app_url        = "http://13.235.xxx.xxx:8501"
instance_id    = "i-0abc123def456"
public_ip      = "13.235.xxx.xxx"
selected_ami   = "ami-0xxxxxxxxxxxxxxxxx"
security_group_id = "sg-0xxxxxxxxxxxxxxxxx"
ssh_command    = "ssh -i ~/.ssh/ecommerce-key.pem ubuntu@13.235.xxx.xxx"
```

> **Wait ~90 seconds** after apply before opening the app URL.
> The EC2 instance needs time to install Docker and start the container.

### SSH into the Instance (to debug)
```bash
# Use the ssh_command output value
ssh -i ~/.ssh/ecommerce-key.pem ubuntu@<public-ip>

# Check bootstrap progress
sudo tail -f /var/log/userdata.log

# Check if Docker container is running
docker ps

# Check container logs
docker logs ecommerce-app
```

### Tear Down (Save AWS Costs)
```bash
# Destroy all created resources
terraform destroy -var="dockerhub_image=jaybharuka/ecommerce-analytics:latest"
# Type 'yes' when prompted
```

---

## Full Jenkins Pipeline Flow

When you trigger a Jenkins build, these 8 stages run automatically:

```
Stage 1: Checkout          → git clone from GitHub
Stage 2: Build Docker      → docker build -t jaybharuka/ecommerce-analytics:<BUILD_NUMBER>
Stage 3: Push DockerHub    → docker push (versioned tag + :latest)
Stage 4: Terraform Init    → terraform init -input=false
Stage 5: Terraform Validate→ terraform validate
Stage 6: Terraform Plan    → terraform plan -var="dockerhub_image=..." -out=tfplan
Stage 7: Terraform Apply   → terraform apply -auto-approve tfplan
Stage 8: Deployment Info   → Print public_ip and app_url from terraform output
```

**On failure:** The pipeline automatically runs `terraform destroy` to clean up orphaned AWS resources.

---

## Common Errors & Fixes

### Error 1: `InvalidKeyPair.NotFound`
```
Error: InvalidKeyPair.NotFound: The key pair 'ecommerce-key' does not exist
```
**Cause:** The key pair name in `terraform.tfvars` doesn't match what's in AWS.

**Fix:**
```bash
# Check existing key pairs in AWS
aws ec2 describe-key-pairs --region ap-south-1

# Or go to AWS Console → EC2 → Key Pairs
# Make sure the name matches exactly (case-sensitive)
```

---

### Error 2: `AuthFailure` / `UnauthorizedOperation`
```
Error: AuthFailure: AWS was not able to validate the provided access credentials
```
**Cause:** Wrong or missing AWS credentials.

**Fix:**
```bash
# Set environment variables (Linux/Mac)
export AWS_ACCESS_KEY_ID="your-key-id"
export AWS_SECRET_ACCESS_KEY="your-secret-key"

# Verify credentials work
aws sts get-caller-identity
```
In Jenkins: verify the `aws-credentials` credential ID is spelled correctly and the values are correct.

---

### Error 3: Port 8501 Not Accessible (App Not Loading)
```
Browser: This site can't be reached  (http://<ip>:8501)
```
**Cause (A):** EC2 bootstrap not finished yet.

**Fix:** Wait 60–90 seconds and retry. SSH in and check:
```bash
sudo tail -f /var/log/userdata.log
```

**Cause (B):** Docker container not running.

**Fix:**
```bash
ssh -i ~/.ssh/ecommerce-key.pem ubuntu@<ip>
docker ps                          # container should be listed
docker logs ecommerce-app          # check for startup errors
```

**Cause (C):** Security group not allowing port 8501.

**Fix:** Verify in AWS Console → EC2 → Security Groups → `ecommerce-analytics-sg` → Inbound Rules. Should have:
- Port 22, Source: 0.0.0.0/0
- Port 8501, Source: 0.0.0.0/0

---

### Error 4: Docker Pull Fails on EC2
```
# In /var/log/userdata.log:
Error response from daemon: pull access denied for jaybharuka/ecommerce-analytics
```
**Cause:** Image not pushed to DockerHub yet, or wrong image name.

**Fix:**
```bash
# Manually push your image to DockerHub first
docker login
docker build -t jaybharuka/ecommerce-analytics:latest -f streamlit/dockerfile .
docker push jaybharuka/ecommerce-analytics:latest
```
Then re-run `terraform apply`.

---

### Error 5: `terraform init` Fails After Provider Upgrade
```
Error: Failed to query available provider packages
```
**Cause:** Lock file has old provider hash, needs upgrade.

**Fix:**
```bash
terraform init -upgrade
```
This regenerates `.terraform.lock.hcl` for the new `~> 6.0` provider.

---

### Error 6: `ResourceAlreadyExists` for Security Group
```
Error: InvalidGroup.Duplicate: The security group 'ecommerce-analytics-sg' already exists
```
**Cause:** Previous `terraform apply` created the group but state was lost.

**Fix:**
```bash
# Import the existing resource into state
terraform import aws_security_group.app_sg <sg-id>
# Then run terraform apply again
```
Or delete the existing security group from AWS Console and re-apply.

---

## Review 2 Demonstration Checklist

### What to Show:

1. **Jenkins Pipeline** (all 8 stages green)
   - Show build log with DockerHub push success
   - Show Terraform plan output: `2 to add`
   - Show Terraform apply output with `app_url`

2. **AWS Console**
   - EC2 → Instances → Show running instance (`ecommerce-analytics-instance`)
   - EC2 → Security Groups → Show inbound rules (22, 8501)

3. **DockerHub**
   - Show pushed image with versioned tag (`:42`, `:43`, etc.) + `:latest`

4. **Live Application**
   - Open `http://<public-ip>:8501` in browser
   - Show Streamlit dashboard running on cloud

5. **Terraform Outputs**
   - Show terminal with all 6 outputs printed

---

## Key Talking Points for Presentation

1. **"Why data source for AMI instead of hardcoded?"**
   > Hardcoded AMI IDs become outdated when Ubuntu releases patches. The `data "aws_ami"` block queries AWS at runtime and always picks the latest stable Ubuntu 24.04 from Canonical's official account (`099720109477`).

2. **"How does the app get onto the EC2?"**
   > Jenkins builds the Docker image, tags it with the build number, and pushes it to DockerHub. Terraform's `user_data` script runs on first EC2 boot — it installs Docker, then pulls that exact image and runs it with `--restart always`.

3. **"What does `--restart always` do?"**
   > If the EC2 reboots or Docker crashes, it automatically restarts the container — no manual intervention needed.

4. **"How are credentials handled securely?"**
   > AWS credentials are stored in Jenkins as encrypted credentials (not in any file). Terraform reads them via environment variables. No secrets are committed to Git.

5. **"What is `terraform destroy` used for?"**
   > AWS charges by the hour. After the demo, `terraform destroy` tears down the EC2 and security group in ~30 seconds so you don't get billed. The code stays in Git — you can re-provision anytime.

---

## Review 2 Status

| Component | Status | Detail |
|---|---|---|
| Provider upgraded to v6.0 | ✅ | `~> 6.0` in provider.tf |
| AMI auto-selection | ✅ | `data "aws_ami"` — always latest Ubuntu 24.04 |
| Key pair support | ✅ | `key_name` variable added |
| User data bootstrap | ✅ | Docker install + pull + run on boot |
| All outputs | ✅ | ip, url, ssh_command, ami_id |
| Jenkins 8-stage pipeline | ✅ | Full build→push→deploy cycle |
| `terraform apply` | ⚡ Ready | Needs AWS credentials + key pair to execute |
| `terraform.tfvars.example` | ✅ | Template provided |

**Status: Ready for `terraform apply` — requires AWS credentials and EC2 key pair setup.**

---

*Prepared for Review 2 — E-Commerce Analytics DevOps Project*
