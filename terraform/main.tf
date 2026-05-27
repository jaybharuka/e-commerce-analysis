# ============================================================
# main.tf — Core AWS Resources
# Provider configuration lives in provider.tf
# All configurable values are declared in variables.tf
# ============================================================

# ------------------------------------------------------------
# Data Source: Latest Ubuntu 24.04 LTS AMI
# Automatically selects the most recent Ubuntu 24.04 (Noble)
# published by Canonical in the configured region.
# This avoids hardcoding an AMI ID that goes stale over time.
# ------------------------------------------------------------
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical's official AWS account ID

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd*/ubuntu-noble-24.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

# ------------------------------------------------------------
# Security Group
# Allows inbound:
#   22   — SSH management
#   8501 — Streamlit application
#   9090 — Prometheus metrics UI
#   3000 — Grafana dashboard
# All outbound traffic is permitted so the instance can reach
# DockerHub and the apt package mirrors on boot.
# ------------------------------------------------------------
resource "aws_security_group" "app_sg" {
  name        = "${var.project_name}-sg"
  description = "Allow SSH and Streamlit app traffic for ${var.project_name}"

  ingress {
    description = "SSH management access"
    from_port   = var.ssh_port
    to_port     = var.ssh_port
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "Streamlit application port"
    from_port   = var.app_port
    to_port     = var.app_port
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "Prometheus metrics UI"
    from_port   = 9090
    to_port     = 9090
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "Grafana dashboard"
    from_port   = 3000
    to_port     = 3000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "Allow all outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "${var.project_name}-sg"
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

# ------------------------------------------------------------
# EC2 Instance
# Bootstrap script (user_data) runs once on first boot and:
#   1. Updates the OS package index
#   2. Installs Docker CE
#   3. Starts + enables the Docker service
#   4. Pulls the specified DockerHub image
#   5. Runs the container with auto-restart on port 8501
# The image tag is injected by Jenkins at plan/apply time via
#   -var="dockerhub_image=<repo>:<build_number>"
# ------------------------------------------------------------
resource "aws_instance" "app" {
  ami                    = data.aws_ami.ubuntu.id # Latest Ubuntu 24.04 LTS via data source
  instance_type          = var.instance_type      # t2.micro (free tier)
  key_name               = var.key_name           # EC2 Key Pair for SSH (null = no SSH key)
  vpc_security_group_ids = [aws_security_group.app_sg.id]

  # cloud-init / user_data — executed as root on first boot
  # All output is logged to /var/log/userdata.log for easy debugging.
  # To watch progress after SSH: sudo tail -f /var/log/userdata.log
  user_data = <<-EOF
    #!/bin/bash
    exec > /var/log/userdata.log 2>&1

    echo "[$(date)] Starting EC2 bootstrap..."

    # ── Install Docker via convenience script ──────────────────
    apt-get update -y
    apt-get install -y curl
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
    echo "[$(date)] Docker installed and started."

    # ── Streamlit Application ───────────────────────────────────
    docker pull ${var.dockerhub_image}
    docker run -d \
        --net=host \
        --name ecommerce-app \
        --restart always \
        ${var.dockerhub_image}
    echo "[$(date)] Streamlit app started on port ${var.app_port}."

    # ── Node Exporter ───────────────────────────────────────────
    docker run -d \
        --net=host \
        --name node-exporter \
        --restart always \
        prom/node-exporter:latest
    echo "[$(date)] Node Exporter started on port 9100."

    # ── Prometheus config ───────────────────────────────────────
    mkdir -p /etc/prometheus
    printf '%s\n' \
        'global:' \
        '  scrape_interval: 15s' \
        'scrape_configs:' \
        '  - job_name: node-exporter' \
        '    static_configs:' \
        "      - targets: ['localhost:9100']" \
        > /etc/prometheus/prometheus.yml

    # ── Prometheus ──────────────────────────────────────────────
    docker run -d \
        --net=host \
        --name prometheus \
        --restart always \
        -v /etc/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml \
        prom/prometheus:latest \
        --config.file=/etc/prometheus/prometheus.yml \
        --web.listen-address=:9090
    echo "[$(date)] Prometheus started on port 9090."

    # ── Grafana ─────────────────────────────────────────────────
    docker run -d \
        --net=host \
        --name grafana \
        --restart always \
        -e GF_SECURITY_ADMIN_PASSWORD=${var.grafana_admin_password} \
        grafana/grafana:latest
    echo "[$(date)] Grafana started on port 3000."

    echo "[$(date)] Bootstrap complete. All services running."
  EOF

  tags = {
    Name        = "${var.project_name}-instance"
    Environment = var.environment
    ManagedBy   = "Terraform"
  }

  lifecycle {
    ignore_changes = [user_data]
  }
}
