# ============================================================
# Outputs — printed at the end of `terraform apply`
# The Jenkins stage echoes these so the public URL is
# immediately visible in the build log.
# ============================================================

output "instance_id" {
  description = "EC2 instance ID."
  value       = aws_instance.app.id
}

output "public_ip" {
  description = "Public IPv4 address of the deployed EC2 instance."
  value       = aws_instance.app.public_ip
}

output "app_url" {
  description = "Direct URL to access the Streamlit application."
  value       = "http://${aws_instance.app.public_ip}:${var.app_port}"
}

output "security_group_id" {
  description = "ID of the security group attached to the instance."
  value       = aws_security_group.app_sg.id
}

output "ssh_command" {
  description = "SSH command to connect to the EC2 instance. Requires key_name variable to be set."
  value       = var.key_name != null ? "ssh -i ~/.ssh/${var.key_name}.pem ubuntu@${aws_instance.app.public_ip}" : "No key pair attached — SSH access disabled."
}

output "selected_ami" {
  description = "AMI ID selected by the data source (Ubuntu 24.04 LTS)."
  value       = data.aws_ami.ubuntu.id
}
