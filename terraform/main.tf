provider "aws" {
  region = "ap-south-1"
}

resource "aws_security_group" "ecommerce_sg" {
  name        = "ecommerce_sg"
  description = "Allow SSH and Streamlit traffic"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 8501
    to_port     = 8501
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "ecommerce_app" {
  ami           = "ami-0287a05f0ef0e9d9a" # Ubuntu 24.04 LTS in ap-south-1 (update if needed)
  instance_type = "t2.micro"
  vpc_security_group_ids = [aws_security_group.ecommerce_sg.id]

  tags = {
    Name = "EcommerceAnalyticsApp"
  }
}
