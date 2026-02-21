pipeline {
    agent any

    environment {
        DOCKER_IMAGE = 'ecommerce-analytics'
        DOCKER_PORT = '8501'
    }

    stages {
        stage('Checkout') {
            steps {
                // Replace with your actual GitHub repository URL
                git branch: 'main', url: 'https://github.com/jaybharuka/e-commerce-analysis.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    echo "Building Docker image..."
                    sh "docker build -t ${DOCKER_IMAGE} -f streamlit/dockerfile ."
                }
            }
        }

        stage('Run Container') {
            steps {
                script {
                    echo "Running Docker container..."
                    // Stop and remove existing container if it exists
                    sh "docker stop ${DOCKER_IMAGE} || true"
                    sh "docker rm ${DOCKER_IMAGE} || true"
                    
                    // Run the new container
                    sh "docker run -d -p ${DOCKER_PORT}:${DOCKER_PORT} --name ${DOCKER_IMAGE} ${DOCKER_IMAGE}"
                }
            }
        }
    }

    post {
        always {
            echo "Pipeline execution completed."
        }
        success {
            echo "Deployment successful! App is running on port ${DOCKER_PORT}."
        }
        failure {
            echo "Deployment failed. Check the logs."
        }
    }
}
