// ============================================================
// Jenkinsfile — E-Commerce Analytics CI/CD Pipeline
// ============================================================
// Flow:
//   1. Checkout source from GitHub
//   2. SonarQube Analysis → code quality scan (bugs, smells, coverage)
//   3. Build Docker image (tagged with build number)
//   4. Push image to DockerHub
//   5. Terraform Init  → download providers, initialise backend
//   6. Terraform Validate → syntax / config check
//   7. Terraform Plan  → show what will change
//   8. Terraform Apply → provision EC2, start container
//   9. Post-build      → print app URL, cleanup workspace
//
// Jenkins Credentials required (configure at
//   Manage Jenkins → Credentials → System → Global):
//
//   ID                       Kind                    Usage
//   ──────────────────────── ─────────────────────── ──────────────────────────────────
//   dockerhub-creds          Username with password  DockerHub login
//   aws-credentials          Username with password  AWS_ACCESS_KEY_ID (username)
//                                                    AWS_SECRET_ACCESS_KEY (password)
//   sonarqube-token          Secret text             SonarQube user token
//   grafana-admin-password   Secret text             Grafana admin password
//                                                    (injected as TF_VAR_grafana_admin_password)
// ============================================================

pipeline {
    agent any

    // ── Global environment variables ──────────────────────────────────────
    environment {
        // DockerHub repository — change <username> to your DockerHub handle
        DOCKERHUB_REPO  = 'jaybharuka18/ecommerce-analytics'

        // Image tag uses the Jenkins build number for full traceability
        IMAGE_TAG       = "${BUILD_NUMBER}"

        // Fully-qualified image reference used throughout the pipeline
        FULL_IMAGE      = "${DOCKERHUB_REPO}:${IMAGE_TAG}"

        // Port the Streamlit app listens on
        APP_PORT        = '8501'

        // SonarQube server URL (running locally via docker-compose on port 9001)
        SONAR_HOST_URL  = 'http://localhost:9001'

        // Relative path to the Terraform directory inside the repository
        TF_DIR          = 'terraform'

        // Terraform auto-approve flag (set to '' to require manual confirmation)
        TF_AUTO_APPROVE = '-auto-approve'
    }

    stages {

        // ── 1. Checkout ───────────────────────────────────────────────────
        stage('Checkout') {
            steps {
                echo "==> Checking out source code from GitHub..."
                git branch: 'main',
                    url: 'https://github.com/jaybharuka/e-commerce-analysis.git'
                echo "==> Workspace contents:"
                sh 'ls -la'
            }
        }

        // ── 2. Unit Tests ─────────────────────────────────────────────────
        stage('Unit Tests') {
            steps {
                script {
                    echo "==> Running unit tests..."
                    sh """
                        pip install pytest pyspark --quiet
                        pytest tests/ -v --tb=short --junit-xml=test-results.xml
                    """
                }
            }
            post {
                always {
                    junit allowEmptyResults: true, testResults: 'test-results.xml'
                }
            }
        }

        // ── 3. SonarQube Analysis ─────────────────────────────────────────
        stage('SonarQube Analysis') {
            steps {
                catchError(buildResult: 'UNSTABLE', stageResult: 'UNSTABLE') {
                    script {
                        echo "==> Running SonarQube code analysis..."
                        withCredentials([
                            string(
                                credentialsId: 'sonarqube-token',
                                variable: 'SONAR_TOKEN'
                            )
                        ]) {
                            sh """
                                docker run --rm \\
                                    --network host \\
                                    -e SONAR_HOST_URL=${SONAR_HOST_URL} \\
                                    -e SONAR_TOKEN=${SONAR_TOKEN} \\
                                    -v \$(pwd):/usr/src \\
                                    sonarsource/sonar-scanner-cli:latest \\
                                    -Dsonar.projectKey=ecommerce-analytics \\
                                    -Dsonar.projectName="E-Commerce Analytics" \\
                                    -Dsonar.sources=. \\
                                    -Dsonar.exclusions=**/__pycache__/**,**/*.pyc,**/node_modules/**,ml_results/**,terraform/**,configs/**,dags/**,**/*.md,**/*.txt,**/*.pdf
                            """
                        }
                        echo "==> SonarQube analysis complete. View results at ${SONAR_HOST_URL}"
                    }
                }
            }
        }

        // ── 3. Build Docker Image ─────────────────────────────────────────
        stage('Build Docker Image') {
            steps {
                script {
                    echo "==> Building Docker image: ${FULL_IMAGE}"
                    sh """
                        docker build \
                            -t ${FULL_IMAGE} \
                            -t ${DOCKERHUB_REPO}:latest \
                            -f streamlit/dockerfile \
                            .
                    """
                    echo "==> Build complete. Image: ${FULL_IMAGE}"
                }
            }
        }

        // ── 3. Push to DockerHub ──────────────────────────────────────────
        stage('Push to DockerHub') {
            steps {
                script {
                    echo "==> Pushing image to DockerHub..."
                    withCredentials([
                        usernamePassword(
                            credentialsId: 'dockerhub-creds',
                            usernameVariable: 'DOCKER_USER',
                            passwordVariable: 'DOCKER_PASS'
                        )
                    ]) {
                        sh """
                            echo "${DOCKER_PASS}" | docker login -u "${DOCKER_USER}" --password-stdin

                            # Push versioned tag (immutable, traceable to this build)
                            docker push ${FULL_IMAGE}

                            # Push 'latest' tag so EC2 user_data can use a fixed reference
                            docker push ${DOCKERHUB_REPO}:latest

                            docker logout
                        """
                    }
                    echo "==> Image pushed: ${FULL_IMAGE}"
                }
            }
        }

        // ── 4. Terraform Init ─────────────────────────────────────────────
        stage('Terraform Init') {
            steps {
                script {
                    echo "==> Initialising Terraform in ./${TF_DIR}/ ..."
                    withCredentials([
                        usernamePassword(
                            credentialsId: 'aws-credentials',
                            usernameVariable: 'AWS_ACCESS_KEY_ID',
                            passwordVariable: 'AWS_SECRET_ACCESS_KEY'
                        )
                    ]) {
                        sh """
                            cd ${TF_DIR}
                            terraform init -input=false
                        """
                    }
                }
            }
        }

        // ── 5. Terraform Validate ─────────────────────────────────────────
        stage('Terraform Validate') {
            steps {
                script {
                    echo "==> Validating Terraform configuration..."
                    withCredentials([
                        usernamePassword(
                            credentialsId: 'aws-credentials',
                            usernameVariable: 'AWS_ACCESS_KEY_ID',
                            passwordVariable: 'AWS_SECRET_ACCESS_KEY'
                        )
                    ]) {
                        sh """
                            cd ${TF_DIR}
                            terraform validate
                        """
                    }
                    echo "==> Terraform configuration is valid."
                }
            }
        }

        // ── 6. Terraform Plan ─────────────────────────────────────────────
        stage('Terraform Plan') {
            steps {
                script {
                    echo "==> Generating Terraform execution plan..."
                    withCredentials([
                        usernamePassword(
                            credentialsId: 'aws-credentials',
                            usernameVariable: 'AWS_ACCESS_KEY_ID',
                            passwordVariable: 'AWS_SECRET_ACCESS_KEY'
                        ),
                        string(
                            credentialsId: 'grafana-admin-password',
                            variable: 'TF_VAR_grafana_admin_password'
                        )
                    ]) {
                        sh """
                            cd ${TF_DIR}
                            terraform plan \
                                -input=false \
                                -var="dockerhub_image=${FULL_IMAGE}" \
                                -out=tfplan
                        """
                    }
                    echo "==> Plan saved to ./${TF_DIR}/tfplan"
                }
            }
        }

        // ── 7. Terraform Apply ────────────────────────────────────────────
        stage('Terraform Apply') {
            steps {
                script {
                    echo "==> Applying Terraform plan — provisioning AWS infrastructure..."
                    withCredentials([
                        usernamePassword(
                            credentialsId: 'aws-credentials',
                            usernameVariable: 'AWS_ACCESS_KEY_ID',
                            passwordVariable: 'AWS_SECRET_ACCESS_KEY'
                        ),
                        string(
                            credentialsId: 'grafana-admin-password',
                            variable: 'TF_VAR_grafana_admin_password'
                        )
                    ]) {
                        sh """
                            cd ${TF_DIR}
                            terraform apply ${TF_AUTO_APPROVE} tfplan
                        """
                    }
                    echo "==> Infrastructure provisioned successfully."
                }
            }
        }

        // ── 8. Output App URL ─────────────────────────────────────────────
        stage('Deployment Info') {
            steps {
                script {
                    echo "==> Fetching deployment outputs from Terraform..."
                    withCredentials([
                        usernamePassword(
                            credentialsId: 'aws-credentials',
                            usernameVariable: 'AWS_ACCESS_KEY_ID',
                            passwordVariable: 'AWS_SECRET_ACCESS_KEY'
                        )
                    ]) {
                        // Capture the public IP output from Terraform state
                        def appUrl = sh(
                            script: "cd ${TF_DIR} && terraform output -raw app_url",
                            returnStdout: true
                        ).trim()

                        def publicIp = sh(
                            script: "cd ${TF_DIR} && terraform output -raw public_ip",
                            returnStdout: true
                        ).trim()

                        echo "=============================================="
                        echo " DEPLOYMENT SUCCESSFUL"
                        echo " Build       : #${BUILD_NUMBER}"
                        echo " Docker Image: ${FULL_IMAGE}"
                        echo " Instance IP : ${publicIp}"
                        echo " App URL     : ${appUrl}"
                        echo " NOTE: Allow ~60 seconds for EC2 user_data"
                        echo "       bootstrap to complete before accessing."
                        echo "=============================================="
                    }
                }
            }
        }
    }

    // ── Post-build actions ────────────────────────────────────────────────
    post {
        always {
            echo "==> Pipeline finished. Build #${BUILD_NUMBER}."
            // Remove dangling Docker images to keep the Jenkins host clean
            sh 'docker image prune -f || true'
        }
        success {
            echo "==> SUCCESS — E-Commerce app deployed and running."
        }
        failure {
            echo "==> FAILURE — Review the stage logs above for details."
            // Attempt a Terraform destroy on failure to avoid orphaned resources
            script {
                withCredentials([
                    usernamePassword(
                        credentialsId: 'aws-credentials',
                        usernameVariable: 'AWS_ACCESS_KEY_ID',
                        passwordVariable: 'AWS_SECRET_ACCESS_KEY'
                    )
                ]) {
                    sh """
                        cd ${TF_DIR}
                        terraform destroy -auto-approve \
                            -var="dockerhub_image=${FULL_IMAGE}" \
                            || true
                    """
                }
            }
        }
    }
}

