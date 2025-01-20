#!/bin/bash
exec 1>/tmp/startup-log.txt 2>&1  # Redirect all output to a log file
set -x  # Print commands as they execute

# Variables will be replaced by the Python script
USER_NAME="__USER_NAME__"
PUBLIC_KEY="__PUBLIC_KEY__"
PRIVATE_KEY="__PRIVATE_KEY__"
AWS_ACCESS_KEY="__AWS_ACCESS_KEY__"
AWS_SECRET_KEY="__AWS_SECRET_KEY__"
AWS_REGION="__AWS_REGION__"
GIT_EMAIL="__GIT_EMAIL__"
GIT_NAME="__GIT_NAME__"

# Create .aws directory
mkdir -p /home/${USER_NAME}/.aws

# Create credentials file
cat > /home/${USER_NAME}/.aws/credentials << EOL
[default]
aws_access_key_id = {aws_access_key}
aws_secret_access_key = {aws_secret_key}
EOL

# Create config file
cat > /home/${USER_NAME}/.aws/config << EOL
[default]
region = ${AWS_REGION}
EOL

# Set proper ownership
chown -R ${USER_NAME}:${USER_NAME} /home/${USER_NAME}/.aws
chmod 600 /home/${USER_NAME}/.aws/credentials
chmod 600 /home/${USER_NAME}/.aws/config

chmod 700 /home/${USER_NAME}/.ssh

# Create id_ed25519 private key
cat > /home/${USER_NAME}/.ssh/id_ed25519 << EOL
${PRIVATE_KEY}
EOL
chmod 600 /home/${USER_NAME}/.ssh/id_ed25519

# Create id_ed25519.pub public key
cat > /home/${USER_NAME}/.ssh/id_ed25519.pub << EOL
${PUBLIC_KEY}
EOL
chmod 644 /home/${USER_NAME}/.ssh/id_ed25519.pub

# Set proper ownership
chown -R ${USER_NAME}:${USER_NAME} /home/${USER_NAME}/.ssh

# Set known hosts
su - ${USER_NAME} -c "ssh-keyscan -t rsa github.com >> /home/${USER_NAME}/.ssh/known_hosts"
chmod 644 /home/${USER_NAME}/.ssh/known_hosts

sudo chown -R ${USER_NAME}:${USER_NAME} /app
# Set git config if available
if [ ! -z "${GIT_EMAIL}" ]; then
  su - ${USER_NAME} -c "git config --global user.email \"${GIT_EMAIL}\""
fi
if [ ! -z "${GIT_NAME}" ]; then
  su - ${USER_NAME} -c "git config --global user.name \"${GIT_NAME}\""
fi
su - ${USER_NAME} -c "git config --global --add safe.directory /app"
su - ${USER_NAME} -c "git config --global pull.rebase false"

su - ${USER_NAME} -c "cd /app && git remote set-url origin git@github.com:horus-ai-labs/DistillFlow.git && git pull --rebase"

cat >> /app/venv/bin/activate << EOL

# Add env vars to .bashrc
S3_ACCESS_KEY=${AWS_ACCESS_KEY}
S3_SECRET_KEY=${AWS_SECRET_KEY}
AWS_REGION=${AWS_REGION}
EOL
chown ${USER_NAME}:${USER_NAME} /app/venv/bin/activate

source /app/venv/bin/activate

echo "Startup script completed" >> /tmp/startup-log.txt