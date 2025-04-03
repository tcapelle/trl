# install git, curl and tmux
apt-get update
apt-get install -y git curl tmux

# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# source
source $HOME/.local/bin/env

export HF_HUB_CACHE=/workspace/cache/