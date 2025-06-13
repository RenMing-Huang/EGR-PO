# export steps=1
# export env_name=antmaze-large-play-v2
# export load_path=None
# sh script/universe.sh

# export steps=5
# export env_name=antmaze-large-play-v2
# export load_path=None
# sh script/universe.sh

export steps=25
export env_name=calvin
export CUDA_VISIBLE_DEVICES=0
export load_path=None
sh script/universe.sh

export steps=5
export env_name=FetchPickAndPlace-v1
export CUDA_VISIBLE_DEVICES=0
export load_path=/data/hard_tasks_2e5/expert_small/FetchPick
sh script/universe.sh

# export steps=50
# export env_name=antmaze-large-play-v2
# export load_path=None
# sh script/universe.sh

# export steps=100
# export env_name=antmaze-large-play-v2
# export load_path=None
# sh script/universe.sh