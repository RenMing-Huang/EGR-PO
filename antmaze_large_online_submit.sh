# export steps=1
# export env_name=antmaze-large-play-v2
# export CUDA_VISIBLE_DEVICES=0
# export load_path=None
# sh script/universe.sh

# export steps=5
# export env_name=antmaze-large-play-v2
# export CUDA_VISIBLE_DEVICES=0
# export load_path=None
# sh script/universe.sh

export steps=25
export group=debug
export env_name=antmaze-large-play-v2
export CUDA_VISIBLE_DEVICES=1
export load_path=None
export pretrain_path=pretrain path
sh script/online_universe.sh

# export steps=50
# export env_name=antmaze-large-play-v2
# export CUDA_VISIBLE_DEVICES=0
# export load_path=None
# sh script/universe.sh

# export steps=100
# export env_name=antmaze-large-play-v2
# export CUDA_VISIBLE_DEVICES=0
# export load_path=None
# sh script/universe.sh