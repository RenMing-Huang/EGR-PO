export steps=25
export env_name=antmaze-umaze-v2
export CUDA_VISIBLE_DEVICES=3
export load_path=None
sh script/universe.sh

export steps=25
export env_name=antmaze-medium-play-v2
export load_path=None # automatically load the d4rl path
sh script/universe.sh

export steps=25
export env_name=antmaze-large-play-v2
export load_path=None
sh script/universe.sh

export steps=25
export env_name=antmaze-ultra-play-v2
export load_path=None
sh script/universe.sh
