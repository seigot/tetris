run:
	docker run -d \
	-it \
	-p 6080:80 \
	--name tetris-pytorch \
	--shm-size=512m \
	--mount type=bind,source=/c/Users/eva_s/projects/tetris,target=/home/ubuntu/tetris \
	seigott/tetris_docker:pytorchv1.10 

start:
	python start.py -l2 -m train_sample -t -1 -d 1