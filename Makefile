build:
	docker image rm -f kwang2049/faiss-instant
	docker build -t kwang2049/faiss-instant .

build-gpu:
	docker image rm -f kwang2049/faiss-instant-gpu
	docker build -t kwang2049/faiss-instant-gpu -f Dockerfile.gpu .

pull:
	docker pull kwang2049/faiss-instant

pull-gpu:
	docker pull kwang2049/faiss-instant-gpu

release:
	docker push kwang2049/faiss-instant

release-gpu:
	docker push kwang2049/faiss-instant-gpu

download:
	wget https://public.ukp.informatik.tu-darmstadt.de/kwang/faiss-instant/dpr-single-nq-base.size-10000/ivf-32-sq-QT_8bit_uniform.txt -P ./resources
	wget https://public.ukp.informatik.tu-darmstadt.de/kwang/faiss-instant/dpr-single-nq-base.size-10000/ivf-32-sq-QT_8bit_uniform.index -P ./resources

run:
	# docker rm -f faiss-instant
	docker run --detach --rm -it -p 5001:5000 -v ${PWD}/resources:/opt/faiss-instant/resources --name faiss-instant kwang2049/faiss-instant

run-gpu:
	# docker rm -f faiss-instant-gpu
	docker run --runtime=nvidia -e CUDA_VISIBLE_DEVICES=0 --detach --rm -it -p 5001:5000 -v ${PWD}/resources:/opt/faiss-instant/resources --name faiss-instant-gpu kwang2049/faiss-instant-gpu

remove:
	docker rm -f faiss-instant
	docker image rm kwang2049/faiss-instant
	docker rm -f faiss-instant-gpu
	docker image rm kwang2049/faiss-instant-gpu

query:
	bash query_example.sh

reload:
	bash reload_example.sh

reload-gpu:
	bash reload_example-gpu.sh

index-list:
	curl -X GET 'http://localhost:5001/index_list'

reconstruct:
	curl -X 'GET' \
		'http://localhost:5001/reconstruct?id=1'

explain:
	bash explain_example.sh
