build:
	docker build -t kwang2049/faiss-instant .

pull:
	docker pull kwang2049/faiss-instant

release:
	docker push kwang2049/faiss-instant

download:
	wget https://public.ukp.informatik.tu-darmstadt.de/kwang/faiss-instant/resources/ids.txt resources/ids.txt
	wget https://public.ukp.informatik.tu-darmstadt.de/kwang/faiss-instant/resources/pq-384-8b.index resources/pq-384-8b.index

run:
	docker run --detach --rm -it -p 5001:5000 -v resources:/opt/faiss-instant/resources --name faiss-instant kwang2049/faiss-instant

remove:
	docker rm -f faiss-instant

query_example:
	bash query_example.sh

reload_example:
	bash reload_example.sh
