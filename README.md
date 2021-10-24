# Faiss Instant
Build a Faiss service instantly. Faiss-instant will simply load **existing** Faiss index (and the corresponding ID mapping) and provide the search service via POST request.

## Usage
First, one needs to put the resource files (the ID mapping and the Faiss index, please refer to [resources/README.md](resources/README.md)) under the folder [./resources](./resources):
```bash
make download  # This will download example resource files
```
Then, one needs to start the faiss-instant service via docker:
```bash
docker pull kwang2049/faiss-instant  # Or `make pull`; or `make build` to build the docker image
docker run --detach --rm -it -p 5001:5000 -v resources:/opt/faiss-instant/resources --name faiss-instant kwang2049/faiss-instant  # Or `make run`; notice here a volume mapping will be made from ./resources to /opt/faiss-instant in the container
```
Finally, do the query:
```bash
bash query_example.sh  # curl 'localhost:5001/search' -X POST -d '{"k": 5, "vectors":  [[0.31800827383995056, -0.19993115961551666, -0.029884858056902885, ...]]}'
```
This will return the mappings from document IDs to the corresponding scores:
```json
[{"2426246":106.54305267333984,"4944584":107.05268096923828,"6195536":106.5833511352539,"6398884":107.19760131835938,"8077664":107.86164093017578}]
```
Whenever update the resources, one needs reload them:
```bash
curl 'localhost:5001/reload' -X GET  # Or `make reload`
```
## Reference
[plippe/faiss-web-service](https://github.com/plippe/faiss-web-service)