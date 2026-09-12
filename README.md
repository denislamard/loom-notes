# loom-memory
Une mémoire locale pour Claude Desktop et mes agents LOOM. Serveur MCP + RAG hybride (BGE-M3, reranking, Qdrant). J'y mets ce que je veux, Claude vient y chercher. Tout tourne en local, sans cloud ni clé API.

docker rm -f qdrant
docker run -d --name qdrant --restart unless-stopped \
  --user "$(id -u):$(id -g)" \
  -e QDRANT__STORAGE__SNAPSHOTS_PATH=/qdrant/storage/snapshots \
  -p 127.0.0.1:6333:6333 \
  -v /home/denis/dev/loom-memory/data/qdrant:/qdrant/storage \
  qdrant/qdrant
sleep 3 && curl -s localhost:6333/
