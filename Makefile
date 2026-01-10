# Makefile para InstructLab_test_for_uni - Turbo Mode

.PHONY: help up down restart clean logs shell-scraper install db-check

help:
	@echo "🚀 Comandos disponibles:"
	@echo "  make up           - Levanta todo el stack (Docker)"
	@echo "  make down         - Apaga el stack"
	@echo "  make clean        - BORRADO NUCLEAR: Elimina contenedores y DATOS (BBDD)"
	@echo "  make logs         - Ver logs de todos los contenedores"
	@echo "  make shell        - Entra en la terminal del contenedor scraper (Python)"
	@echo "  make db-check     - Verifica conexión a Postgres/PGVector"

# ---------------- Docker Management ----------------
up:
	docker compose up -d
	@echo "Stack operativo. Accede a http://localhost:8088 (Superset) o http://localhost:5050 (PgAdmin)"

down:
	docker compose down

restart: down up

logs:
	docker compose logs -f

# ---------------- Utilities ----------------
clean:
	@echo "Destruyendo volúmenes y datos..."
	docker compose down -v
	sudo rm -rf data/postgres_data data/redis_data data/ollama_models
	@echo "💀 Limpieza completada."

shell:
	docker compose exec scraper-worker /bin/bash

db-check:
	docker compose exec postgres psql -U admin -d opinion_intel_db -c "SELECT version();"
	docker compose exec postgres psql -U admin -d opinion_intel_db -c "SELECT * FROM pg_extension WHERE extname = 'vector';"