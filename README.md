# Codabench Worker Setup

This guide explains how to set up a Codabench worker using Docker to run this competition.  
It is adapted from the official Codabench Wiki on [Running a Benchmark](https://github.com/codalab/codabench/wiki#running-a-benchmark).

---

## 1. Install Docker

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker   # Activate new group in current shell (no logout needed)
docker version
```
---

## 2. Install Docker Compose v2 plugin (if missing)

```bash
sudo apt-get update && sudo apt-get install -y docker-compose-plugin
docker compose version
```

---

## 3. Create working directory and volumes

```bash
mkdir -p ~/codabench-worker
cd ~/codabench-worker

# Host directory used by the worker to cache jobs/datasets
sudo mkdir -p /codabench
sudo chown $USER:$USER /codabench

# Create data directory
cd /codabench
mkdir data
```
There are two ways to load reference data for a competition:
	1.	From the `reference_data` folder provided by the competition organizers. This is uploaded during challenge creation and is handled automatically.
	2.	From a local directory on the worker machine
In this case, the organizer manually uploads their reference data to the following path on the worker:

```
/codabench/data/reference_data
```
The scoring program will be able to access the data stored in this path.


---

## 4. Create the `.env` file

1. Create a `.env` file in `~/codabench-worker`.
2. Replace `<desired broker URL>` with the actual one from **Queue Management, Actions, Copy Broker URL**.
3. Create `docker-compose.yml` with the appropriate configuration.

Example `.env`:
```
# Queue URL
BROKER_URL=<desired broker URL>

# Location to store submissions/cache -- absolute path!
HOST_DIRECTORY=/codabench

# If SSL isn't enabled, then comment or remove the following line
BROKER_USE_SSL=True
```

Example `docker-compose.yml`:
```
# Codabench Worker
services:
    worker:
        image: codalab/competitions-v2-compute-worker:latest
        container_name: compute_worker
        volumes:
            - /codabench:/codabench
            - /var/run/docker.sock:/var/run/docker.sock
        env_file:
            - .env
        restart: unless-stopped
        #hostname: ${HOSTNAME}
        logging:
            options:
                max-size: 50m
                max-file: 3

```
---

## 5. Pull and start containers

```bash
# Pull worker images
# Use a customized Docker image with the necessary Python packages installed
docker pull wanyingge/pytorch-speech:2.8.0-cu126

# Start the worker
docker compose up -d

# View logs
docker logs -f compute_worker

# If everything is running correctly, you should see:
# [2025-08-12 07:12:32,996: INFO/MainProcess] compute-worker@xxx ready.
```

---

## 6. Point your competition to the queue

Edit the **challenge homepage** to add the Queue with the corresponding name in your **Queue Management**.

---

## Notes

- Ensure you have enough disk space for datasets and logs in `/codabench`.
- If you change your `.env` or `docker-compose.yml`, run:
```bash
docker compose down && docker compose up -d
```
