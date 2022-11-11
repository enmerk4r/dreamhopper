# Dreamhopper
This project is a fork of [dreamfields-3D](https://github.com/shengyu-meng/dreamfields-3D). It was developed during the 2022 AEC Tech hackathon hosted by CORE studio at Thornton Tomasetti.

Dreamhopper turns an implementation of a [dreamfields](https://github.com/google-research/google-research/tree/master/dreamfields) NeRF model into a web-based API that can be easily incorporated into a Rhino + Grasshopper workflow. We wrapped the [dreamfields-3D](https://github.com/shengyu-meng/dreamfields-3D) model in a Flask server connected to a Redis cache, which means that now you will be able to easily generate prompt-based 3D meshes from your favorite design software.

![](https://github.com/enmerk4r/dreamhopper/blob/main/assets/dreamhopper-150.gif)

# Team members
- Alfredo Chavez
- Pablo Zamorano
- Francisco Landeros
- Sergey Pigach
- Alexandra Geremia
- Nathaniel Steinrueck
- Pedro Cortes
- Jefferey Moser
- Juan Arredondo

# Architecture
The architecture of this app is pretty simple. It has three main components:
- A Flask webserver that serves the NeRF model
- A REDIS cache that allows Flask to keep track of the jobs that are queued and the jobs that are complete
- A Rhino plugin that lets you pass prompts to the server and displays the results
```
+---------------------+
|     Flask Server    |
+---------------------+
| +---------------- + |            +--------------+
| |    NeRF model   | |------------| Rhino Plugin |
| +-----------------+ |            +--------------+
+---------------------+         
           |                    
           |                    
+---------------------+
|      Redis Cache    |
+---------------------+
```

![](https://github.com/enmerk4r/dreamhopper/blob/main/assets/dreamhopper-200.gif)

# Server Setup Instructions
The code framework is based on [torch-ngp](https://github.com/ashawkey/torch-ngp).

### 1. Clone the repo
```bash
git clone https://github.com/enmerk4r/dreamhopper.git
cd dreamhopper
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```
### 3. Install a custom verion of pymarchingcubes
```bash
bash scripts/install_PyMarchingCubes.sh
```

### 4. Build extensions
```bash
# install all extension modules
bash scripts/install_ext.sh
# if you prefer to install manually:
cd raymarching
python setup.py build_ext --inplace # build ext only, do not install (only can be used in the parent directory)
pip install . # install to python path (you still need the raymarching/ folder, since this only install the built extension.)
```

### 5. Install Redis
```bash
sudo apt install lsb-release
curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list
sudo apt-get update
sudo apt-get -y install redis
```

### 6. Start the Redis server
```bash
redis-server
```

### 7. Configure Flask server
In order to connect your Redis cache to the Flask server, please edit the [config.json](https://github.com/enmerk4r/dreamhopper/blob/main/config.json) file so that it reflects the HTTP port used by the Redis server. By default it is 6379
```json
{
    "redis": 6379
}
```
### 8. Start the Flask server
```bash
python server.py
```

### Tested environments
(For dreamfields-torch, not for dreamfileds-3D)
* Ubuntu 20 with torch 1.10 & CUDA 11.3 on a TITAN RTX.
* Ubuntu 20 with torch 1.13 & CUDA 11.7 on an RTX A6000.
* Windows 10 with torch 1.11 & CUDA 11.3 on an RTX 3070.

Currently, `--ff` only supports GPUs with CUDA architecture `>= 70`.
For GPUs with lower architecture, `--tcnn` can still be used, but the speed will be slower compared to more recent GPUs.


# Usage

Since inference takes considerable time even on powerful GPUs, this API uses a "ticketing" system. First, a `POST` request is sent to the `/generate` endpoint that starts mesh generation on a separate thread and returns the id of the submitted job. This id can be used by the client to periodically check whether the mesh has been generated. The overall flow looks something like this:
### 1. POST to /generate
```
Request:
[POST] /generate
{
  "text": "airplane",
  "iters": 15000.0,
  "seed": -1.0,
  "w": 224,
  "h": 224,
  "W": 384,
  "H": 384
}

Response:
{
  "id": "9a7c7512-8de0-4eaf-ab95-cebf78a27083"
}
```

### 2. Periodically check job status
You can then use the id to periodically check the status of your request:
```
Request:
[POST] /check
{
  "id": "9a7c7512-8de0-4eaf-ab95-cebf78a27083"
}
```
The response object from the '/check' route contains tons of parameters, but the two important ones are `done` and `mesh`. The former is a boolean that indicates whether mesh generation is complete, and once it is, the latter will be populated with output mesh geometry

# Acknowledgement

* The great paper and official JAX implementation of [dreamfields](https://ajayj.com/dreamfields):
    ```
    @article{jain2021dreamfields,
        author = {Jain, Ajay and Mildenhall, Ben and Barron, Jonathan T. and Abbeel, Pieter and Poole, Ben},
        title = {Zero-Shot Text-Guided Object Generation with Dream Fields},
        journal = {arXiv},
        month = {December},
        year = {2021},
    }   
    ```
