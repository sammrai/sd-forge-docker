# Stable Diffusion WebUI Forge Docker Image

This Docker image is pre-configured for [Stable Diffusion WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) ** (not Auto1111 WebUI), offering a streamlined way to run the WebUI with all necessary components bundled. By using this Docker image, you gain access to the following features:

- **Pre-configured WebUI Forge setup**
- **Optional CivitAI model downloader integration**
- **Optional Cloudflare Tunnel configuration**

Simply clone the repository and start the service with `docker compose up -d`. Note that models are not included and must be downloaded separately using the provided options.

---

## Supported CUDA Versions

To ensure compatibility between your host machine's NVIDIA driver version and the appropriate Docker image, please refer to the following table:

| Docker Image                                 | Compatible Host NVIDIA Driver Version (Linux) | Forge Configuration                                      |
|----------------------------------------------|-----------------------------------------------|----------------------------------------------------------|
| `sammrai/sd-forge-docker:12.4.0`             | 550.54.14 or higher                           | CUDA 12.4 + Pytorch 2.4 (Fastest, but MSVC may be broken, xformers may not work)   |
| `sammrai/sd-forge-docker:12.1.0`             | 530.30.02 or higher                           | CUDA 12.1 + Pytorch 2.3.1 (Recommended)                                              |

**Important:** Ensure that your host machine's NVIDIA driver version falls within the specified range for the chosen Docker image.
**Note:** The driver version references above are based on publicly available information and do not guarantee actual compatibility or official endorsement.

You can check your current driver version by running:

```bash
nvidia-smi
```

This command will display the `Driver Version` installed on your system.
For more detailed information on CUDA and driver compatibility, please refer to NVIDIA's official documentation. 

## Setup Instructions

### 1. Clone the Repository

Run the following commands to clone the repository and navigate to the project directory:

```bash
git clone https://github.com/sammrai/sd-forge-docker.git
cd sd-forge-docker
```

### 2. Start the Docker Containers

Use Docker Compose to start the containers:

```bash
docker compose up -d
```

**Note:** Models are not included by default. You can either download them manually or use the optional CivitAI integration.

---

## Deployment Options

Choose the deployment method that best suits your needs:

### 1. Standard GPU Usage

By default, the `docker-compose.yml` file is configured for NVIDIA GPUs. Ensure your system has:

- The appropriate NVIDIA drivers
- The **NVIDIA Container Toolkit** installed

### 2. CPU-Only Usage

If GPU acceleration is not required, modify the `docker-compose.yml` file to disable GPU support and enable CPU-specific options. Update the `ARGS` environment variable as follows:

```diff
   environment:
-     ARGS: "--listen --enable-insecure-extension-access --port 7680 --api"
+     ARGS: "--listen --enable-insecure-extension-access --port 7680 --api --always-cpu --skip-torch-cuda-test"
-   deploy:
-     resources:
-       reservations:
-         devices:
-           - driver: nvidia
-             count: 1
-             capabilities: [gpu]
```

### 3. Using Cloudflare Tunnel

For secure external access to the WebUI, configure a **Cloudflare Tunnel**. Follow these steps:

1. **Set Up Environment Variables:**

   Add your `TUNNEL_TOKEN` to the `.env` file:
   ```
   TUNNEL_TOKEN=yourtoken...
   ```

2. **Replace `docker-compose.yml`:**

   Overwrite the default `docker-compose.yml` with `docker-compose-tunnel.yml`:
   ```bash
   cp docker-compose-tunnel.yml docker-compose.yml
   ```

3. **Start the Containers with the Tunnel Configuration:**

   ```bash
   docker compose up -d
   ```

4. **Complete the Cloudflare Tunnel Setup:**

   Refer to the [Cloudflare Tunnel Documentation](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/) for detailed setup instructions. This includes authenticating with Cloudflare and establishing the tunnel using your `TUNNEL_TOKEN`.

---

## CivitAI Integration (Optional)

The **CivitAI** model downloader automates the downloading and placement of models. While optional, it is **highly recommended** to add your **CIVITAI\_TOKEN** in the `.env` file, as many models require authentication to download.

### 1. Add Your CIVITAI Token

Include your **CIVITAI\_TOKEN** in the `.env` file to enable seamless integration:

```env
CIVITAI_TOKEN=yourtoken...
```

### 2. Supported Model Types and Example Commands

The following model types are supported for automatic downloads, along with their corresponding aliases to ensure models are placed in the correct directories:

- **Lora**: `@lora`
- **VAE**: `@vae`
- **Embed**: `@embed`
- **Checkpoint**: `@checkpoint`

Here are some example commands to download and place models:

```bash
docker compose exec webui civitdl 257749 439889 @checkpoint
docker compose exec webui civitdl 332646 @embed
docker compose exec webui civitdl 660673 @vae
docker compose exec webui civitdl 341353 @lora
```


