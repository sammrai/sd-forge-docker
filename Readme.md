
# Stable Diffusion WebUI Forge Docker Image

This Docker image is pre-configured for **Stable Diffusion WebUI Forge** (not Auto1111 WebUI), providing an easy way to run the WebUI with everything bundled inside the image. By using this Docker image, you get everything you need, including:

- **WebUI Forge configuration**
- **CivitAI model downloader integration**
- **Cloudflare Tunnel setup**

Simply clone the repository, configure your environment variables, and start the service with `docker compose up -d`. Models will be automatically placed in the correct directories based on the aliases you specify.

## Warning

> **CUDA 12.4 Required**  
> This Dockerfile is based on CUDA 12.4, which requires an Nvidia driver version >= 545.  
> To update the driver on **Ubuntu 22.04**, run the following commands:
> 
> ```bash
> sudo ubuntu-drivers install nvidia:545
> sudo reboot
> ```
> (Thanks to [@casao](https://github.com/Casao) for the guidance)

## Prerequisites

You will need the **NVIDIA Container Toolkit** to run this Docker image. You can follow the installation guide for your system here:  
[Install NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/sammrai/sd-forge-docker.git
   cd sd-forge-docker
   ```

2. Run the following command to start the Docker containers:
   ```bash
   docker compose up -d
   ```

## CivitAI Integration

To use the **CivitAI** model downloader, you'll need to provide your **CIVITAI_TOKEN** in the `.env` file. This integration supports automatic downloading and placing of models in the appropriate directories.

Model types supported for automatic download:
- Lora
- VAE
- Embed
- Checkpoint

You must use specific aliases for each model type when downloading. These aliases ensure that the models are placed in the correct directories:

- `@lora` for Lora models
- `@vae` for VAE models
- `@embed` for Embed models
- `@checkpoint` for Checkpoint models

By specifying the correct alias, the models will be automatically placed in the appropriate directory, avoiding the need for manual sorting.

To download a model, use the following commands:

```bash
docker compose exec webui civitdl 257749 439889 @checkpoint
docker compose exec webui civitdl 332646 @embed
docker compose exec webui civitdl 660673 @vae
docker compose exec webui civitdl 341353 @lora
```

## Cloudflare Tunnel

To access the WebUI from outside your local network, you can set up a **Cloudflare Tunnel**. Refer to other documentation for setting up the tunnel.

To use this feature, specify your **TUNNEL_TOKEN** in the `.env` file.

## Sample `.env` File

Here is an example of the `.env` file structure:

```
TUNNEL_TOKEN=yourtoken..
CIVITAI_TOKEN=yourtoken...
```
