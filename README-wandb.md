# wandb sux

You must do logging outside of WANDB because wandb/run/* files do not have an API or anything that you can use to read or parse them locally in Python. You are forced to upload them to wandb website.


## wandb server
is not enough


## docker conflicts

podman Kubernetes doesn't play nice with wandb and docker without this

#### `/etc/containers/registries.conf.d/10-unqualified-search-registries.conf`
```
unqualified-search-registries = ["docker.io"]
```
