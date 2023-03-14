# Example use:

```
docker run --rm --name civitai-cache-0 -it -v /mnt/user/models/civitai-cache:/app/models \
-e MONGODB_CONNECTION=mongodb://localhost:27017 \
-e MONGODB_DATABASE=database \
-e S3_BUCKET=your.bucket.name \
-e MODELS_COLLECTION=models \
-e S3_AWS_SERVER_PUBLIC_KEY=[your public key] \
-e S3_AWS_SERVER_SECRET_KEY=[your S3 secret key] \
entmike/civitai-cache python images.py
```
