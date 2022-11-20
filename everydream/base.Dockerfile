# 
# Dockerfile for base layer
#
FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

# Install base utilities
RUN apt-get update && \
    apt-get install -y wget git git-lfs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
    