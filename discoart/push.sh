for n in {1..10}; do
    docker push entmike/discoart:preload && break;
done
