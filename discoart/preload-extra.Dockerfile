FROM entmike/discoart:preload

COPY models/discoart/extra /models/.cache/discoart
COPY models/clip/extra /models/.cache/clip

CMD [ "/bin/bash" ]