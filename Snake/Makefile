.PHONY: install train play

install:
	python -m pip install --upgrade -r requirements.txt

train:
	./server.py --fast-train --num-episodes 10

play:
	./server.py --num-episodes 10

play-human:
	./server.py --agent human

train2D:
	./server.py --fast-train --num-episodes 40 --two-d --plot

play2D:
	./server.py --num-episodes 10 --two-d

play-human2D:
	./server.py --agent human --two-d

test2D:
	./server.py --test --num-episodes 40 --num-runs 10 --two-d

clean:
	find . -regex '.*\(__pycache__\|\.py[cod]\|\.h5\)' -delete