
import sys
from itertools import count

from pixelpipes import Engine, read_pipeline


def main():

    pipeline = read_pipeline(sys.argv[1])
    engine = Engine(pipeline, batch_size=1)

    for i in count(1):
        print(i)
        engine.next_batch()

if __name__ == "__main__":
    main()
