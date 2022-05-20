
import sys
from itertools import count

from pixelpipes import read_pipeline


def main():

    pipeline = read_pipeline(sys.argv[1])

    for i in count(1):
        print(i)
        pipeline.run(i)

if __name__ == "__main__":
    main()