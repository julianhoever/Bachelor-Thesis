#!/usr/bin/env python3

import sys
import gzip

if __name__ == "__main__":
    # Define compress level for the gzip algorithmus
    COMPRESSLEVEL = 9

    # Path to the target file
    file_path = sys.argv[1]

    # Open target file and read all uncompressed data
    with open(file_path, "rb") as f_in:
        uncompressed_data = f_in.read()

    # Compress data
    compressed_data = gzip.compress(uncompressed_data, COMPRESSLEVEL)

    # Print sizes
    print("### File:", file_path)
    print("Uncompressed size (MB): {:.4f}".format(len(uncompressed_data)/1e6))
    print("Compressed size (MB): {:.4f}".format(len(compressed_data)/1e6))
