import struct
import numpy as np
import mlmath

labels = []
images = np.empty((1))
normalizedImages = np.empty((1))


def loadLabels(labelsPath="train-labels.idx1-ubyte"):
    global labels
    with open(labelsPath, "rb") as labelsFile:
        labelsFile.seek(8)
        labelBytes = labelsFile.read()
        labels = struct.unpack(">" + "B" * (len(labelBytes)), labelBytes)


def loadImages(imagesPath="train-images.idx3-ubyte"):
    global images
    with open(imagesPath, "rb") as imagesFile:
        imagesFile.seek(4)
        numImages = int.from_bytes(imagesFile.read(4), "big")
        imageRows = int.from_bytes(imagesFile.read(4), "big")
        imageColumns = int.from_bytes(imagesFile.read(4), "big")

        images = np.empty((numImages, imageRows, imageColumns), np.ubyte)
        normalizedImages = np.empty((numImages, imageRows, imageColumns), float)
        imageBytes = imagesFile.read()
        stepSize = imageRows * imageColumns

        imagesLin = [
            imageBytes[i : i + stepSize] for i in range(0, len(imageBytes), stepSize)
        ]

        for imageIdx in range(numImages):
            for row in range(imageRows):
                for col in range(imageColumns):
                    images[imageIdx, row, col] = imagesLin[imageIdx][
                        row * imageColumns + col
                    ]
                    normalizedImages[imageIdx, row, col] = mlmath.normalize(
                        images[imageIdx, row, col], 0, 255
                    )
