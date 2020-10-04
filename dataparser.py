import struct
import numpy as np
import mlmath
from image import Image

labels = []
images = []
# imagesLin = np.empty((1))
# normalizedImages = np.empty((1))


def loadLabels(labelsPath="train-labels.idx1-ubyte", updateImages=True):
    global labels
    with open(labelsPath, "rb") as labelsFile:
        labelsFile.seek(8)
        labelBytes = labelsFile.read()
        labels = struct.unpack(">" + "B" * (len(labelBytes)), labelBytes)
    if updateImages and len(images) > 0:
        for imageIndex in range(len(images)):
            images[imageIndex].label = labels[imageIndex]


def loadImages(imagesPath="train-images.idx3-ubyte"):
    global images
    with open(imagesPath, "rb") as imagesFile:
        imagesFile.seek(4)
        numImages = int.from_bytes(imagesFile.read(4), "big")
        imageRows = int.from_bytes(imagesFile.read(4), "big")
        imageColumns = int.from_bytes(imagesFile.read(4), "big")

        images = np.empty((numImages, imageRows, imageColumns), np.ubyte)
        imageBytes = imagesFile.read()
        stepSize = imageRows * imageColumns

        imagesLin = [
            imageBytes[i : i + stepSize] for i in range(0, len(imageBytes), stepSize)
        ]

        for imageIndex in range(numImages):
            imageData = imagesLin[imageIndex]
            images.append(
                Image(
                    imageData,
                    (-1 if len(labels) <= 0 else labels[imageIndex]),
                    imageColumns,
                    imageRows,
                )
            )
