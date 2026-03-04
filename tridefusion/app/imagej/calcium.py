import imagej
ij = imagej.init('/Applications/Fiji.app', mode='headless')
ij.ui().showUI()
print(f"ImageJ version: {ij.getVersion()}")