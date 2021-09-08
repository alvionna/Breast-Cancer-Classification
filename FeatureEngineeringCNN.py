
#Load the dependancies
from fastai.basics import *
from fastai.callback.all import *
from fastai.vision.all import *
from fastai.medical.imaging import *
from PIL import Image
import pydicom
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pandas as pd
import csv
import numpy as np
import cv2
import os
import skimage.transform
import cv2
from skimage import data, io, filters

keepCCHeight =0
keepCCWidth =0
keepMLOHeight =0
keepMLOWidth = 0
keepCCRatio = 0
keepMLORatio = 0

def MarginsBinary(MassMargins):
	margins = np.zeros(5)
	if 'SPICULATED' in MassMargins:
		margins[0]=1
	if 'ILL_DEFINED' in MassMargins:
		margins[1]=1
	if 'CIRCUMSCRIBED' in MassMargins:
		margins[2]=1
	if 'OBSCURED' in MassMargins:
		margins[3]=1
	if 'MICROLOBULATED' in MassMargins:
		margins[4]=1
	return margins

def maskImage(fullImage, ROIMask):
	ROIMaskArr = ROIMask.pixel_array
	fullImageArr = fullImage.pixel_array
	ROI = np.where(ROIMaskArr>1, fullImageArr, 1)
	return ROI

def ShapeBinary(MassShape):
	shapes = np.zeros(8)
	if 'IRREGULAR' in MassShape:
		shapes[0]=1
	if 'OVAL' in MassShape:
		shapes[1]=1
	if 'ARCHITECTURAL_DISTORTION' in MassShape:
		shapes[2]=1
	if 'LYMPH_NODE' in MassShape:
		shapes[3]=1
	if 'LOBULATED' in MassShape:
		shapes[4]=1
	if 'FOCAL_ASYMMETRIC_DENSITY' in MassShape:
		shapes[5]=1
	if 'ROUND' in MassShape:
		shapes[6]=1
	if 'ASYMMETRIC_BREAST_TISSUE' in MassShape:
		shapes[7]=1
	return shapes

def indicesSetup (pathStartNew, CSVFile):
	#get the indices of first instance of each unique patient id and the number of patient ids
	pIDs, pIDStartO = np.unique(CSVFile[:]['patient_id'], return_index=True)
	CCStart = np.copy(pIDStartO)
	MLOStartO = CCStart+1
	#isolate cases where there is MLO and CID
	i=-1
	for j in range(pIDStartO.size-1):
		i=i+1
		startI = pIDStartO[i]
		if (CSVFile.at[startI,'image view']!="CC"):
			pIDStartO = np.delete(pIDStartO,[i], None)
			MLOStartO=np.delete(MLOStartO,[i],None)
			i=i-1
		else:
			MLOPre=False
			if (np.size(pIDStartO)<=i+1):
				startNext = startI+1
			else:
				startNext = pIDStartO[i+1]
			MLOStartI=startI+1
			while (MLOStartI<startNext):
				if(CSVFile.at[MLOStartI,'image view']=="MLO"):
					if(MLOPre==False):
						MLOStartO[i]=MLOStartI
					MLOPre = True
				MLOStartI+=1
			if (MLOPre == False):
				pIDStartO = np.delete(pIDStartO,[i], None)
				MLOStartO=np.delete(MLOStartO,[i],None)
				i=i-1
	pathStart = 'drive/My Drive/Colab Notebooks/FeatureExtraction/'
	arrSize = np.size(pIDStart)
	CCrop = np.array([pydicom.dcmread(pathStart+'1-1.dcm')]*(arrSize))
	MLOCrop =np.array([pydicom.dcmread(pathStart+'1-1.dcm')]*arrSize)
	#load the DICOM files into an an array
	for i in range(arrSize):
		CCPath = pathStartNew+CSVFile.at[pIDStart[i],'cropped image file path']#+'.dcm'
		CCPath = CCPath.strip('000000.dcm')
		try:
			CCPath = 'd'+CCPath+'1-2.dcm'
			CCrop[i]=pydicom.dcmread(CCPath)
			if(CCrop[i].SeriesDescription=='ROI mask images'):
				CCPath = CCPath.strip('1-2.dcm')
				CCPath = 'd'+CCPath+'1-1.dcm'
		 		CCrop[i]=pydicom.dcmread(CCPath)
		except IOError:
			CCPath = CCPath.strip('1-2.dcm')
			CCPath = 'd'+CCPath+'1-1.dcm'
			CCrop[i]=pydicom.dcmread(CCPath)
			MLOPath = pathStartNew+CSVFile.at[MLOStart[i],'cropped image file path']#+'.dcm'
			MLOPath = MLOPath.strip('000000.dcm')
		try:
			MLOPath='d'+MLOPath+'1-2.dcm'
			MLOCrop[i]=pydicom.dcmread(MLOPath)
			if(MLOCrop[i].SeriesDescription=='ROI mask images'):
				MLOPath = MLOPath.strip('1-2.dcm')
				MLOPath = 'd'+MLOPath+'1-1.dcm'
				MLOCrop[i]=pydicom.dcmread(MLOPath)
		except IOError:
			MLOPath = MLOPath.strip('1-2.dcm')
			MLOPath = 'd'+MLOPath+'1-1.dcm'
			MLOCrop[i]=pydicom.dcmread(MLOPath)
	widthsCC= np.zeros(arrSize)
	heightsCC = np.zeros(arrSize)
	CCRatio = np.zeros(arrSize)
	widthsMLO = np.zeros(arrSize)
	heightsMLO=np.zeros(arrSize)
	MLORatio = np.zeros(arrSize)
	#standardize the heights and widths of each view
	for i in range(arrSize):
		widthsCC[i]=CCrop[i].pixel_array.shape[1]
		heightsCC[i]=CCrop[i].pixel_array.shape[0]
		CCRatio[i]=CCrop[i].pixel_array.shape[0]/CCrop[i].pixel_array.shape[1]
		widthsMLO[i]=MLOCrop[i].pixel_array.shape[1]
		heightsMLO[i]=MLOCrop[i].pixel_array.shape[0]
		MLORatio[i]=MLOCrop[i].pixel_array.shape[0]/MLOCrop[i].pixel_array.shape[1]
	if(pathStartNew=='CBIS-DDSM'):
		keepCCHeight = heightsCCMean
		keepCCWidth = heightsCCMean/ratioCCMean
		keepMLOHeight = heightsMLOMean
		keepMLOWidth = heightsMLOMean/ratioMLOMean
		keepCCRatio = ratioCCMean
		keepMLORatio = ratioMLOMean
	CCImages = []
	MLOImages = []
	#make sure the images were re-sized properly
	for i in range(arrSize):
		scale = 0
		croppedRescCC = ndimage.interpolation.zoom(CCrop[i].pixel_array, scale)
	 if CCRatio[i]>keepCCRatio:
		scale=keepCCWidth/widthsCC[i]
		rescaledCC = ndimage.interpolation.zoom(CCrop[i].pixel_array, scale)
	 	cutOffSides = (rescaledCC.shape[0]-round(keepCCHeight))/2
		secondCutoff = int(rescaledCC.shape[0]-cutOffSides)
		croppedRescCC = rescaledCC[int(cutOffSides):secondCutoff,:]
	elif CCRatio[i]==keepCCRatio:
		scale=keepCCWidth/widthsCC[i]
		croppedRescCC = ndimage.interpolation.zoom(CCrop[i].pixel_array, scale)
	else:
		scale=keepCCHeight/heightsCC[i]
		rescaledCC = ndimage.interpolation.zoom(CCrop[i].pixel_array, scale)
		cutOffSides = (rescaledCC.shape[1]-round(keepCCWidth))/2
		secondCutoff = int(rescaledCC.shape[1]-cutOffSides)
		croppedRescCC = rescaledCC[:,int(cutOffSides):secondCutoff]
		CCImages.append(croppedRescCC)
		scale = 0
		croppedRescMLO = ndimage.interpolation.zoom(MLOCrop[i].pixel_array, scale)
	if MLORatio[i]>keepMLORatio:
		scale=keepMLOWidth/widthsMLO[i]
		rescaledMLO = ndimage.interpolation.zoom(MLOCrop[i].pixel_array, scale)
		cutOffSides =(rescaledMLO.shape[0]-round(keepMLOHeight))/2
		secondCutoff = int(rescaledMLO.shape[0]-cutOffSides)
		croppedRescMLO = rescaledMLO[int(cutOffSides):secondCutoff,:]
	elif MLORatio[i]==keepMLORatio:
		scale=keepMLOWidth/widthsMLO[i]
		croppedRescMLO = ndimage.interpolation.zoom(MLOCrop[i].pixel_array, scale)
	else:
		scale=keepMLOHeight/heightsMLO[i]
		rescaledMLO = ndimage.interpolation.zoom(MLOCrop[i].pixel_array, scale)
		cutOffSides = (rescaledMLO.shape[1]-round(keepMLOWidth))/2
		secondCutoff = int(rescaledMLO.shape[1]-cutOffSides)
		croppedRescMLO = rescaledMLO[:,int(cutOffSides):secondCutoff]
	MLOImages.append(croppedRescMLO)
	ccFlatten = np.size(CCImages[0].flatten())
	mloFlatten = np.size(MLOImages[0].flatten())
	outArrSize = int(2+ccFlatten+mloFlatten)
	outArr = np.zeros((arrSize, outArrSize))
	#combine the y data from the metadata csv file with the flattened image arrays
	for i in range(arrSize):
		if('MALIGNANT' in CSVFile.loc[pIDStart[i], 'pathology']):
			outArr[i][0]=1
		else:
			outArr[i][0]=0
		outArr[i][1] = CSVFile.loc[pIDStart[i], 'assessment']
		outArr[i][2:(ccFlatten+2)]=CCImages[i].flatten()
		outArr[i][(ccFlatten+2):outArrSize]=MLOImages[i].flatten()
	if(pathStartNew=='CBIS-DDSM'):
		np.savetxt('CroppedPicsTrain.csv', totalMFeatures, delimiter=',', fmt="%d")
	else:
		np.savetxt('CroppedPicsTest.csv', totalMFeatures, delimiter=',', fmt="%d")

pathStartNew = 'CBIS-DDSM/'
CSVFile = pd.read_csv('mass_case_description_train_set.csv')
indicesSetup (pathStartNew, CSVFile)

pathStartNew = 'TestSet/'
CSVFile = pd.read_csv('mass_case_description_train_set.csv')
indicesSetup (pathStartNew, CSVFile)