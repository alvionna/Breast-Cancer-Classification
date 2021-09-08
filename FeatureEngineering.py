
#Load the dependancies
from fastai.basics import *
from fastai.callback.all import *
from fastai.vision.all import *
from fastai.medical.imaging import *
from PIL import Image
import pydicom
#import seaborn as sns
#matplotlib.rcParams['image.cmap'] = 'bone'
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

#binarize the mass margins
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

#mask the image to get the ROI
def maskImage(fullImage, ROIMask):
	ROIMaskArr = ROIMask.pixel_array
	fullImageArr = fullImage.pixel_array
	ROI = np.where(ROIMaskArr>1, fullImageArr, 1)
	return ROI

def ROIFeatures(ROI):
	#extract some features from the images (ROI) CC and MLO
	#[max, min, intensity, width, height, area]
	mamFeatures = np.zeros(6)
	mamFeatures[0] = ROI.max()
	mamFeatures[1] = mamFeatures[0]
	maxRow = np.amax(ROI, axis=0)
	maxCol = np.amax(ROI,axis=1)
	sum=0
	for i in range(ROI.shape[1]-1):
		if(maxRow[i]==1):
			i+=1
		else:
			heightCheck=0
			mamFeatures[3]+=1
			for j in range(ROI.shape[0]-1):
				if(maxCol[j]==1):
					j+=1   
				else:
					heightCheck+=1
					checkVal = ROI[j][i]
					if(checkVal>1):
						mamFeatures[5]+=1
						sum+=checkVal
						if(checkVal<mamFeatures[1]):
							mamFeatures[1] = checkVal
				if(heightCheck>mamFeatures[4]):
					mamFeatures[4]=heightCheck
	mamFeatures[2] = sum/mamFeatures[5]
	return(mamFeatures)
#binarize the mass shape
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
	#print(pIDStart.size)
	CCStart = np.copy(pIDStartO)
	MLOStartO = CCStart+1#np.zeros(np.size(pIDStart))#-160)
	#print(np.size(MLOStartO))
	#isolate cases where there is MLO and CID
	#count = 0
	#countM = 0
	i=-1
	for j in range(pIDStartO.size-1):
		i=i+1
		#if (i<np.size(pIDStart)):
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
	arrSize = np.size(pIDStart)
	print('billy')
	CCFull = np.array([pydicom.dcmread('1-1.dcm')]*arrSize)
	CCMask = np.array([pydicom.dcmread('1-1.dcm')]*arrSize)
	MLOFull =np.array([pydicom.dcmread('1-1.dcm')]*arrSize)
	MLOMask =np.array([pydicom.dcmread('1-1.dcm')]*arrSize)
	#load the DICOM files into an an array
	for i in range(arrSize):
		#load in full CC image
		CCFullPath = pathStartNew+CSVFile.at[pIDStart[i],'image file path']
		CCFullPath = CCFullPath.strip('000000.dcm\n')
		CCFullPath = CCFullPath+'1-1.dcm'
		CCFull[i]=pydicom.dcmread(CCFullPath)
		#load in CC mask
		CCMaskPath = pathStartNew+CSVFile.at[pIDStart[i],'ROI mask file path']
		if '0.dcm' in CCMaskPath:
			CCMaskPath = CCMaskPath.strip('000000.dcm\n')
		else:
			CCMaskPath = CCMaskPath.strip('000001.dcm\n')
		try:
			CCMaskPath=CCMaskPath+'1-1.dcm'
			CCMask[i]=pydicom.dcmread(CCMaskPath)
			if(CCMask[i].SeriesDescription!='ROI mask images'):
				CCMaskPath = CCMaskPath.strip('1-1.dcm')
				CCMaskPath = CCMaskPath+'1-2.dcm'
				CCMask[i]=pydicom.dcmread(CCMaskPath)
		except IOError:
			CCMaskPath = CCMaskPath.strip('1-1.dcm\n')
			CCMaskPath=CCMaskPath+'1-2.dcm'
			CCMask[i]=pydicom.dcmread(CCMaskPath)
		#load full MLO image
		MLOFullPath = pathStartNew+CSVFile.at[MLOStart[i],'image file path']
		MLOFullPath = MLOFullPath.strip('000000.dcm\n')
		MLOFullPath=MLOFullPath+'1-1.dcm'
		MLOFull[i]=pydicom.dcmread(MLOFullPath)
		#load in MLO mask
		MLOMaskPath = pathStartNew+CSVFile.at[MLOStart[i],'ROI mask file path']
		if '0.dcm' in MLOMaskPath:
			MLOMaskPath = MLOMaskPath.strip('000000.dcm\n')
		else:
			MLOMaskPath = MLOMaskPath.strip('000001.dcm\n')
		try:
			MLOMaskPath=MLOMaskPath+'1-1.dcm'
			MLOMask[i]=pydicom.dcmread(MLOMaskPath)
			if(MLOMask[i].SeriesDescription!='ROI mask images'):
				MLOMaskPath = MLOMaskPath.strip('1-1.dcm')
				MLOMaskPath = MLOMaskPath+'1-2.dcm'
				MLOMask[i]=pydicom.dcmread(MLOMaskPath)
		except IOError:
			MLOMaskPath=MLOMaskPath.strip('1-1.dcm')
			MLOMaskPath=MLOMaskPath+'1-2.dcm'
			MLOMask[i]=pydicom.dcmread(CCMaskPath)	        
	print('hi')
	count = 0
	MLOROIArr = np.array([MLOFull[1]]*arrSize)
	CCROIArr = np.array([CCFull[1]]*arrSize)
	i=-1
	for j in range(pIDStart.size-1):
		i=i+1
	i=-1
	#exclude entries where the mask and full image are not the same size
	for j in range(arrSize):
		i=i+1
		if (MLOFull[i].Columns!=MLOMask[i].Columns) or (MLOFull[i].Rows!=MLOMask[i].Rows):
			print(MLOFull[i].PatientID)
			pIDStart = np.delete(pIDStart, [i], None)
			MLOIDStart = np.delete(MLOStart, [i], None)
			CCFull = np.delete(CCFull, [i], None)
			CCMask = np.delete(CCMask, [i], None)
			MLOFull = np.delete(MLOFull, [i], None)
			MLOMask = np.delete(MLOMask, [i], None)
			MLOROIArr = np.delete(MLOROIArr, [i], None)
			CCROIArr = np.delete(CCROIArr, [i], None)
			count=count+1
			i=i-1
		elif (CCFull[i].Columns!=CCMask[i].Columns) or (CCFull[i].Rows!=CCMask[i].Rows):
			print(CCFull[i].PatientID)
			pIDStart = np.delete(pIDStart, [i], None)
			MLOIDStart = np.delete(MLOStart, [i], None)
			CCFull = np.delete(CCFull, [i], None)
			CCMask = np.delete(CCMask, [i], None)
			MLOFull = np.delete(MLOFull, [i], None)
			MLOMask = np.delete(MLOMask, [i], None)
			MLOROIArr = np.delete(MLOROIArr, [i], None)
			CCROIArr = np.delete(CCROIArr, [i], None)
			i=i-1
			count=count+1
		else: 
			MLOROIArr[i]=maskImage(MLOFull[i], MLOMask[i])
			CCROIArr[i]=maskImage(CCFull[i], CCMask[i])
			arrSize = np.size(pIDStart)
	print('# of entries had to exclude:')
	print(count)
	arrSize = np.size(MLOROIArr)
	totalMFeatures = np.zeros((arrSize, 43))
	
	#get and combine the features from the metadata and image
	for i in range(arrSize):#-2):
		if('MALIGNANT' in CSVFile.loc[pIDStart[i], 'pathology']):
			totalMFeatures[i][0]=1
		else:
			totalMFeatures[i][0]=0
		totalMFeatures[i][1] = CSVFile.loc[pIDStart[i], 'assessment']
		totalMFeatures[i][2]= CSVFile.loc[pIDStart[i], 'breast_density']
		totalMFeatures[i][3:8] = MarginsBinary(str(CSVFile.loc[pIDStart[i], 'mass margins'])) 
		totalMFeatures[i][8] = CSVFile.loc[pIDStart[i], 'subtlety']
		totalMFeatures[i][9:17] = ShapeBinary(str(CSVFile.loc[pIDStart[i], 'mass shape']))                           
		#totalMFeatures = np.hstack((totalMFeatures, pMetadata))
		totalMFeatures[i][17:22] = MarginsBinary(str(CSVFile.loc[MLOStart[i], 'mass margins']))
		totalMFeatures[i][22] =CSVFile.loc[MLOStart[i], 'subtlety']
		totalMFeatures[i][23:31] = ShapeBinary(str(CSVFile.loc[MLOStart[i], 'mass shape']))
		totalMFeatures[i][31:37] = ROIFeatures(CCROIArr[i])
		totalMFeatures[i][37:43] = ROIFeatures(MLOROIArr[i])
	if(pathStartNew=='CBIS-DDSM'):
		np.savetxt('EngineeredFeaturesTrainNew.csv', totalMFeatures, delimiter=',', fmt="%d")
	else:
		np.savetxt('EngineeredFeaturesTestNew.csv', totalMFeatures, delimiter=',', fmt="%d")
	return 0

pathStartNew = 'CBIS-DDSM/'
CSVFile = pd.read_csv('mass_case_description_train_set.csv')
indicesSetup (pathStartNew, CSVFile)

pathStartNew = 'TestSet/'
CSVFile = pd.read_csv('mass_case_description_train_set.csv')
indicesSetup (pathStartNew, CSVFile)