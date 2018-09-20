## seg length [4,8,16,20,32,48,56,64], no slide window
#python featureVectorGenerator.py 1 -segLen 4  -lag 4  -isAcc 1 -isRotationRate 1 -isBrightness 1 -isPressure 1 -isMagnetic 1 -isNormalize 1 -isAddUID 0
#python featureVectorGenerator.py 1 -segLen 8  -lag 8  -isAcc 1 -isRotationRate 1 -isBrightness 1 -isPressure 1 -isMagnetic 1 -isNormalize 1 -isAddUID 0
#python featureVectorGenerator.py 1 -segLen 16 -lag 16 -isAcc 1 -isRotationRate 1 -isBrightness 1 -isPressure 1 -isMagnetic 1 -isNormalize 1 -isAddUID 0
#python featureVectorGenerator.py 1 -segLen 20 -lag 20 -isAcc 1 -isRotationRate 1 -isBrightness 1 -isPressure 1 -isMagnetic 1 -isNormalize 1 -isAddUID 0
#python featureVectorGenerator.py 1 -segLen 32 -lag 32 -isAcc 1 -isRotationRate 1 -isBrightness 1 -isPressure 1 -isMagnetic 1 -isNormalize 1 -isAddUID 0
#python featureVectorGenerator.py 1 -segLen 48 -lag 48 -isAcc 1 -isRotationRate 1 -isBrightness 1 -isPressure 1 -isMagnetic 1 -isNormalize 1 -isAddUID 0
#python featureVectorGenerator.py 1 -segLen 56 -lag 56 -isAcc 1 -isRotationRate 1 -isBrightness 1 -isPressure 1 -isMagnetic 1 -isNormalize 1 -isAddUID 0
#python featureVectorGenerator.py 1 -segLen 64 -lag 64 -isAcc 1 -isRotationRate 1 -isBrightness 1 -isPressure 1 -isMagnetic 1 -isNormalize 1 -isAddUID 0

## seg length = 16, slide from 2,4,8,10,12
#python featureVectorGenerator.py 1 -segLen 16 -lag 2  -isAcc 1 -isRotationRate 1 -isBrightness 1 -isPressure 1 -isMagnetic 1 -isNormalize 1 -isAddUID 0
#python featureVectorGenerator.py 1 -segLen 16 -lag 4  -isAcc 1 -isRotationRate 1 -isBrightness 1 -isPressure 1 -isMagnetic 1 -isNormalize 1 -isAddUID 0
#python featureVectorGenerator.py 1 -segLen 16 -lag 6  -isAcc 1 -isRotationRate 1 -isBrightness 1 -isPressure 1 -isMagnetic 1 -isNormalize 1 -isAddUID 0
#python featureVectorGenerator.py 1 -segLen 16 -lag 8  -isAcc 1 -isRotationRate 1 -isBrightness 1 -isPressure 1 -isMagnetic 1 -isNormalize 1 -isAddUID 0
#python featureVectorGenerator.py 1 -segLen 16 -lag 10 -isAcc 1 -isRotationRate 1 -isBrightness 1 -isPressure 1 -isMagnetic 1 -isNormalize 1 -isAddUID 0
#python featureVectorGenerator.py 1 -segLen 16 -lag 12 -isAcc 1 -isRotationRate 1 -isBrightness 1 -isPressure 1 -isMagnetic 1 -isNormalize 1 -isAddUID 0


## todo: different combination of sensors

## single sensors
#python featureVectorGenerator.py 1 -segLen 16 -lag 16 -isAcc 1 -isRotationRate 0 -isBrightness 0 -isPressure 0 -isMagnetic 0 -isNormalize 1 -isAddUID 0
#python featureVectorGenerator.py 1 -segLen 16 -lag 16 -isAcc 0 -isRotationRate 1 -isBrightness 0 -isPressure 0 -isMagnetic 0 -isNormalize 1 -isAddUID 0
#python featureVectorGenerator.py 1 -segLen 16 -lag 16 -isAcc 0 -isRotationRate 0 -isBrightness 1 -isPressure 0 -isMagnetic 0 -isNormalize 1 -isAddUID 0
#python featureVectorGenerator.py 1 -segLen 16 -lag 16 -isAcc 0 -isRotationRate 0 -isBrightness 0 -isPressure 1 -isMagnetic 0 -isNormalize 1 -isAddUID 0
#python featureVectorGenerator.py 1 -segLen 16 -lag 16 -isAcc 0 -isRotationRate 0 -isBrightness 0 -isPressure 0 -isMagnetic 1 -isNormalize 1 -isAddUID 0

## COmbinations: 2 sensors:
#### a + r
#python featureVectorGenerator.py 1 -segLen 16 -lag 16 -isAcc 1 -isRotationRate 1 -isBrightness 0 -isPressure 0 -isMagnetic 0 -isNormalize 1 -isAddUID 0
#### a + b
#python featureVectorGenerator.py 1 -segLen 16 -lag 16 -isAcc 1 -isRotationRate 0 -isBrightness 1 -isPressure 0 -isMagnetic 0 -isNormalize 1 -isAddUID 0
#### a + p
#python featureVectorGenerator.py 1 -segLen 16 -lag 16 -isAcc 1 -isRotationRate 0 -isBrightness 0 -isPressure 1 -isMagnetic 0 -isNormalize 1 -isAddUID 0
#### a + m
#python featureVectorGenerator.py 1 -segLen 16 -lag 16 -isAcc 1 -isRotationRate 0 -isBrightness 0 -isPressure 0 -isMagnetic 1 -isNormalize 1 -isAddUID 0
#### r + b
#python featureVectorGenerator.py 1 -segLen 16 -lag 16 -isAcc 0 -isRotationRate 1 -isBrightness 1 -isPressure 0 -isMagnetic 0 -isNormalize 1 -isAddUID 0
#### r + p
#python featureVectorGenerator.py 1 -segLen 16 -lag 16 -isAcc 0 -isRotationRate 1 -isBrightness 0 -isPressure 1 -isMagnetic 0 -isNormalize 1 -isAddUID 0
#### r + m
#python featureVectorGenerator.py 1 -segLen 16 -lag 16 -isAcc 0 -isRotationRate 1 -isBrightness 0 -isPressure 0 -isMagnetic 1 -isNormalize 1 -isAddUID 0
#### b + p
#python featureVectorGenerator.py 1 -segLen 16 -lag 16 -isAcc 0 -isRotationRate 0 -isBrightness 1 -isPressure 1 -isMagnetic 0 -isNormalize 1 -isAddUID 0
#### b + m
#python featureVectorGenerator.py 1 -segLen 16 -lag 16 -isAcc 0 -isRotationRate 0 -isBrightness 1 -isPressure 0 -isMagnetic 1 -isNormalize 1 -isAddUID 0
#### p + m
#python featureVectorGenerator.py 1 -segLen 16 -lag 16 -isAcc 0 -isRotationRate 0 -isBrightness 0 -isPressure 1 -isMagnetic 1 -isNormalize 1 -isAddUID 0
#
## Combination of 3 sensors:
#### a + r + b
#python featureVectorGenerator.py 1 -segLen 16 -lag 16 -isAcc 1 -isRotationRate 1 -isBrightness 1 -isPressure 0 -isMagnetic 0 -isNormalize 1 -isAddUID 0
#### a + b + p
#python featureVectorGenerator.py 1 -segLen 16 -lag 16 -isAcc 1 -isRotationRate 0 -isBrightness 1 -isPressure 1 -isMagnetic 0 -isNormalize 1 -isAddUID 0
#### a + p + m
#python featureVectorGenerator.py 1 -segLen 16 -lag 16 -isAcc 1 -isRotationRate 0 -isBrightness 0 -isPressure 1 -isMagnetic 1 -isNormalize 1 -isAddUID 0
#### a + m + r
#python featureVectorGenerator.py 1 -segLen 16 -lag 16 -isAcc 1 -isRotationRate 1 -isBrightness 0 -isPressure 0 -isMagnetic 1 -isNormalize 1 -isAddUID 0
#### r + b + p
#python featureVectorGenerator.py 1 -segLen 16 -lag 16 -isAcc 0 -isRotationRate 1 -isBrightness 1 -isPressure 1 -isMagnetic 0 -isNormalize 1 -isAddUID 0
#### r + p + m 
#python featureVectorGenerator.py 1 -segLen 16 -lag 16 -isAcc 0 -isRotationRate 1 -isBrightness 0 -isPressure 1 -isMagnetic 1 -isNormalize 1 -isAddUID 0
#### r + m + b
#python featureVectorGenerator.py 1 -segLen 16 -lag 16 -isAcc 0 -isRotationRate 1 -isBrightness 1 -isPressure 0 -isMagnetic 1 -isNormalize 1 -isAddUID 0
#### b + p + m
#python featureVectorGenerator.py 1 -segLen 16 -lag 16 -isAcc 0 -isRotationRate 0 -isBrightness 1 -isPressure 1 -isMagnetic 1 -isNormalize 1 -isAddUID 0
#### b + m + a
#python featureVectorGenerator.py 1 -segLen 16 -lag 16 -isAcc 1 -isRotationRate 0 -isBrightness 1 -isPressure 0 -isMagnetic 1 -isNormalize 1 -isAddUID 0
#### a + r + p
#python featureVectorGenerator.py 1 -segLen 16 -lag 16 -isAcc 1 -isRotationRate 1 -isBrightness 0 -isPressure 1 -isMagnetic 0 -isNormalize 1 -isAddUID 0
#

### Combination of 4 sensors:
#### a + r + b + p
#python featureVectorGenerator.py 1 -segLen 16 -lag 16 -isAcc 1 -isRotationRate 1 -isBrightness 1 -isPressure 1 -isMagnetic 0 -isNormalize 1 -isAddUID 0
#### a + b + p + m
#python featureVectorGenerator.py 1 -segLen 16 -lag 16 -isAcc 1 -isRotationRate 0 -isBrightness 1 -isPressure 1 -isMagnetic 1 -isNormalize 1 -isAddUID 0
#### r + b + p + m 
#python featureVectorGenerator.py 1 -segLen 16 -lag 16 -isAcc 0 -isRotationRate 1 -isBrightness 1 -isPressure 1 -isMagnetic 1 -isNormalize 1 -isAddUID 0
#### a + m + r + p
#python featureVectorGenerator.py 1 -segLen 16 -lag 16 -isAcc 1 -isRotationRate 1 -isBrightness 0 -isPressure 1 -isMagnetic 1 -isNormalize 1 -isAddUID 0
#### r + b + a + m
#python featureVectorGenerator.py 1 -segLen 16 -lag 16 -isAcc 1 -isRotationRate 1 -isBrightness 1 -isPressure 0 -isMagnetic 1 -isNormalize 1 -isAddUID 0

#### ADD UID:
#python featureVectorGenerator.py 1 -segLen 16 -lag 16 -isAcc 1 -isRotationRate 1 -isBrightness 1 -isPressure 1 -isMagnetic 1 -isNormalize 1 -isAddUID 1 -isAndroid 1

#python featureVectorGenerator.py 1 -segLen 16 -lag 16 -isAcc 1 -isRotationRate 1 -isBrightness 0 -isPressure 1 -isMagnetic 1 -isNormalize 1 -isAddUID 1 -isAndroid 0

### deprecated above. 
## now we use json configuration file. The only arg that needs to pass in is the configuration file name

python featureVectorGenerator.py -config 'config.json'

##
