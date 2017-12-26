from keras.models import Sequential#used for neural network sequence of layers from the image
from keras.layers import Convolution2D#used for convolution layers imput:image output:feature maps
from keras.layers import MaxPooling2D#input feature map output:pooled feature map
from keras.layers import Flatten#pooled feature map into vector
from keras.layers import Dense

classifier=Sequential()#create object for sequential

classifier.add(Convolution2D(20,3,3,input_shape=(128,128,3), activation='relu'))
#first Convolution2D(32,3,3----32 feature detectors of size 3 by 3
#second input_shape=(64,64,3)----is the size of input image (for tensorflow syntax(64,64,3) 64*64 pixels of feature detector and 3 as rgb colours )
#third argument in rectified activation function to avoid linearity of image


#pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))
#will help to reduce size of feature maps,(2,2) will halv input 
#size of pool is usually half of feature map. so if feature map is ceiling(5*5/2)=3*3 

classifier.add(Convolution2D(32,3,3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#flattening
classifier.add(Flatten())

#construction of neural network
classifier.add(Dense(output_dim=128,activation = 'relu')) #no of nodes in hidden layer(output layer) is usually chosen between no of input nodes and output nodes 128 is random value
classifier.add(Dense(output_dim=1,activation = 'sigmoid'))#activation function for final output#imp...if have output as more than two categories you need to use softmax activation function

#compiling
classifier.compile(optimizer = 'adam', loss= 'binary_crossentropy', metrics=['accuracy'])
#if more than two outputs than  loss=categorical_crossentropy

##image augmentation

from keras.preprocessing.image import ImageDataGenerator


#augmented code from keras documentation in image processing
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255) 

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(128, 128),#from above input shape
                                                    batch_size=10,
                                                    class_mode='binary')#only because it is dog or cat

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000, #because 8000 images
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)#2000 test sets 
