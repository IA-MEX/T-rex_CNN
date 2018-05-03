import numpy as np
import pyautogui
import win32api as wapi
import pyautogui
import cv2
from PIL import ImageGrab

# Constants:
# GAME_OVER --> file to get the game over

# Beating t-rex Google Chrome Game with DRL

# Actions:
#  1->Nothing
#  0->Jump!!!

# Reward: time

GAME_OVER='game_over.png'
from keras.models import Sequential
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Activation, Flatten, AveragePooling2D
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.utils import np_utils

model=Sequential()
def createModel():
    #create the model, the same of playing atari games of deepmind
    

    model.add(Convolution2D(32,(8,8),activation='relu', input_shape=(80,80,1)))
    model.add(MaxPooling2D(pool_size=(3,3)))
    
    model.add(Convolution2D(32,(4,4),activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3)))
    
    model.add(Convolution2D(64,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))

    model.add(Dense(2, activation='softmax'))

    adam=Adam()
    model.compile(loss='mean_squared_error',optimizer=adam)
Xmodel_Evaluation=[]
Ymodel_Evaluation=[]


def screen_record(): 
    # wapi.GetKeyState change between 0-1 when a key is pressed, save the current status
    actual=wapi.GetKeyState(38)
    #image list of the episode
    img_list=[]
    #actions took of the images
    A=[]
    #click to start a game one for activate the window, one more fore start the game
    pyautogui.click(x=200,y=350)
    pyautogui.click(x=200,y=350)
    # reward, the number of frames in the episode
    reward=0
    #Take the first frame of the game, resize and convert to grayscale
    previous_1=cv2.cvtColor(np.array(ImageGrab.grab(bbox=(180,310,370,500))),cv2.COLOR_BGR2GRAY)
    #resize for the input of the neural network
    previous_1=cv2.resize(previous_1,(80,80),interpolation=cv2.INTER_CUBIC)
    #increment the reward
    reward+=1
    #number of episode to train the CNN
    episodios=0
    #Loop forever
    while(True):
        #Current frame
        current = cv2.cvtColor(np.array(ImageGrab.grab(bbox=(180,310,370,500))),cv2.COLOR_BGR2GRAY)
        #resize the frame 
        current = cv2.resize(current,(80,80),interpolation=cv2.INTER_CUBIC)
        #find if the up key is pressed
        pressed=wapi.GetKeyState(38)
        #if the pressed is different of the current and > to 0
        #apend 0 to actions and add the frame to the list
        if pressed!=actual and pressed>=0:
            #Normalize the image
            img_list.append(np.divide(previous_1,255))
            A.append(0)
            actual=wapi.GetKeyState(38)
        #if game over is in the screen train the model
        elif pyautogui.locateOnScreen(GAME_OVER)!=None:
            #print the episode and the reward
            print ('Ended episode',episodios,", Reward:",reward-1)
            #increment the episodios
            episodios+=1
            #if the episodes are equal to XX=10 train
            if episodios==10:
                #erase the 3 final frames and actions of the episode
                for i in range(3):
                    img_list.pop()
                    A.pop()
                #convert the image list to np array and expand the dim
                img_data=np.array(img_list)
                img_data=np.expand_dims(img_data,axis=4)
                #convert action (classes) to categorical
                actions=np_utils.to_categorical(A,2)
                #train and save the model
                model.fit(img_data,actions,epochs=200)
                model.save('model_E'+str(episodios)+'.h5')
                #restart the image list and actions
                img_list=[]
                A=[]
                #for one loop: break, if not erase the break;
                break;
            #erase the 3 final frames and actions of the episode
            else:
                for i in range(3):
                    img_list.pop()
                    A.pop()
            pyautogui.click(x=200,y=350)
        #if the action is none append 0
        else:
            A.append(1)
            img_list.append(np.divide(previous_1,255))
        previous_1=np.copy(current)
        reward+=1


createModel()
screen_record()