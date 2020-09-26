import numpy as np
from itertools import chain
from predict import Predictor
from load_model import Model
from PIL import ImageEnhance, Image
import os
import shutil
import glob


class IMAGE: # Environment
    def __init__(self):
        self.t_step=0
        self.max_step=10
        self.done=False
        self.goal = 0.6

        #0:do noting, 1:contrast(1.1), 2:contrast(0.9), 3:saturation(1.1), 4:saturation(0.9), 
        # 5:exposure(1.1), 6:exposure(0.9)
        self.t_action=["0","1","2","3","4","5","6"]
        self.actions={}

        self.get_reward = Predictor()


    def resets(self,episode):
        os.makedirs("test_assets/{}".format(str(episode)), exist_ok=True)
        shutil.copyfile("./test_assets/original/target.jpg", "./test_assets/{}/target.jpg".format(episode))
        files = glob.glob("test_assets/{}/*.jpg".format(episode))
        X = []
        image = Image.open(files[0])
        image = image.convert("RGB")
        image = image.resize((50, 50))
        data = np.asarray(image)
        X.append(data)
        
        
        X = np.array(X)

        X = X.astype('float32')
        state = X / 255.0
        self.__init__()
        self.done=False
        self.state = state.reshape(-1)
        self.t_step=0

    def getstate(self,episode):
        X = []
        image = Image.open("test_assets/{}/target.jpg".format(episode))
        image = image.convert("RGB")
        image = image.resize((50,50))
        data = np.asarray(image)
        X.append(data)

        X = np.array(X)
        X = X.astype('float32')
        state = X / 255.0


        return state

    def step(self,action,episode,model):
        Reward=self.move(action,episode,model)
        state=self.getstate(episode)
        terminate=self.game_over(Reward)
        #ペナルティの設計する必要あり
        # if terminate and Reward<0.5:
        #     Reward=-0.1
        self.t_step+=1
        return [state,Reward,terminate]

    def move(self, action,episode,model):
        action=self.t_action[action]
        image = Image.open("test_assets/{}/target.jpg".format(episode))

        print(action)

        if action == "0":
            pass
        elif action == "1":
            image = ImageEnhance.Contrast(image)
            image = image.enhance(1.1)
        elif action == "2":
            image = ImageEnhance.Contrast(image)
            image = image.enhance(0.9)
        elif action == "3":
            image = ImageEnhance.Color(image)
            image = image.enhance(1.1)
        elif action == "4":
            image = ImageEnhance.Color(image)
            image = image.enhance(0.9)
        elif action == "5":
            image = ImageEnhance.Brightness(image)
            image = image.enhance(1.1)
        elif action == "6":
            image = ImageEnhance.Brightness(image)
            image = image.enhance(0.9)

        image.save("test_assets/{}/target.jpg".format(episode))

        return self.get_reward.predict(model,episode)


    def game_over(self,Reward):
        if self.t_step<self.max_step:
            if self.goal<=Reward:
                self.done=True
                return self.done
            else:
                return self.done
        else:
            self.done=True
            return self.done
