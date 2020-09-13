import numpy as np
from itertools import chain
from predict import Predictor
from load_model import Model
from PIL import ImageEnhance, Image
import os
import shutil
import glob
# m = Model()
# model = m.load_model()

class IMAGE: # Environment
    def __init__(self):
        self.t_step=0
        self.max_step=10
        self.done=False
        self.goal = 0.6
        # self.t_action=['U','R','L','D']

        #0:do noting, 1:contrast(1.1), 2:contrast(0.9), 3:saturation(1.1), 4:saturation(0.9), 
        # 5:exposure(1.1), 6:exposure(0.9)
        self.t_action=["0","1","2","3","4","5","6"]
        self.actions={}

        self.get_reward = Predictor()

        #それぞれのマスに行動振ってる
        # for i in range(self.height):
        #     for j in range(self.width):
        #         if self.goal!=(i,j):
        #             if  self.breaker!=(i,j):
        #                 self.actions[(i,j)]=[]


        #Should be modified

        # self.rewards= {self.goal:g_reward}


        # for k,v in self.actions.items():       
        #     for a in range(len(self.t_action)):       
        #         cur_n=self.state_n([k[0],k[1]],self.t_action[a]) 
        #         if not((cur_n[0]>=self.height or cur_n[0]<0) or (cur_n[1]>=self.width or cur_n[1]<0)):
        #             if (self.goal==cur_n or self.breaker==cur_n)==0:
        #                 self.actions[k].append(self.t_action[a])

        # self.state=self.getstate(episode)

    # def state_n(self,image,action):
    #     if action == 0:
    #         pass
    #     elif action == 1:
    #         image = ImageEnhance.Contrast(image)
    #         image = image.enhance(1.1)
    #     elif action == 2:
    #         image = ImageEnhance.Contrast(image)
    #         image = image.enhance(0.9)
    #     elif action == 3:
    #         image = ImageEnhance.Color(image)
    #         image = image.enhance(1.1)
    #      elif action == 4:
    #         image = ImageEnhance.Color(image)
    #         image = image.enhance(0.9)
    #     elif action == 5:
    #         image = ImageEnhance.Brightness(image)
    #         image = image.enhance(1.1)
    #     elif action == 6:
    #         image = ImageEnhance.Brightness(image)
    #         image = image.enhance(0.9)
    #     return image
        
    # def set(self, s):
    #     self.i = s[0]
    #     self.j = s[1]

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
        # state=[]
        # for key,_ in self.all_states().items():
        #     if key==(self.i,self.j):
        #         state.append(1)
        #     else:
        #         state.append(0)
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
        if terminate and Reward==0:
            Reward=-0.1
        self.t_step+=1
        return [state,Reward,terminate]

    # def current_state(self):
    #     return self.getstate()

    # def image(self):
    #     return (self.i,self.j)


    def move(self, action,episode,model):
        # check if legal move first
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

    # def undo_move(self, action):
    #     if action == 'U':
    #       self.i += 1
    #     elif action == 'D':
    #       self.i -= 1
    #     elif action == 'R':
    #       self.j -= 1
    #     elif action == 'L':
    #       self.j += 1
    #     assert(self.current_state() in self.all_states())

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

    # def all_states(self):

    #     return dict(chain.from_iterable(d.items() for d in (self.actions, self.rewards)))

    # def draw_board(self):
    #     board = []
    #     for i in range(self.height):
    #         for j in range(self.width):
    #             if self.current_state()==[i,j]:
    #                 board.append("!")
    #             elif self.start==(i,j):
    #                 board.append("S")
    #             elif self.goal==(i,j):
    #                 board.append("G")
    #             elif self.breaker==(i,j):
    #                 board.append("K")
    #             else:
    #                 board.append(" ")
    #     print(" "*15, ".....................")
    #     print(" "*15,"|","".join(board[0]), " |", "".join(board[1]), " |", "".join(board[2])," |","".join(board[3])," |")
    #     print(" "*15,"|----|----|----|----|")
    #     print(" "*15,"|","".join(board[4]), " |", "".join(board[5]), " |", "".join(board[6])," |","".join(board[7])," |")
    #     print(" "*15,"|----|----|----|----|")
    #     print(" "*15,"|","".join(board[8]), " |", "".join(board[9]), " |", "".join(board[10])," |","".join(board[11])," |")

    #     print(" "*15, "''''''''''''''''''''")