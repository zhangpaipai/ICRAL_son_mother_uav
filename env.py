import pygame
import math
import sys

fclock = pygame.time.Clock()
fps = 300
size = (width, height) = (900, 700)
# 4个炮管到uav最左端的距离（0是不发射子弹，为了让每个step持续的时间相同）
gun_dict = {0: 256*0.3, 1: 84*0.3, 2: 155*0.3, 3: 355*0.3, 4: 425*0.3}


class Env:
    def __init__(self):
        pygame.init()
        # 屏幕大小
        self.screen = pygame.display.set_mode((900, 700))
        # 屏幕背景为白色
        self.background = (255, 255, 255)
        self.speed = [1,0]
        # 无人机是否左移(1是左移，0为右移)
        self.left = 0
        pygame.display.set_caption("子母式投弹智能决策")
        # uav以及5个目标物体
        self.uav = pygame.transform.rotozoom(pygame.image.load("uav.png"), 0, 0.3)#(图片，旋转角度，缩放倍数)
        self.helicopter = pygame.image.load("helicopter.jpeg")#(100,69)
        self.house1 = pygame.image.load("house1.jpeg")#(100,88)
        self.house2 = pygame.image.load("house2.jpeg")#(100,100)
        self.house3 = pygame.image.load("house3.jpeg")#(100,100)
        self.enemy = pygame.image.load("enemy.jpeg")#(100,67)
        # 炸弹初始化
        self.bullet = pygame.image.load("bullet.png")#(,)
        # uav的矩形框
        self.uav_rect = self.uav.get_rect()
        # uav的矩形长度
        self.uav_length = self.uav_rect.width
        # bullet的矩形框
        self.bullet_rect = self.bullet.get_rect()
        """ # uav的4个炮管的横坐标
        self.gun_1 = self.uav_rect.left + round(84*0.3)
        self.gun_2 = self.uav_rect.left + round(155*0.3)
        self.gun_3 = self.uav_rect.left + round(355*0.3)
        self.gun_4 = self.uav_rect.left + round(425*0.3) """
    
    # 计算中点横坐标
    def middle(self, surface):
        return surface.left + surface.width/2
    
    # 实时计算4个炮管横坐标
    def cal_dis(self, i):
        dis = self.uav_rect.left + round(gun_dict[i])
        return dis

    def render(self):
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        self.screen.fill(self.background)
        self.screen.blit(self.uav, self.uav_rect)
        self.screen.blit(self.helicopter, (0,600))
        self.screen.blit(self.house1, (200,600))
        self.screen.blit(self.house2, (400,600))
        self.screen.blit(self.house3, (600,600))
        self.screen.blit(self.enemy, (800,600))
            
        self.uav_rect = self.uav_rect.move(self.speed[0], self.speed[1])

        if self.uav_rect.left < 0 or self.uav_rect.right > width:
            self.speed[0] = - self.speed[0]
            
        self.screen.blit(self.uav, self.uav_rect)
        # self.screen.blit(self.bullet, self.bullet_rect)
        # 更新显示屏幕
        pygame.display.update()

    def reset(self):
        # uav的矩形框位置为屏幕左上角(0,0)
        self.uav_rect.left = 0
        self.uav_rect.top = 0
        """ while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

            self.screen.fill(self.background)
            self.screen.blit(self.uav, self.uav_rect)
            self.screen.blit(self.helicopter, (0,600))
            self.screen.blit(self.house1, (200,600))
            self.screen.blit(self.house2, (400,600))
            self.screen.blit(self.house3, (600,600))
            self.screen.blit(self.enemy, (800,600))
            
            self.uav_rect = self.uav_rect.move(self.speed[0], self.speed[1])

            if self.uav_rect.left < 0 or self.uav_rect.right > width:
                self.speed[0] = - self.speed[0]
            
            self.screen.blit(self.uav, self.uav_rect)
            # 更新显示屏幕
            pygame.display.update()
            # 每秒更新fps次（频率）
            fclock.tick(fps) """
        # state为uav中点到五个目标的距离
        state = [self.uav_rect.midbottom[0]-50, self.uav_rect.midbottom[0]-250, self.uav_rect.midbottom[0]-450, self.uav_rect.midbottom[0]-650, self.uav_rect.midbottom[0]-850]
            
        return state
        
    
    def step(self, action):
        
        done = False
        # 不投弹
        if action==0:
            reward = -0.1
            self.move_bullet(0)
        # 炮管1\2\3\4投弹
        else:
            # 击中总部house2
            if (self.cal_dis(action)>400 and self.cal_dis(action)<500):
                reward = 2
            # 击中其他目标
            elif (self.cal_dis(action)>0 and self.cal_dis(action)<100)or(self.cal_dis(action)>200 and self.cal_dis(action)<300)or(self.cal_dis(action)>600 and self.cal_dis(action)<700)or(self.cal_dis(action)>800 and self.cal_dis(action)<900):
                reward = 0.5
            # 未击中目标
            else:
                reward = -1
            self.move_bullet(action)
        """ # 炮管2投弹
        elif action==2:
            # 击中总部house2
            if (self.gun_2>400 and self.gun_2<500):
                reward = 2
            # 击中其他目标
            elif (self.gun_2>0 and self.gun_2<100)or(self.gun_2>200 and self.gun_2<300)or(self.gun_2>600 and self.gun_2<700)or(self.gun_2>800 and self.gun_2<900):
                reward = 0.5
            # 未击中目标
            else:
                reward = -1
            self.move_bullet(2)
        # 炮管3投弹
        elif action==3:
            # 击中总部house2
            if (self.gun_3>400 and self.gun_3<500):
                reward = 2
            # 击中其他目标
            elif (self.gun_3>0 and self.gun_3<100)or(self.gun_3>200 and self.gun_3<300)or(self.gun_3>600 and self.gun_3<700)or(self.gun_3>800 and self.gun_3<900):
                reward = 0.5
            # 未击中目标
            else:
                reward = -1
            self.move_bullet(3)
        # 炮管4投弹
        elif action==4:
            # 击中总部house2
            if (self.gun_4>400 and self.gun_4<500):
                reward = 2
            # 击中其他目标
            elif (self.gun_4>0 and self.gun_4<100)or(self.gun_4>200 and self.gun_4<300)or(self.gun_4>600 and self.gun_4<700)or(self.gun_4>800 and self.gun_4<900):
                reward = 0.5
            # 未击中目标
            else:
                reward = -1
            self.move_bullet(4) """
        next_state = [self.uav_rect.midbottom[0]-50, self.uav_rect.midbottom[0]-250, self.uav_rect.midbottom[0]-450, self.uav_rect.midbottom[0]-650, self.uav_rect.midbottom[0]-850]

        return next_state, reward, done
    
    def move_bullet(self, i):    
        # 子弹下落
        # bullet_rect = self.bullet.get_rect()
        self.bullet_rect.center = (self.uav_rect.left+gun_dict[i],self.uav_rect.height) 
        while True:
            self.screen.fill(self.background)
            self.screen.blit(self.uav, self.uav_rect)
            self.screen.blit(self.helicopter, (0,600))
            self.screen.blit(self.house1, (200,600))
            self.screen.blit(self.house2, (400,600))
            self.screen.blit(self.house3, (600,600))
            self.screen.blit(self.enemy, (800,600))
                
            """ self.uav_rect = self.uav_rect.move(self.speed[0], self.speed[1])

            if self.uav_rect.left < 0 or self.uav_rect.right > width:
                self.speed[0] = - self.speed[0] """
            if self.left == 0:
                self.uav_rect.x += 1
            else:
                self.uav_rect.x -= 1
            
            if self.uav_rect.right > width:
                self.left = 1
                self.uav_rect.x -= 1
            if self.uav_rect.left < 0:
                self.left = 0
                self.uav_rect.x += 1

            self.bullet_rect = self.bullet_rect.move(2*self.speed[1], 2*self.speed[0])
            if self.bullet_rect.bottom > 700 :
                break    
            self.screen.blit(self.uav, self.uav_rect)
            if i != 0:
                self.screen.blit(self.bullet, self.bullet_rect)
            pygame.display.update()