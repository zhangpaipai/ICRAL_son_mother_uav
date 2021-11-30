import pygame
import sys

pygame.init()

WHITE = (255, 255, 255)
speed = [1,0]
fclock = pygame.time.Clock()
fps = 500
size = (width, height) = (900, 700)
screen = pygame.display.set_mode(size)

pygame.display.set_caption("子母式投弹智能决策")
uav = pygame.image.load("uav.png")#(512,512)

uav = pygame.transform.rotozoom(uav, 0, 0.3)
helicopter = pygame.image.load("helicopter.jpeg")#(100,69)
house1 = pygame.image.load("house1.jpeg")#(100,88)
house2 = pygame.image.load("house2.jpeg")#(100,100)
house3 = pygame.image.load("house3.jpeg")#(100,100)
enemy = pygame.image.load("enemy.jpeg")#(100,67)
bullet = pygame.image.load("bullet.png")#(,)
""" print(uav.get_size())
print(helicopter.get_size())
print(house1.get_size())
print(house2.get_size())
print(house3.get_size())
print(enemy.get_size()) """

uav_rect = uav.get_rect()
uav_rect.left = 0
uav_rect.top = 0
bullet_rect = bullet.get_rect()
#bullet_rect.left = 84*0.3
#bullet_rect.top = uav_rect.height
bullet_rect.center = (84*0.3,uav_rect.height)

print(uav_rect.width)#153
print(bullet_rect.width)#22
print(uav_rect.midbottom[0])#22
print(uav_rect.bottom)
left = 0

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    screen.fill(WHITE)
    screen.blit(uav, uav_rect)
    screen.blit(helicopter, (0,600))
    screen.blit(house1, (200,600))
    screen.blit(house2, (400,600))
    screen.blit(house3, (600,600))
    screen.blit(enemy, (800,600))
    screen.blit(bullet, bullet_rect)

    #uav_rect = uav_rect.move(speed[0], speed[1])
    if left == 0:
        uav_rect.x += 1
    else:
        uav_rect.x -= 1
    # bullet_rect = bullet_rect.move(5*speed[1], 5*speed[0])

    #if uav_rect.left < 0 or uav_rect.right > width:
    #    speed[0] = - speed[0]
    if uav_rect.right > width:
        left = 1
        uav_rect.x -= 1
    if uav_rect.left < 0:
        left = 0
        uav_rect.x += 1
    
    bullet_rect = bullet_rect.move(2*speed[1], 2*speed[0])
    if bullet_rect.bottom > 700 :
        bullet_rect.center = (uav_rect.left+84*0.3,uav_rect.height)
    
    screen.blit(uav, uav_rect)
    
    pygame.display.update()
    fclock.tick(fps)
    # pygame.time.wait(18000)