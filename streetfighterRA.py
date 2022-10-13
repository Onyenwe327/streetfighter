# Import retro to play Street Fighter using a ROM
import retro
# Import time to slow down game
import time
# Starts up the game environment
env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')
#Reset environment
obs = env.reset()
done = False
for game in range(1): 
    while not done: 
        if done: 
            obs = env.reset()
        env.render()
        obs, reward, done, info = env.step(env.action_space.sample())
        time.sleep(0.001)
        print(reward)

info