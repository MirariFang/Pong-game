import TrainSmall
import TrainBig
import gui
if __name__ == '__main__':
    # for i in range(10):
    #     if i == 0:
    #         continue
    #     result = TrainSmall.train(100000, 1000, i / 10.0, 5, 10)
    #     print(result)
    result = TrainSmall.train(100000, 1000, 0.4, 8, 21)
    print(result, flush=True)
    
    gui.draw() # Animation of the pong game