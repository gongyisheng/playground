import math

# https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a/
def log_loss():
    label = 1
    for pos_prob in range(1, 10):
        pos_prob = pos_prob / 10
        loss = (label*math.log(pos_prob) + (1 - label)*math.log(1-pos_prob))*(-1)
        print(pos_prob, loss)

if __name__ == "__main__":
    log_loss()