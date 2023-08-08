import keyboard

def abc(x):
    if x.event_type == 'down':
        print(x.event_type, x.name, x.scan_code, x.time)

def listen():
    keyboard.hook(abc)
    keyboard.wait()

if __name__ == "__main__":
    listen()
