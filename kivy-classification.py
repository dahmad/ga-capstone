from kivy.app import App
from kivy.core.audio import SoundLoader
from kivy.core.window import Window, Keyboard
from kivy.uix.widget import Widget
import numpy as np
from glob import glob
import time
import pandas as pd

files = glob('01/*.wav')
files = sorted(files)

clean_titles = glob('01/*.wav')
for x in range(len(files)):
    clean_titles[x] = sorted(glob('01/*.wav'))[x][46:]

counter = 0
classification = []

class MyKeyboardListener(Widget):

    def __init__(self, **kwargs):
        super(MyKeyboardListener, self).__init__(**kwargs)
        self._keyboard = Window.request_keyboard(
            self._keyboard_closed, self, 'text')
        if self._keyboard.widget:
            pass
        self._keyboard.bind(on_key_down=self._on_keyboard_down)

    def _keyboard_closed(self):
        print('My keyboard have been closed!')
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 'escape':
            keyboard.release()

        if keycode[1] == 'a':
            global counter
            global classification 
            try:
            	sound = SoundLoader.load(files[counter])
            except:
            	df = pd.DataFrame(classification, columns=['Selection'])
            	df.to_csv('results.csv')
            sound.play()
            time.sleep(sound.length)
            sound.play()
            time.sleep(sound.length)
            sound.play()
            time.sleep(sound.length)
            sound.play()
            time.sleep(sound.length)
            counter += 1

        if keycode[1] == ']':
        	classification.append((1))
        	print(counter, 1)

        if keycode[1] == '[':
        	classification.append((0))
        	print(counter, 0)

        return True

if __name__ == '__main__':
    from kivy.base import runTouchApp
    runTouchApp(MyKeyboardListener())