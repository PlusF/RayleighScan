from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import StringProperty, ObjectProperty
from kivy.uix.popup import Popup
from kivy_garden.graph import Graph, LinePlot
from kivy.clock import Clock
from kivy.config import Config

Config.set('graphics', 'width', '1000')
Config.set('graphics', 'height', '500')

import os
import numpy as np
import datetime
import time


class SaveDialog(FloatLayout):
    save = ObjectProperty(None)
    text_input = ObjectProperty(None)
    cancel = ObjectProperty(None)
    folder = StringProperty('')


class YesNoDialog(FloatLayout):
    message = ObjectProperty(None)
    yes = ObjectProperty(None)
    cancel = ObjectProperty(None)


class RASDriver(BoxLayout):
    start_pos = ObjectProperty(np.empty(3))
    current_pos = ObjectProperty(np.empty(3))
    goal_pos = ObjectProperty(np.empty(3))
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.xdata = np.empty(100)
        self.ydata = np.empty(100)
        self.coord = np.empty(3)
        self.integration = 0
        self.accumulation = 0
        self.interval = 0
        self.folder = './'

        self.graph = Graph(
            xlabel = 'Wavelength [nm]', ylabel = 'Counts',
            xmin=0, xmax=1000, ymin=0, ymax=1000,
            x_ticks_minor = 2, x_ticks_major = 100, y_ticks_minor = 2, y_ticks_major = 100,
            y_grid_label = True, x_grid_label = True,
        )
        self.ids.graph.add_widget(self.graph)

        self.lineplot = LinePlot(color=[0, 1, 0, 1], line_width=1.5)
        self.graph.add_plot(self.lineplot)

        self.saved_previous = True
        self.popup_acquire = Popup(
            title="Confirmation",
            content=YesNoDialog(message='Previous data is not saved. Proceed?', yes=self._popup_yes_acquire, cancel=lambda :self.popup_acquire.dismiss()),
            size_hint=(0.4, 0.3)
        )
        self.popup_scan = Popup(
            title="Confirmation",
            content=YesNoDialog(message='Previous data is not saved. Proceed?', yes=self._popup_yes_scan, cancel=lambda :self.popup_scan.dismiss()),
            size_hint=(0.4, 0.3)
        )
        self.popup_save = Popup(
            title="Save file",
            content=SaveDialog(save=self.save, cancel=lambda :self.popup_save.dismiss(), folder=self.folder),
            size_hint=(0.9, 0.9)
        )

    def update(self, dt):
        if self.current_temperature == -80:
            self.finish_initialization()
            return
        self.current_temperature -= 1

    def initialize(self):
        self.ids.button_initialize.disabled = True
        self.initialize_event = Clock.schedule_interval(self.update, 0.1)

    def finish_initialization(self):
        Clock.unschedule(self.initialize_event)
        Clock.schedule_interval(self.update_graph, 0.5)
        Clock.schedule_interval(self.update_pos, 0.5)

    def update_graph(self, dt):
        self.xdata = np.linspace(0, 10, 100)
        self.ydata = np.sin(self.xdata + np.random.random(1))
        self.lineplot.points = [(t, temp) for t, temp in zip(self.xdata, self.ydata)]

    def update_pos(self, dt):
        self.current_pos = self.current_pos * dt

    def set_start(self):
        self.start_pos = self.current_pos

    def set_goal(self):
        self.goal_pos = self.current_pos

    def go(self, x, y, z):
        try:
            self.current_pos = np.array([x, y, z], float)
        except ValueError:
            print('invalid value')

    def set_integration(self, val):
        try:
            self.integration = float(val)
        except ValueError:
            print('invalid value')

    def set_accumulation(self, val):
        try:
            self.accumulation = int(val)
        except ValueError:
            print('invalid value')

    def set_interval(self, val):
        try:
            self.interval = float(val)
        except ValueError:
            print('invalid value')

    def start_acquire(self):
        if self.saved_previous:
            self.acquire()
            return
        self.popup_acquire.open()

    def start_scan(self):
        if self.saved_previous:
            self.scan()
            return
        self.popup_scan.open()

    def acquire(self):
        print('acquire')
        time.sleep(3)
        self.saved_previous = False

    def scan(self):
        print('scan')
        self.saved_previous = False

    def _popup_yes_acquire(self):
        # Start acquisition
        self.acquire()
        self.popup_acquire.dismiss()

    def _popup_yes_scan(self):
        # Start scan
        self.scan()
        self.popup_scan.dismiss()

    def show_save(self):
        self.popup_save.open()

    def save(self, path, filename):
        if not '.' in filename:
            filename += '.txt'

        with open(os.path.join(path, filename), 'w') as f:
            now = datetime.datetime.now()
            f.write(f'# time: {now.strftime("%Y-%m-%d-%H-%M")}\n')
            f.write(f'# integration: {self.integration}\n')
            f.write(f'# accumulation: {self.accumulation}\n')
            f.write(f'# coord: {self.coord}\n')
            for x, y in zip(self.xdata, self.ydata):
                f.write(f'{x},{y}\n')

        self.popup_save.dismiss()
        self.saved_previous = True
        self.popup_save.folder = path


class RASApp(App):
    def build(self):
        self.driver = RASDriver()
        return self.driver


if __name__ == '__main__':
    RASApp().run()