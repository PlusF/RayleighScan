from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import NumericProperty, StringProperty, ObjectProperty
from kivy.uix.popup import Popup
from kivy_garden.graph import Graph, LinePlot
from kivy.config import Config
Config.set('graphics', 'width', '1000')
Config.set('graphics', 'height', '500')
from kivy.core.window import Window, Clock

import os
if os.name == 'nt':
    from pyAndorSDK2 import atmcd, atmcd_codes, atmcd_errors
else:
    atmcd =  atmcd_codes = atmcd_errors = None
import numpy as np
import datetime
import time
import serial
import threading
from ConfigLoader import ConfigLoader
from HSC103Controller import HSC103Controller

UM_PER_PULSE = 0.01


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
    current_temperature = NumericProperty(0)
    start_pos = ObjectProperty(np.zeros(3), force_dispatch=True)
    current_pos = ObjectProperty(np.zeros(3), force_dispatch=True)
    goal_pos = ObjectProperty(np.zeros(3), force_dispatch=True)
    progress_acquire_value = ObjectProperty(0)
    progress_scan_value = ObjectProperty(0)
    integration = ObjectProperty(30)
    accumulation = ObjectProperty(1)
    interval = ObjectProperty(500)
    msg = StringProperty('')
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(on_request_close=self.quit)

        self.folder = './'

        self.cl = ConfigLoader('./config.json')
        if self.cl.mode == 'RELEASE':
            self.folder = self.cl.folder
            if not os.path.exists(self.folder):
                os.mkdir(self.folder)

        self.xdata = np.array([])
        self.ydata = np.array([])
        self.coord = np.array([])

        self.graph = Graph(
            xlabel = 'Wavelength [nm]', ylabel = 'Counts',
            xmin=0, xmax=1000, ymin=0, ymax=1000,
            x_ticks_minor = 2, x_ticks_major = 100, y_ticks_minor = 2, y_ticks_major = 100,
            y_grid_label = True, x_grid_label = True,
        )
        self.ids.graph.add_widget(self.graph)
        self.lineplot = LinePlot(color=[0, 1, 0, 1], line_width=1)
        self.graph.add_plot(self.lineplot)
        self.lineplot.points = [(i, i) for i in range(1000)]

        self.ids.button_acquire.disabled = True
        self.ids.button_scan.disabled = True
        self.ids.button_save.disabled = True

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

        self.open_ports()
        self.create_and_start_thread_position()

    def open_ports(self):
        if self.cl.mode == 'RELEASE':
            self.sdk = atmcd()
            self.ser = serial.Serial(self.cl.port, self.cl.baudrate, write_timeout=0)
            self.hsc = HSC103Controller(self.ser)
        elif self.cl.mode == 'DEBUG':
            self.sdk = None
            self.ser = None
            self.hsc = HSC103Controller(self.ser)
        else:
            raise ValueError('Error with config.json. mode must be DEBUG or RELEASE.')

    def create_and_start_thread_position(self):
        self.thread_pos = threading.Thread(target=self.update_position)
        self.thread_pos.daemon = True
        self.thread_pos.start()

    def create_and_start_thread_cool(self):
        self.thread_cool = threading.Thread(target=self.update_temperature)
        self.thread_cool.daemon = True
        self.thread_cool.start()

    def create_and_start_thread_acquire(self):
        self.prepare_acquisition()
        self.clear_things()
        self.num_pos = 1
        self.clock_acquire = Clock.schedule_interval(self.update_progress_acquire, 1)
        self.thread_acq = threading.Thread(target=self.acquire)
        self.thread_acq.daemon = True
        self.thread_acq.start()

    def create_and_start_thread_scan(self):
        self.num_pos = int(np.linalg.norm(self.goal_pos - self.start_pos) // self.interval + 1)
        if self.num_pos == 1:
            self.msg = 'Check the interval value again.'
            return
        self.prepare_acquisition()
        self.clear_things()
        self.progress_scan_value = 0
        self.prepare_scan()
        self.clock_acquire = Clock.schedule_interval(self.update_progress_acquire, 1)
        self.clock_scan = Clock.schedule_interval(self.update_progress_scan, 1)
        self.thread_scan = threading.Thread(target=self.scan)
        self.thread_scan.daemon = True
        self.thread_scan.start()

    def clear_things(self):
        self.ydata = np.empty([0, self.xpixels])
        self.coord = np.empty([0, 3])
        self.lineplot.points = []
        self.progress_acquire_value = 0

    def initialize(self):
        # 初期化
        if self.cl.mode == 'RELEASE':
            if self.sdk.Initialize('') == atmcd_errors.Error_Codes.DRV_SUCCESS:
                self.ids.button_initialize.disabled = True
                self.msg = 'Successfully initialized. Now cooling...'
                self.sdk.SetTemperature(self.cl.temperature)
                self.sdk.CoolerON()
            else:
                self.msg = 'Initialization failed.'
                return
        elif self.cl.mode == 'DEBUG':
            self.ids.button_initialize.disabled = True
        self.create_and_start_thread_cool()

        self.ids.button_acquire.disabled = False
        self.ids.button_scan.disabled = False

        if self.cl.mode == 'RELEASE':
            ret, self.xpixels, ypixels = self.sdk.GetDetector()
            self.sdk.handle_return(ret)
        elif self.cl.mode == 'DEBUG':
            self.xpixels = 1024

    def update_temperature(self):
        if self.cl.mode == 'RELEASE':
            while True:
                ret, self.current_temperature = self.sdk.GetTemperature()
                if ret == atmcd_errors.Error_Codes.DRV_TEMP_STABILIZED:
                    break
                time.sleep(self.cl.dt * 0.001)
        elif self.cl.mode == 'DEBUG':
            self.current_temperature = self.cl.temperature
        self.msg = 'Cooling finished.'

    def update_graph(self):
        self.xdata = np.linspace(0, 10, self.xpixels)
        self.graph.xmin = float(np.min(self.xdata))
        self.graph.xmax = float(np.max(self.xdata))
        self.graph.ymin = float(np.min(self.ydata.sum(axis=0)))
        self.graph.ymax = float(np.max(self.ydata.sum(axis=0)))
        self.lineplot.points = [(x, y) for x, y in zip(self.xdata, self.ydata.sum(axis=0))]

    def update_position(self):
        while True:
            if self.cl.mode == 'RELEASE':
                self.current_pos = np.array(self.hsc.get_position())
            elif self.cl.mode == 'DEBUG':
                pass
            time.sleep(self.cl.dt * 0.001)

    def set_start(self):
        self.start_pos = self.current_pos

    def set_goal(self):
        self.goal_pos = self.current_pos

    def go(self, x, y, z):
        try:
            self.current_pos = np.array([x, y, z], float)
        except ValueError:
            print('invalid value.')
            return

        pos = (np.array([x, y, z], float) - self.current_pos) / UM_PER_PULSE
        if self.cl.mode == 'RELEASE':
            self.hsc.move_linear(pos)
        elif self.cl.mode == 'DEBUG':
            pass

    def set_integration(self, val):
        try:
            self.integration = float(val)
        except ValueError:
            self.msg = 'Invalid value.'

    def set_accumulation(self, val):
        try:
            self.accumulation = int(val)
        except ValueError:
            self.msg = 'Invalid value.'

    def set_interval(self, val):
        try:
            self.interval = float(val)
        except ValueError:
            self.msg = 'invalid value.'

    def start_acquire(self):
        if self.saved_previous:
            self.create_and_start_thread_acquire()
            return
        self.popup_acquire.open()

    def start_scan(self):
        if self.saved_previous:
            self.create_and_start_thread_scan()
            return
        self.popup_scan.open()

    def prepare_acquisition(self):
        if self.cl.mode == 'RELEASE':
            self.sdk.handle_return(self.sdk.SetAcquisitionMode(atmcd_codes.Acquisition_Mode.SINGLE_SCAN))
            self.sdk.handle_return(self.sdk.SetReadMode(atmcd_codes.Read_Mode.FULL_VERTICAL_BINNING))
            self.sdk.handle_return(self.sdk.SetTriggerMode(atmcd_codes.Trigger_Mode.INTERNAL))
            self.sdk.handle_return(self.sdk.SetExposureTime(self.integration))  # TODO: 露光時間入力の例外処理
            self.sdk.handle_return(self.sdk.PrepareAcquisition())
        elif self.cl.mode == 'DEBUG':
            print('prepare acquisition')

    def acquire(self, stop_clock=True):
        for i in range(self.accumulation):
            if self.cl.mode == 'RELEASE':
                self.sdk.handle_return(self.sdk.StartAcquisition())
                self.sdk.handle_return(self.sdk.WaitForAcquisition())
                ret, spec, first, last = self.sdk.GetImages16(1, 1, self.xpixels)
                self.ydata = np.append(self.ydata, np.array([spec]), axis=0)
                self.sdk.handle_return(ret)
            elif self.cl.mode == 'DEBUG':
                time.sleep(self.integration)
                print(f'acquired {i}')
                spec = np.sin(np.linspace(0, np.random.random() * 10, self.xpixels))
                self.ydata = np.append(self.ydata, np.array([spec]), axis=0)
            if len(self.ydata) == 0:
                raise ValueError('something went wrong')
            self.coord = np.append(self.coord, self.current_pos.reshape([1, 3]), axis=0)
            self.update_graph()

        if stop_clock:
            Clock.unschedule(self.clock_acquire)
        self.progress_acquire_value = 1
        self.saved_previous = False
        self.ids.button_save.disabled = False

    def prepare_scan(self):
        # init
        self.hsc.set_speed_max()

        # move to start position
        self.hsc.move_abs(self.start_pos / UM_PER_PULSE)
        distance = np.linalg.norm(np.array(self.current_pos - self.start_pos) / UM_PER_PULSE)
        time.sleep(distance / 40000 + 1)  # TODO: 到着を確認してから次に進む

    def scan(self):
        start = self.start_pos / UM_PER_PULSE
        goal = self.goal_pos / UM_PER_PULSE
        number = 1
        while number <= self.num_pos:
            time_left = np.ceil((self.num_pos - number + 1) * self.integration * 2  * self.accumulation / 60)
            self.msg = f'Acquisition {number} of {self.num_pos}... {time_left} minutes left.'

            point = start + (goal - start) * (number - 1) / (self.num_pos - 1)
            self.hsc.move_abs(point)
            distance = np.linalg.norm(np.array(point - start))
            time.sleep(distance / 40000 + 1)  # TODO: 到着を確認してから次に進む

            self.acquire(stop_clock=False)
            number += 1

        self.progress_acquire_value = 1
        self.progress_scan_value = 1
        Clock.unschedule(self.clock_acquire)
        Clock.unschedule(self.clock_scan)
        self.saved_previous = False
        self.ids.button_save.disabled = False
        self.msg = 'Scan finished.'

    def update_progress_acquire(self, dt):
        self.progress_acquire_value += 1 / self.integration / self.accumulation / 1.3  # なんとなく
        if self.progress_acquire_value > 1:
            self.progress_acquire_value -= 1
    def update_progress_scan(self, dt):
        self.progress_scan_value += 1 / self.integration / self.accumulation / self.num_pos / 1.3  # なんとなく

    def _popup_yes_acquire(self):
        # Start acquisition
        self.create_and_start_thread_acquire()
        self.popup_acquire.dismiss()

    def _popup_yes_scan(self):
        # Start scan
        self.create_and_start_thread_scan()
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
            f.write(f'# interval: {self.interval}\n')
            f.write(f'index,pos_x,pos_y,pos_z,{",".join(self.xdata.astype(str))}\n')
            for i, (pos, y) in enumerate(zip(self.coord.astype(str), self.ydata.astype(str))):
                f.write(f'{",".join([str(i),*pos,*y])}\n')

        self.popup_save.dismiss()
        self.saved_previous = True
        self.popup_save.folder = path

    def quit(self, instance):
        if self.cl.mode == 'RELEASE':
            self.sdk.ShutDown()
            self.ser.close()


class RASApp(App):
    def build(self):
        self.driver = RASDriver()
        return self.driver


if __name__ == '__main__':
    RASApp().run()